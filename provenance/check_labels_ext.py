"""Validate the round-2 (extension) labelling workbooks and compute agreement. Every one of the 208 answers is labelled by both."""
import csv, os
from collections import Counter
from openpyxl import load_workbook

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root; L = os.path.join(HERE, "labelling_ext"); C = os.path.join(L, "completed")
VALID = {"Compliance label": {"COMPLIANT", "NON_COMPLIANT"}, "Matches gold": {"YES", "PARTIAL", "NO"},
         "Error type": {"", "WRONG_VERDICT", "MISSING_CONDITION", "WRONG_INSTITUTION_RULE", "INVENTED_RULE",
                        "OVER_RESTRICTIVE", "NO_CAVEAT_UNNAMED", "FAILED_REFUSAL", "OTHER"}}
key = {r["row_id"]: r for r in csv.DictReader(open(os.path.join(L, "LABELLING_KEY_do_not_share.csv"), encoding="utf-8"))}


def read(path):
    wb = load_workbook(path, read_only=True)
    ws = next(s for s in wb.worksheets if (next(s.iter_rows(values_only=True, max_row=1)) or [None])[0] == "row_id")
    rows = list(ws.iter_rows(values_only=True)); hdr = [h for h in rows[0]]
    return ws.title, hdr, [{h: ("" if v is None else str(v).strip()) for h, v in zip(hdr, r)} for r in rows[1:] if r and r[0]]


MAP = {"comp": {"C": "COMPLIANT", "N": "NON_COMPLIANT"}, "match": {"Y": "YES", "P": "PARTIAL", "NO": "NO"},
       "err": {"": "", "WV": "WRONG_VERDICT", "MC": "MISSING_CONDITION", "NCU": "NO_CAVEAT_UNNAMED", "OR": "OVER_RESTRICTIVE",
               "FR": "FAILED_REFUSAL", "WIR": "WRONG_INSTITUTION_RULE", "IR": "INVENTED_RULE", "OT": "OTHER"}}
B_TSV = os.path.join(C, "annotator_B_labels_from_chat_paste.tsv")   # B's workbook could not be downloaded; labels pasted in chat


def read_b_tsv():
    import re
    _, _, orig = read(os.path.join(L, "labelling_ext_annotator_B.xlsx"))
    o = {r["row_id"]: r for r in orig}
    norm = lambda t: re.sub(r"[^a-z0-9$]", "", t.lower().replace("’", "'"))
    rows, bad = [], []
    for t in csv.DictReader(open(B_TSV, encoding="utf-8"), delimiter="	"):
        if not norm(o[t["row_id"]]["Question"]).startswith(norm(t["question_prefix"])): bad.append(t["row_id"])
        rows.append({**o[t["row_id"]], "Compliance label": MAP["comp"][t["comp"]], "Matches gold": MAP["match"][t["match"]],
                     "Error type": MAP["err"][t["err"] or ""], "Notes": t["notes"] or ""})
    print(f"B from chat paste: {len(rows)} rows, ids complete: {sorted(r['row_id'] for r in rows) == sorted(o)}, "
          f"question-prefix mismatches: {bad}")
    return rows


labels = {}
for who in ("A", "B"):
    if who == "B" and os.path.exists(B_TSV):
        rows = read_b_tsv()
        problems = [(r["row_id"], "Error type") for r in rows if (r["Matches gold"] in ("PARTIAL", "NO")) != bool(r["Error type"])]
        print("validation problems:", problems)
        for col in ("Compliance label", "Matches gold", "Error type"):
            print(f"  {col}: {dict(Counter(r.get(col) or '(blank)' for r in rows))}")
        labels[who] = {(key[r["row_id"]]["qid"], key[r["row_id"]]["answer_model"]): r for r in rows}
        continue
    _, hdr0, orig = read(os.path.join(L, f"labelling_ext_annotator_{who}.xlsx"))
    sheet, hdr, rows = read(os.path.join(C, f"labelling_ext_annotator_{who}_completed.xlsx"))
    print(f"\n=== Annotator {who}: sheet '{sheet}', {len(rows)} rows (original {len(orig)}) ===")
    print("header unchanged:", hdr == hdr0, "" if hdr == hdr0 else f"\n  original {hdr0}\n  returned {hdr}")
    o = {r["row_id"]: r for r in orig}
    ids = [r["row_id"] for r in rows]
    print("rows missing:", sorted(set(o) - set(ids))[:10], "| unknown/duplicate ids:", [i for i in ids if i not in o][:5], len(ids) - len(set(ids)))
    altered = [r["row_id"] for r in rows if r["row_id"] in o and any(r.get(c, "") != o[r["row_id"]][c] for c in ("Question", "Answer", "Gold verdict (reference)"))]
    print("read-only columns altered:", len(altered), altered[:5])
    problems = []
    for r in rows:
        for col, ok in VALID.items():
            v = r.get(col, "")
            if col != "Error type" and not v: problems.append((r["row_id"], col, "blank"))
            elif v not in ok: problems.append((r["row_id"], col, v))
        if r.get("Matches gold") in ("PARTIAL", "NO") and not r.get("Error type"):
            problems.append((r["row_id"], "Error type", "missing although Matches gold is " + r["Matches gold"]))
        if r.get("Matches gold") == "YES" and r.get("Error type"):
            problems.append((r["row_id"], "Error type", "filled although Matches gold is YES: " + r["Error type"]))
    print("validation problems:", len(problems)); [print("  ", p) for p in problems[:15]]
    for col in ("Compliance label", "Matches gold", "Error type"):
        print(f"  {col}: {dict(Counter(r.get(col) or '(blank)' for r in rows))}")
    print("  notes filled:", sum(bool(r.get("Notes")) for r in rows))
    labels[who] = {(key[r["row_id"]]["qid"], key[r["row_id"]]["answer_model"]): r for r in rows if r["row_id"] in key}


def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


shared = sorted(set(labels["A"]) & set(labels["B"]))
print(f"\n=== Agreement on {len(shared)} answers ===")
for col in ("Compliance label", "Matches gold", "Error type"):
    pairs = [(labels["A"][k][col] or "-", labels["B"][k][col] or "-") for k in shared]
    po, k = kappa(pairs); extra = f"  PABAK {2 * po - 1:.2f}" if col == "Compliance label" else ""
    print(f"  {col:18} raw {po:.1%}  kappa {k:.2f}{extra}")
for m in ("Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    ks = [k for k in shared if k[1] == m]
    print(f"  {m}: violations A {sum(labels['A'][k]['Compliance label'] == 'NON_COMPLIANT' for k in ks)}/{len(ks)}, "
          f"B {sum(labels['B'][k]['Compliance label'] == 'NON_COMPLIANT' for k in ks)}/{len(ks)}, "
          f"both {sum(labels['A'][k]['Compliance label'] == labels['B'][k]['Compliance label'] == 'NON_COMPLIANT' for k in ks)}")
dis = [k for k in shared if labels["A"][k]["Compliance label"] != labels["B"][k]["Compliance label"]
       or labels["A"][k]["Matches gold"] != labels["B"][k]["Matches gold"]]
dc = [k for k in dis if labels["A"][k]["Compliance label"] != labels["B"][k]["Compliance label"]]
print(f"\ndisagreements: compliance {len(dc)}, compliance-or-matches {len(dis)}")
print("compliance disagreements by direction:", dict(Counter(labels['A'][k]['Compliance label'][:6] + '/' + labels['B'][k]['Compliance label'][:6] for k in dc)))
with open(os.path.join(C, "disagreements_ext.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "answer_model", "row_id_A", "row_id_B", "compliance_A", "compliance_B", "matches_A", "matches_B",
                "error_A", "error_B", "notes_A", "notes_B"])
    for k in dis:
        a, b = labels["A"][k], labels["B"][k]
        w.writerow([k[0], k[1], a["row_id"], b["row_id"], a["Compliance label"], b["Compliance label"], a["Matches gold"],
                    b["Matches gold"], a["Error type"], b["Error type"], a.get("Notes", ""), b.get("Notes", "")])
