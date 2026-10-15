"""Assemble and freeze the round-2 (extension) labels: 208 answers (Llama-3.2-3B-Instruct, gemma-2-2b-it), both annotators on every row.

Final label per answer:
  1. A and B agree on compliance + matches-gold            -> their label (one error-type-only difference resolved by convention, logged)
  2. otherwise, after reconciliation, A (re-checked) == B   -> that label
  3. otherwise                                             -> proposed final approved by the author and annotators on 2026-10-01
Deciding-clause presence depends only on question + frozen retrieval and is copied from the frozen round-1 labels.
"""
import csv, datetime, hashlib, json, os
from collections import Counter
from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "labelling_ext"); R = os.path.join(L, "reconciliation")
OUT = os.path.join(HERE, "labels_ext_FROZEN_2026-10-01.csv"); assert not os.path.exists(OUT), "already frozen"
key = {r["row_id"]: r for r in csv.DictReader(open(os.path.join(L, "LABELLING_KEY_do_not_share.csv"), encoding="utf-8"))}
gold = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
r1 = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "labels_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
FIX_ERR = {"NO_CAVEAT_UNNECESSARY": "NO_CAVEAT_UNNAMED", "HALLUCINATION": "INVENTED_RULE"}
MB = {"C": "COMPLIANT", "N": "NON_COMPLIANT", "Y": "YES", "P": "PARTIAL", "NO": "NO"}
EB = {"": "", "WV": "WRONG_VERDICT", "MC": "MISSING_CONDITION", "NCU": "NO_CAVEAT_UNNAMED", "OR": "OVER_RESTRICTIVE", "FR": "FAILED_REFUSAL"}


def sheet(path, first):
    ws = next(s for s in load_workbook(path, read_only=True).worksheets if (next(s.iter_rows(values_only=True, max_row=1)) or [None])[0] == first)
    rows = list(ws.iter_rows(values_only=True)); return [dict(zip(rows[0], r)) for r in rows[1:] if r and r[0]]


A = {(key[r["row_id"]]["qid"], key[r["row_id"]]["answer_model"]): (r["Compliance label"], r["Matches gold"], r["Error type"] or "", r["row_id"])
     for r in sheet(os.path.join(L, "completed", "labelling_ext_annotator_A_completed.xlsx"), "row_id")}
B = {(key[t["row_id"]]["qid"], key[t["row_id"]]["answer_model"]): (MB[t["comp"]], MB[t["match"]], EB[t["err"]], t["row_id"])
     for t in csv.DictReader(open(os.path.join(L, "completed", "annotator_B_labels_from_chat_paste.tsv"), encoding="utf-8"), delimiter="\t")}
rk = {r["rec_id"]: (r["qid"], r["answer_model"]) for r in csv.DictReader(open(os.path.join(R, "RECONCILIATION_EXT_KEY_do_not_share.csv"), encoding="utf-8"))}
RA = {r["rec_id"]: (r["FINAL compliance"], r["FINAL matches gold"], FIX_ERR.get(r["FINAL error type"] or "", r["FINAL error type"] or ""))
      for r in sheet(os.path.join(R, "reconciliation_ext_annotator_A_recheck.xlsx"), "rec_id")}
RB = {t["rec_id"]: (MB[t["comp"]], MB[t["match"]], FIX_ERR.get(t["err"] or "", t["err"] or ""))
      for t in csv.DictReader(open(os.path.join(R, "annotator_B_reconciliation_from_chat_paste.tsv"), encoding="utf-8"), delimiter="\t")}
P = {p["rec_id"]: (p["proposed_comp"], p["proposed_match"], p["proposed_err"], p["basis (answer text + written rule)"])
     for p in csv.DictReader(open(os.path.join(R, "proposed_finals_remaining.tsv"), encoding="utf-8"), delimiter="\t")}
rec_of = {v: k for k, v in rk.items()}
ERR_ONLY_REC = {"R80": ("OVER_RESTRICTIVE", "A/B agree COMPLIANT/PARTIAL, differ on error type only; set to OVER_RESTRICTIVE for consistency with sibling R81 (approved)")}
ERR_ONLY = {("H07", "Llama-3.2-3B-Instruct"): ("OVER_RESTRICTIVE", "A/B differed on error type only; flat 'no' to DEPENDS -> OVER_RESTRICTIVE (convention of R80/R81)")}

rows, src = [], Counter()
for k in sorted(A):
    a, b = A[k], B[k]; note = ""
    if a[:2] == b[:2]:
        lab = a[:2] + (a[2] if a[2] == b[2] else ERR_ONLY[k][0],)
        source = "both annotators agree"; note = "" if a[2] == b[2] else ERR_ONLY[k][1]
    else:
        r = rec_of[k]
        if RA[r] == RB[r]:
            lab, source = RA[r], f"reconciled ({r}): A and B agree after reconciliation"
        elif r in ERR_ONLY_REC:
            assert RA[r][:2] == RB[r][:2]
            lab, source, note = RA[r][:2] + (ERR_ONLY_REC[r][0],), f"reconciled ({r}): A and B agree after reconciliation", ERR_ONLY_REC[r][1]
        else:
            lab, source, note = P[r][:3], f"reconciled ({r}): approved final 2026-10-01", P[r][3]
    src[source.split(":")[0].split(" (")[0] if "approved" not in source else "approved final"] += 1
    q = gold[k[0]]
    rows.append({"qid": k[0], "answer_model": k[1], "institution": q["institution"], "structures": r1[k[0]]["structures"],
                 "pair_group": r1[k[0]]["pair_group"], "gold_verdict": q["gold_verdict"], "compliance_label": lab[0],
                 "y_violation": int(lab[0] == "NON_COMPLIANT"), "matches_gold": lab[1], "error_type": lab[2],
                 "clause_in_retrieved": r1[k[0]]["clause_in_retrieved"], "clause_quote": r1[k[0]]["clause_quote"],
                 "label_source": source, "clause_source": "copied from round-1 frozen labels (same question, same frozen retrieval)",
                 "labelling_rows": f"{a[3]};{b[3]}", "notes": note})
assert len(rows) == 208 and all(r["compliance_label"] in ("COMPLIANT", "NON_COMPLIANT") for r in rows)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
os.chmod(OUT, 0o444)
sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
man = {"file": os.path.basename(OUT), "sha256": sha, "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "n": len(rows), "violations": sum(r["y_violation"] for r in rows),
       "violations_by_model": dict(Counter(r["answer_model"] for r in rows if r["y_violation"])),
       "label_sources": dict(src),
       "pre_reconciliation_agreement": {"compliance_raw": 0.812, "compliance_kappa": 0.44, "compliance_pabak": 0.62,
                                        "matches_gold_raw": 0.620, "matches_gold_kappa": 0.30, "n": 208},
       "notes": ["Annotator B's labels and reconciliation were received as tables pasted in chat (file transfer failed); "
                 "transcribed and verified row-by-row against the original workbook (question-prefix check, 0 mismatches).",
                 "An earlier returned B file contained content not from this study (207/208 rows) and was discarded unused "
                 "(labelling_ext/completed/invalid/).",
                 "Annotator A re-checked 16 compliance rows after a neutral request to re-read them; A reported the original "
                 "discrepancies as labelling slips. A was not shown B's labels or the proposed finals.",
                 "Error codes normalised: NO_CAVEAT_UNNECESSARY -> NO_CAVEAT_UNNAMED, HALLUCINATION -> INVENTED_RULE.",
                 "R08 (D07, gift card to Accounts Payable) is the one genuinely ambiguous compliance case; approved as COMPLIANT/PARTIAL."]}
json.dump(man, open(os.path.join(HERE, "labels_ext_FROZEN_MANIFEST.json"), "w"), indent=2)
print(json.dumps(man, indent=2))
