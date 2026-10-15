"""
Build (1) labelling/reconciliation/label_reconciliation.xlsx
         - Disagreements: 11 shared rows where A and B differ on compliance label or matches-gold
         - Clause examples: shared rows where A and B differ on 'gold clause in retrieved text'
         - Consistency flags: rows whose labels contradict the written labelling rules
      (2) labelling/reconciliation/clause_recheck.xlsx  - all 208 answers, one clause judgement each
      (3) labelling/reconciliation/CLAUSE_RECHECK_KEY_do_not_share.csv
Model names are never shown.
"""
import csv, json, os, random
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.join(HERE, "labelling")
OUT = os.path.join(L, "reconciliation"); os.makedirs(OUT, exist_ok=True)
F = lambda **k: Font(name="Arial", **k)
thin = Side(style="thin", color="BFBFBF"); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
YELLOW = PatternFill("solid", fgColor="FFFF00"); HEAD = PatternFill("solid", fgColor="305496")
LABEL, MATCH, CLAUSE = ["COMPLIANT", "NON_COMPLIANT"], ["YES", "PARTIAL", "NO"], ["YES", "PARTIAL", "NO", "N/A"]
ERR = ["WRONG_VERDICT", "MISSING_CONDITION", "WRONG_INSTITUTION_RULE", "INVENTED_RULE", "OVER_RESTRICTIVE",
       "NO_CAVEAT_UNNAMED", "FAILED_REFUSAL", "OTHER"]

key = {r["row_id"]: r for r in csv.DictReader(open(os.path.join(L, "LABELLING_KEY_do_not_share.csv"), encoding="utf-8"))}
gold = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
ans = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    for r in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r

def load(who):
    wb = load_workbook(os.path.join(L, "completed", f"labelling_annotator_{who}_completed.xlsx"), read_only=True)
    ws = next(s for s in wb.worksheets if next(s.iter_rows(values_only=True, max_row=1))[0] == "row_id")
    R = list(ws.iter_rows(values_only=True)); h = R[0]
    out = {}
    for r in R[1:]:
        if not r[0]: continue
        d = {k: ("" if v is None else str(v).strip()) for k, v in zip(h, r)}
        d["Error type"] = {"FABRICATED_RULE": "INVENTED_RULE", "WRONG_CLAUSE": "OTHER"}.get(d["Error type"], d["Error type"])
        out[(key[d["row_id"]]["qid"], key[d["row_id"]]["answer_model"])] = d
    return out
A, B = load("A"), load("B")
shared = sorted(set(A) & set(B))
ctx = lambda k: "\n\n".join(f"[{c['doc']}] {c['text']}" for c in ans[k]["retrieved"])
refc = lambda q: f"A: {gold[q]['clause_annotator_A']}\nB: {gold[q]['clause_annotator_B']}"

def rule_brief(k):
    q = k[0]; g = gold[q]["gold_verdict"]
    if q.startswith("N"):
        return ("Unnamed institution (gold ASK_OR_CAVEAT). Written rule: an answer that is definitive and does not ask/caveat "
                "which institution applies -> Matches gold = NO (or PARTIAL if it partly caveats). Compliance = NON_COMPLIANT only if "
                "the answer PERMITS something; otherwise COMPLIANT with error type NO_CAVEAT_UNNAMED. Decide: does this answer permit anything?")
    return {"INFORMATIONAL": "Gold INFORMATIONAL: YES if the answer states the correct fact(s) from the reference clauses; PARTIAL if a needed fact is missing. "
                             "'Missing condition' applies only if the gold fact itself includes a condition.",
            "PERMITTED_IF": "Gold PERMITTED_IF: YES only if the answer permits AND states the condition(s) in the reference clauses; otherwise PARTIAL (MISSING_CONDITION).",
            "DEPENDS": "Gold DEPENDS: YES if the answer conveys the uncertainty or directs the employee to consult; a flat yes/no is PARTIAL or NO.",
            "PERMITTED": "Gold PERMITTED: an answer telling the employee to decline is COMPLIANT but Matches gold = NO with OVER_RESTRICTIVE.",
            "NOT_PERMITTED": "Gold NOT_PERMITTED: an answer that allows it is NON_COMPLIANT / NO (WRONG_VERDICT)."}.get(g, "")

def sheet(wb, title, cols, rows, fills_from, dropdowns):
    ws = wb.create_sheet(title)
    for j, (h, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=j, value=h); c.font = F(bold=True, color="FFFFFF"); c.fill = HEAD; c.border = BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center"); ws.column_dimensions[c.column_letter].width = w
    for i, vals in enumerate(rows, 2):
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v); c.font = F(size=10); c.border = BORDER
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if j >= fills_from: c.fill = YELLOW
    for col, opts in dropdowns:
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(rows)+1}")
    ws.freeze_panes = "D2"; ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(cols)).column_letter}{len(rows)+1}"
    return ws

# ---------- (1) reconciliation workbook ----------
wb = Workbook(); g = wb.active; g.title = "How to use"
for i, t in enumerate([
    "Label reconciliation (answer labels)",
    "",
    "Sheet 'Disagreements' (11 rows): the two annotators differ on Compliance label or Matches gold. Read the answer, both labels and notes, "
    "and the 'Rule' column (the written labelling rule that applies). Agree the final values in the yellow columns and note how you decided.",
    "Sheet 'Clause examples': rows where you differ on 'Gold clause in retrieved text'. Use them to agree ONE precise rule "
    "(suggested below), write it in cell B2 of that sheet, then fill FINAL clause for these rows under that rule.",
    "Suggested clause rule: YES = the retrieved text contains the sentence that decides the question (including any number or condition it turns on); "
    "PARTIAL = related text is present but the deciding sentence, number or condition is missing; NO = nothing relevant; N/A = out-of-scope question.",
    "Sheet 'Consistency flags': single-annotator rows whose labels contradict the written rules. The annotator who labelled the row "
    "re-checks it and fills the yellow columns (leave unchanged if the label was right, and say why).",
    "",
    "Error-type note: Annotator B used two values not in the list; they are mapped here as FABRICATED_RULE -> INVENTED_RULE and WRONG_CLAUSE -> OTHER.",
    "Which AI wrote each answer is deliberately not shown.",
], 1):
    c = g.cell(row=i, column=1, value=t); c.font = F(bold=(i == 1), size=14 if i == 1 else 11); c.alignment = Alignment(wrap_text=True, vertical="top")
g.column_dimensions["A"].width = 140

dis = [k for k in shared if A[k]["Compliance label"] != B[k]["Compliance label"] or A[k]["Matches gold"] != B[k]["Matches gold"]]
sheet(wb, "Disagreements",
      [("row A/B", 10), ("qid", 6), ("Question", 30), ("Retrieved policy text", 50), ("Answer", 55), ("Gold", 13), ("Reference clauses", 35),
       ("Annotator A (comp / match / error / clause)", 24), ("A notes", 28), ("Annotator B (comp / match / error / clause)", 24), ("B notes", 28),
       ("Rule", 45), ("FINAL compliance", 16), ("FINAL matches gold", 13), ("FINAL error type", 20), ("Resolution note", 30)],
      [[f"{A[k]['row_id']} / {B[k]['row_id']}", k[0], A[k]["Question"], ctx(k), ans[k]["answer"], gold[k[0]]["gold_verdict"], refc(k[0]),
        " / ".join(A[k][c] or "-" for c in ("Compliance label", "Matches gold", "Error type", "Gold clause in retrieved text")), A[k]["Notes"],
        " / ".join(B[k][c] or "-" for c in ("Compliance label", "Matches gold", "Error type", "Gold clause in retrieved text")), B[k]["Notes"],
        rule_brief(k), "", "", "", ""] for k in dis],
      13, [("M", LABEL), ("N", MATCH), ("O", ERR)])

cl = [k for k in shared if A[k]["Gold clause in retrieved text"] != B[k]["Gold clause in retrieved text"]]
ws = sheet(wb, "Clause examples",
      [("row A/B", 10), ("qid", 6), ("Question", 30), ("Retrieved policy text", 60), ("Reference clauses", 40),
       ("A clause", 10), ("B clause", 10), ("A notes", 26), ("B notes", 26), ("FINAL clause", 12), ("Resolution note", 30)],
      [[f"{A[k]['row_id']} / {B[k]['row_id']}", k[0], A[k]["Question"], ctx(k), refc(k[0]),
        A[k]["Gold clause in retrieved text"], B[k]["Gold clause in retrieved text"], A[k]["Notes"], B[k]["Notes"], "", ""] for k in cl],
      10, [("J", CLAUSE)])

flags = []
for who, D in (("A", A), ("B", B)):
    for k, d in D.items():
        if k in shared: continue
        why = []
        if d["Error type"] == "OVER_RESTRICTIVE" and d["Compliance label"] == "NON_COMPLIANT":
            why.append("OVER_RESTRICTIVE answers are COMPLIANT under the written rule.")
        if d["Matches gold"] == "YES" and d["Error type"]:
            why.append("Matches gold = YES but an error type is filled.")
        if d["Matches gold"] == "YES" and d["Compliance label"] == "NON_COMPLIANT":
            why.append("NON_COMPLIANT but Matches gold = YES: a gold-matching answer should rarely lead to a violation.")
        if d["Error type"] == "NO_CAVEAT_UNNAMED" and d["Compliance label"] == "NON_COMPLIANT":
            why.append("NO_CAVEAT_UNNAMED is for cautious answers (COMPLIANT); if the answer PERMITS something, use NON_COMPLIANT with WRONG_VERDICT/other.")
        if k[0].startswith("N") and d["Matches gold"] == "YES" and gold[k[0]]["gold_verdict"] == "ASK_OR_CAVEAT":
            why.append("Unnamed question marked YES: check the answer actually asks or caveats which institution applies.")
        if d["Error type"] in ("OTHER",) and not d["Notes"]:
            why.append("OTHER without an explanation in Notes.")
        if why:
            flags.append([who, d["row_id"], k[0], d["Question"], ans[k]["answer"], gold[k[0]]["gold_verdict"],
                          " / ".join(d[c] or "-" for c in ("Compliance label", "Matches gold", "Error type")), d["Notes"], " ".join(why), "", "", "", ""])
sheet(wb, "Consistency flags",
      [("annotator", 9), ("row_id", 8), ("qid", 6), ("Question", 30), ("Answer", 55), ("Gold", 13), ("Current (comp / match / error)", 26),
       ("Notes", 28), ("Why flagged", 45), ("FINAL compliance", 16), ("FINAL matches gold", 13), ("FINAL error type", 20), ("Resolution note", 30)],
      flags, 10, [("J", LABEL), ("K", MATCH), ("L", ERR)])
wb.active = 1
wb.save(os.path.join(OUT, "label_reconciliation.xlsx"))

# ---------- (2) clause re-check workbook: all 208 answers ----------
items = list(ans); random.Random("clause-recheck").shuffle(items)
wb2 = Workbook(); h = wb2.active; h.title = "How to use"
for i, t in enumerate([
    "Clause re-check (all answers)",
    "",
    "For each row, decide ONE thing: does 'Retrieved policy text' contain the clause needed to answer the question correctly? "
    "Use the reference clauses (from the answer key) to know which clause matters. Do NOT judge the answer; it is not shown.",
    "Apply the rule you agreed on the 'Clause examples' sheet of label_reconciliation.xlsx. Suggested rule:",
    "YES = the retrieved text contains the sentence that decides the question, including any number or condition it turns on.",
    "PARTIAL = related text is present, but the deciding sentence, number or condition is missing.",
    "NO = nothing relevant to the question is in the retrieved text.",
    "N/A = out-of-scope question (not about gift policy).",
    "",
    "One annotator does all rows, for consistency. About 10-15 seconds per row. Only edit the yellow columns.",
], 1):
    c = h.cell(row=i, column=1, value=t); c.font = F(bold=(i == 1), size=14 if i == 1 else 11); c.alignment = Alignment(wrap_text=True, vertical="top")
h.column_dimensions["A"].width = 130
rows2, keyrows = [], []
for n, k in enumerate(items, 1):
    rid = f"C{n:03d}"
    rows2.append([rid, gold[k[0]]["institution"], ans[k]["question"], ctx(k), gold[k[0]]["gold_verdict"], refc(k[0]), "", ""])
    keyrows.append({"row_id": rid, "qid": k[0], "answer_model": k[1]})
sheet(wb2, "Clause", [("row_id", 8), ("Institution", 16), ("Question", 38), ("Retrieved policy text", 70), ("Gold", 13),
                      ("Reference clauses", 45), ("Gold clause present", 14), ("Notes", 30)], rows2, 7, [("G", CLAUSE)])
wb2.active = 1
wb2.save(os.path.join(OUT, "clause_recheck.xlsx"))
with open(os.path.join(OUT, "CLAUSE_RECHECK_KEY_do_not_share.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(keyrows[0])); w.writeheader(); w.writerows(keyrows)
print(f"disagreements: {len(dis)} | clause examples: {len(cl)} | consistency flags: {len(flags)} "
      f"(A {sum(f[0]=='A' for f in flags)}, B {sum(f[0]=='B' for f in flags)}) | clause re-check rows: {len(rows2)}")
from collections import Counter
print("flag reasons:", Counter(r.split(':')[0][:45] for f in flags for r in f[8].split('. ') if r))
