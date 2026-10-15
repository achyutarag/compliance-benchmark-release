"""Adjudication sheet for the deciding-clause label (round-1 vs annotator 3, 23 disagreements).
Tab 'Adjudicate': 9 items the automated coverage check could not settle (8 unsure + M05, where it sided with round 1).
Tab 'Auto-adopted': 14 items where the automated check sided with annotator 3 (shown for transparency; no action unless you disagree).
No answers and no model names are shown."""
import csv, json, os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "labelling", "clause_recheck2", "adjudication_sheet.xlsx")
K = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
ret = {json.loads(l)["qid"]: json.loads(l)["chunk_ids"] for l in open(os.path.join(HERE, "index", "retrieval_frozen.jsonl"), encoding="utf-8")}
ch = {json.loads(l)["chunk_id"]: json.loads(l) for l in open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8")}
r1 = {}
for r in csv.DictReader(open(os.path.join(HERE, "labels_FROZEN_2026-09-30.csv"), encoding="utf-8")): r1.setdefault(r["qid"], r)
a3 = {}
key = {r["row_id"]: r["qid"] for r in csv.DictReader(open(os.path.join(HERE, "labelling", "clause_recheck2", "CLAUSE_RECHECK2_KEY_do_not_share.csv"), encoding="utf-8"))}
for r in csv.DictReader(open(os.path.join(HERE, "labelling", "clause_recheck2", "clause_recheck2_annotator3_returned.csv"), encoding="utf-8")): a3[key[r["Row ID"]]] = r
# Auto-adopted = round-1 NO -> annotator-3 YES where the cited clause text is verbatim in the retrieved text (objective check).
# PARTIAL-vs-NO cases (C03, D09, T03, T11, T14) were first auto-adopted by mistake: the coverage check cannot tell PARTIAL from NO,
# and D09 contradicts the agreed worked example (general meals exception present, 'occasional and non-repetitive' missing = PARTIAL).
auto_adopt = ["C06", "H08", "T01", "T06", "U04", "U10", "U11", "U14", "V09"]
decide = ["C03", "C07", "D09", "H07", "M01", "M05", "T03", "T11", "T13", "T14", "U08", "U12", "V06", "V07"]
assert all(a3[q]["Clause present"] != r1[q]["clause_in_retrieved"] for q in auto_adopt + decide) and len(auto_adopt + decide) == 23
F = lambda **k: Font(name="Arial", **k)
thin = Side(style="thin", color="BFBFBF"); B = Border(left=thin, right=thin, top=thin, bottom=thin)
HEAD = PatternFill("solid", fgColor="305496"); YEL = PatternFill("solid", fgColor="FFFF00")
wb = Workbook(); g = wb.active; g.title = "How to use"
HOW = ["Deciding-clause adjudication", "",
       "Two annotators labelled whether the retrieved policy text contains the clause that decides each question; they disagree on these 14 questions.",
       "Convention to apply (agreed earlier): if the general rule is present but its decisive number, condition or qualifier is missing, the label is PARTIAL (e.g. the DePaul meals exception without 'occasional and non-repetitive'). A different, merely related rule is NOT part of the deciding rule.",
       "For each row on the 'Adjudicate' tab, read the Question, the Reference clauses and the Retrieved policy text, then look at both earlier labels and quotes.",
       "Choose the FINAL label using the agreed rule, judging ONLY the retrieved text shown:",
       "YES = the sentence that decides the question is present, including the number or condition it turns on.",
       "PARTIAL = part of the deciding rule is present, but its decisive number, condition or qualifier is missing.",
       "NO = the deciding rule is absent (source lines, headers, definitions or a different rule do not count).",
       "Write one line of reason in the last column (what is or is not in the text).",
       "The 'Auto-adopted' tab lists 9 other disagreements (round-1 'NO' where the cited clause text appears verbatim in the retrieved text); they will be adopted unless you flag one.",
       "No AI assistant, please. The AI answers and model names are deliberately not shown."]
for i, t in enumerate(HOW, 1):
    c = g.cell(row=i, column=1, value=t); c.font = F(bold=(i == 1), size=14 if i == 1 else 10); c.alignment = Alignment(wrap_text=True)
g.column_dimensions["A"].width = 130


def sheet(title, qs, decide_cols):
    ws = wb.create_sheet(title)
    cols = [("qid", 7), ("Institution", 14), ("Question", 40), ("Gold verdict", 13), ("Reference clauses", 45), ("Retrieved policy text", 70),
            ("Round-1 label", 10), ("Round-1 quote", 40), ("Annotator-3 label", 10), ("Annotator-3 quote / note", 45)]
    if decide_cols: cols += [("FINAL clause present", 14), ("Reason (one line)", 45)]
    for j, (h, w) in enumerate(cols, 1):
        c = ws.cell(row=1, column=j, value=h); c.font = F(bold=True, color="FFFFFF"); c.fill = HEAD; c.border = B
        c.alignment = Alignment(wrap_text=True); ws.column_dimensions[c.column_letter].width = w
    for i, q in enumerate(qs, 2):
        k = K[q]; ctx = "\n\n".join(f"[{ch[c]['doc']}] {ch[c]['text']}" for c in ret[q])
        vals = [q, k["institution"], k["question"], k["gold_verdict"], f"A: {k['clause_annotator_A']}\nB: {k['clause_annotator_B']}", ctx,
                r1[q]["clause_in_retrieved"], r1[q]["clause_quote"], a3[q]["Clause present"], a3[q]["Quote from retrieved text"] or ("NOTE: " + a3[q]["Notes"])]
        if decide_cols: vals += ["", ""]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v); c.font = F(size=10); c.border = B; c.alignment = Alignment(wrap_text=True, vertical="top")
            if decide_cols and j >= 11: c.fill = YEL
    if decide_cols:
        dv = DataValidation(type="list", formula1='"YES,PARTIAL,NO"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"K2:K{len(qs) + 1}")
    ws.freeze_panes = "D2"


sheet("Adjudicate", decide, True); sheet("Auto-adopted", auto_adopt, False); wb.active = 1
wb.save(OUT); print("saved", OUT, "| adjudicate", len(decide), "| auto-adopted", len(auto_adopt))
