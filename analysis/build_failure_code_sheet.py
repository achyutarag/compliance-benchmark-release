"""Blind human verification sheet for the 38 coded failures (analysis/oracle_failures_coded.csv).
Shows question, oracle policy text (what the auditors saw), reference rule and the answer; hides my code, the answer model and all labels.
Output: labelling_ext/validation/failure_code_verification.xlsx + FAILURE_CODE_KEY_do_not_share.csv"""
import csv, json, os, random
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(H, "labelling_ext", "validation"); os.makedirs(OUT, exist_ok=True)
coded = list(csv.DictReader(open(os.path.join(H, "analysis", "oracle_failures_coded.csv"), encoding="utf-8")))
key = {r["qid"]: r for r in csv.DictReader(open(os.path.join(H, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
chunks = {c["chunk_id"]: c for c in map(json.loads, open(os.path.join(H, "index", "chunks.jsonl"), encoding="utf-8"))}
orc = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(H, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
ans = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for r in map(json.loads, open(os.path.join(H, "answers", f"answers_{m}.jsonl"), encoding="utf-8")): ans[(r["qid"], m)] = r
CATS = ["OVERRIDDEN_PROHIBITION", "MISAPPLIED_CONDITION", "EXACT_BOUNDARY", "ROLE_SPECIFIC", "OTHER"]
MAP = {"PROHIBITION": CATS[0], "CONDITION": CATS[1], "THRESHOLD": CATS[2], "ROLE": CATS[3]}
rng = random.Random(20261012); order = list(range(len(coded))); rng.shuffle(order)
F = lambda **k: Font(name="Arial", **k); thin = Side(style="thin", color="BFBFBF"); B = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook(); g = wb.active; g.title = "How to use"
HOW = ["Coding AI answers that violate a gift policy", "",
       "Each row is an AI assistant's answer to an employee question that breaks the institution's policy. The policy text shown is what a checker was given. The Reference clause is the rule the answer key says decides the question.",
       "Your task: choose the ONE category that best describes HOW the answer went wrong. Judge from the question, the policy text, the reference rule and the answer. You are not told what anyone else decided.", "",
       "OVERRIDDEN_PROHIBITION: a category of gift or benefit is banned (at any value, or from certain sources), but the answer applies a value threshold or permission from another clause. Example (made up): gift cards are banned at any value, and the answer says a $20 gift card is fine because items under $25 are allowed.",
       "MISAPPLIED_CONDITION: the answer relies on a real exception or permission without its conditions, or ignores a required condition or procedure. Example (made up): an exception allows meals that are 'occasional and non-repetitive', and the answer allows meals every month.",
       "EXACT_BOUNDARY: the answer gets a threshold wrong exactly at the boundary. Example (made up): the rule says 'less than $50' and the answer treats exactly $50 as allowed.",
       "ROLE_SPECIFIC: a separate rule depends on who the person is or whom the gift goes to, and the answer applies the general rule instead. Example (made up): gifts to a supervisor follow a different, lower limit.",
       "OTHER: none of the above fits. Please explain in Notes.", "",
       "Work alone, no AI assistant, only edit the yellow columns, and write a short note when the choice is not clear. About 1-2 minutes per row."]
for i, t in enumerate(HOW, 1):
    c = g.cell(row=i, column=1, value=t); c.font = F(bold=(i == 1), size=14 if i == 1 else 10); c.alignment = Alignment(wrap_text=True)
g.column_dimensions["A"].width = 130
ws = wb.create_sheet("Coding")
cols = [("row_id", 8), ("Institution", 16), ("Question", 40), ("Gold verdict", 14), ("Reference clause (answer key)", 45), ("Policy text shown to the checker", 70), ("Answer", 60), ("How the answer went wrong", 26), ("Notes", 40)]
for j, (h, w) in enumerate(cols, 1):
    c = ws.cell(row=1, column=j, value=h); c.font = F(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496"); c.border = B
    c.alignment = Alignment(wrap_text=True); ws.column_dimensions[c.column_letter].width = w
krows = []
for n, i in enumerate(order, 2):
    c0 = coded[i]; q, m = c0["qid"], c0["answer_model"]; k = key[q]; a = ans[(q, m)]
    pol = "\n\n".join(f"[{chunks[c]['doc']}] {chunks[c]['text']}" for c in orc[q])
    vals = [f"F{n - 1:02d}", k["institution"], k["question"], k["gold_verdict"], f"A: {k['clause_annotator_A']}\nB: {k['clause_annotator_B']}", pol, a["answer"], "", ""]
    for j, v in enumerate(vals, 1):
        cell = ws.cell(row=n, column=j, value=v); cell.font = F(size=10); cell.border = B; cell.alignment = Alignment(wrap_text=True, vertical="top")
        if j >= 8: cell.fill = PatternFill("solid", fgColor="FFFF00")
    krows.append({"row_id": f"F{n - 1:02d}", "qid": q, "answer_model": m, "llm_code": MAP[c0["code"]]})
dv = DataValidation(type="list", formula1='"' + ",".join(CATS) + '"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"H2:H{len(coded) + 1}")
ws.freeze_panes = "D2"; wb.active = 1
wb.save(os.path.join(OUT, "failure_code_verification.xlsx"))
with open(os.path.join(OUT, "FAILURE_CODE_KEY_do_not_share.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(krows[0])); w.writeheader(); w.writerows(krows)
from collections import Counter
print("rows:", len(krows), "| LLM code distribution:", dict(Counter(r["llm_code"] for r in krows)))
