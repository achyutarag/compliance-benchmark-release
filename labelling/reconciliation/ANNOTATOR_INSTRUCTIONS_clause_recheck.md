# Instructions: clause re-check (one annotator, all 208 rows)

## What you're doing
For each row, answer **one question**: does the *retrieved policy text* contain the clause needed to answer the employee's question correctly?

This tells us, for every AI answer, whether a mistake came from **retrieval** (the AI never saw the rule) or from the **answer** (it saw the rule and still got it wrong). The AI's answer is deliberately **not shown**, so this judgement stays independent of whether the answer was right.

## What you have
- `clause_recheck.xlsx`: open the **Clause** tab (208 rows). The **How to use** tab repeats these instructions.
- Each row shows: Institution, Question, **Retrieved policy text**, Gold verdict, and **Reference clauses** (the clauses the answer key relied on, telling you which clause matters).

## The agreed rule (use exactly this)
**Judge only the text shown in "Retrieved policy text".** Don't use the full policy files, and don't rely on your memory of them.

| Value | When to use it |
|---|---|
| **YES** | The sentence that decides the question is present, **including the number or condition it turns on** |
| **PARTIAL** | Part of the deciding rule is present, but its decisive number, condition or qualifier is missing |
| **NO** | The deciding rule is absent. Source lines, headers, definitions, or a *different* rule do **not** count |
| **N/A** | Out-of-scope questions (not about gifts), and questions that don't name an institution |

## Worked examples (agreed in reconciliation)
- **DePaul, monthly vendor dinners:** the meals exception is present, but "occasional and non-repetitive" is missing → **PARTIAL**
- **Penn, "who should I ask?":** "Division of Human Resources" appears only in the document's source line; the "confer with supervisors" sentence is missing → **NO**
- **UVA, $25 logo umbrella:** both the nominal-advertising sentence and the procurement-integrity condition are present → **YES**

## How to fill it in
1. Read the **Question** and the **Reference clauses**, so you know which rule decides it.
2. Scan the **Retrieved policy text** for that rule.
3. Choose YES / PARTIAL / NO / N/A from the dropdown in **Gold clause present** (yellow).
4. **For every YES or PARTIAL, copy-paste the deciding sentence** (or the part that's present) from "Retrieved policy text" into **Quote from retrieved text** (yellow).
   - If you can't find text to paste, the answer isn't YES.
   - Paste it exactly (copy, don't retype). Quotes are checked automatically against the retrieved text.
   - Leave the quote empty for NO and N/A.
5. Use **Notes** (yellow) only when unsure, e.g. "condition present but threshold missing".

You already know these policies, and that's fine: it helps you recognise which rule matters. But mark YES only when you can point to the rule in the text on screen, never because you know it exists in the full policy. The quote column is what makes that checkable.

Only edit the three yellow columns (Gold clause present, Quote, Notes). Don't sort permanently or delete rows (filtering is fine). About 15–20 seconds per row, roughly 60 minutes in total.

## Rules
- Work alone, and don't use an AI assistant.
- Don't open other project files, especially `CLAUSE_RECHECK_KEY_do_not_share.csv`.

## When done
Save without renaming and send it back.
