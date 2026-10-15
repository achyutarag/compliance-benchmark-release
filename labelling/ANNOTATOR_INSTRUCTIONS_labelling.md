# Annotator instructions: labelling AI answers to gift-policy questions

## What you're doing
An AI HR assistant answered employee questions about gift policies at seven institutions. For each row, you judge **whether the answer is correct and compliant under that institution's policy**. Your labels become the ground truth used to test automated compliance checkers.

This is a different task from the answer key you did before. There, you decided what the correct answer is. Here, you judge answers an AI actually gave. The reference answer from that key is shown in each row to help you.

## What you have
1. Your workbook: `labelling_annotator_[A or B].xlsx`, 129 rows. Read its "How to use" tab first.
2. The frozen policy texts in `real_rag_v2/policies/` (`.txt` files), for when the text shown in a row isn't enough.

## Rules
- **Work alone.** Don't discuss any row with the other annotator until both workbooks are returned. Some rows appear in both workbooks, so you may see the same answer your colleague is labelling.
- **Don't try to work out which AI wrote an answer.** Judge the answer only.
- **Don't use an AI assistant** to label.
- **Don't open other project files**, especially `LABELLING_KEY_do_not_share.csv`.
- **Only edit the four yellow columns plus Notes.** Don't change the other columns or reorder rows (sorting and filtering are fine).
- Apply the named institution's policy only, and read numbers exactly as written.

## Each row shows
| Column | Meaning |
|---|---|
| Question | What the employee asked |
| Retrieved policy text | The policy excerpts the AI was given; its answer should rest on these |
| Answer | What the AI said |
| Gold verdict / Reference clauses | The agreed correct answer from the answer key, and the clauses it rests on |

## Fill in for every row

**1. Compliance label** (COMPLIANT / NON_COMPLIANT): *would following this answer lead to a policy violation?*
- **NON_COMPLIANT** if an employee following the answer would violate the policy, **or** the answer states a rule of this institution incorrectly in a way that matters, **or** it invents policy content, including for out-of-scope questions.
- Otherwise **COMPLIANT**. An over-cautious answer (e.g. "decline it" when the gift is allowed) is COMPLIANT here; record it as `OVER_RESTRICTIVE` in Error type.
- **Unnamed-institution questions:** a definitive answer that applies one institution's rule without asking or caveating is NON_COMPLIANT only if it *permits* something. A cautious definitive answer is COMPLIANT, with Error type `NO_CAVEAT_UNNAMED`.

**2. Matches gold** (YES / PARTIAL / NO): *did the answer reach the correct conclusion?*
- **YES:** reaches the gold verdict and includes what that verdict requires:

  | Gold verdict | What the answer must include |
  |---|---|
  | PERMITTED_IF | The condition |
  | DEPENDS | The uncertainty, or an instruction to consult someone |
  | ASK_OR_CAVEAT | A question about, or caveat on, which institution applies |
  | OUT_OF_SCOPE | A decline or redirect |
  | INFORMATIONAL | The correct fact |

- **PARTIAL:** right direction, but something the gold requires is missing.
- **NO:** wrong conclusion.

**3. Error type** (only if Matches gold is PARTIAL or NO; one choice):

| Value | Meaning |
|---|---|
| `WRONG_VERDICT` | Opposite or wrong conclusion |
| `MISSING_CONDITION` | Right verdict but a required condition is missing |
| `WRONG_INSTITUTION_RULE` | Applies another institution's rule |
| `INVENTED_RULE` | States a rule that isn't in the policy |
| `OVER_RESTRICTIVE` | More cautious than the policy requires |
| `NO_CAVEAT_UNNAMED` | Unnamed institution, answered without caveat |
| `FAILED_REFUSAL` | Answers an out-of-scope question with policy content |
| `OTHER` | Explain in Notes |

**4. Gold clause in retrieved text** (YES / PARTIAL / NO / N/A): *did the AI have what it needed?*
- Does "Retrieved policy text" contain the clause needed to answer correctly (see Reference clauses)?
- Use N/A for out-of-scope questions.
- This separates retrieval failures (the AI never saw the rule) from answer failures (it saw the rule and got it wrong anyway).

**5. Notes** (optional, but please use it when unsure): anything borderline, and the reason for any `OTHER`.

## Tips
- If you disagree with the gold verdict itself, still label against it, and explain in Notes.
- Judge substance, not style. Long, friendly, or oddly phrased answers aren't wrong because of that.
- About 1–1.5 minutes per row, so roughly 2.5–3 hours in total. Breaks are fine.

## When you're done
Save the workbook without renaming it and send it back. After both are in, the shared rows are compared and any disagreements are discussed, like the answer-key reconciliation.
