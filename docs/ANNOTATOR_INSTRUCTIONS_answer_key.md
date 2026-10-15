# Annotator instructions: answer key for the gift-policy questions

## What you're doing
You will decide, for each of 104 employee questions, what the **correct answer is under the named institution's policy**. Your answers, together with a second annotator's, will become the ground truth used to grade AI-generated answers later. You are **not** grading any AI answers in this task.

## What you have
1. Your review sheet: `answer_key_review_annotator_[A or B].csv` (104 rows).
2. The policy texts for seven institutions (Hartwick, Penn, Tennessee, DePaul, UVA, UTHealth Houston, Massachusetts).

Use **only** these policy texts. Do not use outside knowledge, other websites, or what you think a sensible policy would be.

## Rules
- **Work alone.** Do not discuss questions with the other annotator until both sheets are submitted.
- **Do not open any other project files.** Some contain a draft answer key, and seeing it would invalidate your work.
- **Do not use an AI assistant** to answer the questions.
- If a question names an institution, apply **only that institution's policy**, even if another policy would answer differently.
- Apply numbers **exactly as written**. "Under $50", "$50 or less", "more than $50" and "$50 or more" are different rules, and exactly $50 matters.

## For each row, fill in four columns

**1. `your_verdict`**: exactly one of these:

| Verdict | Use when |
|---|---|
| `PERMITTED` | The policy allows what the question describes. |
| `PERMITTED_IF` | Allowed **only if** a condition is met (e.g. "only if shared with the office", "only with supervisor approval"). Name the condition in `notes`. |
| `NOT_PERMITTED` | The policy prohibits what the question describes. |
| `DEPENDS` | The policy is genuinely unclear, silent, or leaves it to judgment. Explain why in `notes`. |
| `ASK_OR_CAVEAT` | The question doesn't say which institution, and the answer differs across them. |
| `OUT_OF_SCOPE` | Not a gift-policy question. |

Use `DEPENDS` only when the text really doesn't settle it, not when a question is merely hard. If two clauses seem to conflict, note both.

**2. `clause_you_relied_on`**: a short pointer to the sentence(s) you used, e.g. "cash-equivalent sentence", "5.08(11) perishables", "procurement-staff rule".

**3. `keep_question(1/0)`**: `0` if the question is confusing, unrealistic, or can't be answered from the policy text; otherwise `1`. Say why in `notes`.

**4. `notes`**: conditions for `PERMITTED_IF`, reasons for `DEPENDS`, anything you were unsure about, and any suggested rewording.

## Time and submission
Expect roughly 1–2 hours. Save the CSV without changing the `qid`, `institution` or `question` columns, and send it back. After both sheets are in, the two of you will meet once to resolve the disagreements; that discussion comes **after** submission, not before.
