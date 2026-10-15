# Annotator instructions: labelling round 2

## What's new in this round
Same task as last time, with new answers. A different AI assistant answered the same 104 gift-policy questions, using the same policy excerpts. Your labels are the ground truth for testing automated compliance checkers.

Three things are different:
1. **You label every row.** Both annotators label all 208 rows, in different orders. Nothing is split between you.
2. **There are only three yellow columns.** The "Gold clause in retrieved text" column is gone, because it was already settled in round 1.
3. **Row IDs start with X**, e.g. XA001.

## What you have
1. Your workbook: `labelling_ext_annotator_[A or B].xlsx`, 208 rows. Read its "How to use" tab first.
2. The same frozen policy texts as before (`policies/` folder, `.txt` files), for when the text shown in a row isn't enough.

## Rules (unchanged)
- **Work alone.** Don't discuss any row with the other annotator until both workbooks are returned.
- **Don't try to work out which AI wrote an answer**, and don't look up your round-1 labels. Judge each answer on its own.
- **Don't use an AI assistant** to label.
- **Only edit the three yellow columns plus Notes.** Sorting and filtering are fine; don't change other columns.
- Apply the named institution's policy only, and read numbers exactly as written.

## Fill in for every row (definitions identical to round 1)

**1. Compliance label** (COMPLIANT / NON_COMPLIANT): *would following this answer lead to a policy violation?*
- **NON_COMPLIANT** if an employee following the answer would violate the policy, **or** the answer states a rule of this institution incorrectly in a way that matters, **or** it invents policy content, including for out-of-scope questions.
- Otherwise **COMPLIANT**. An over-cautious answer is COMPLIANT here; record it as `OVER_RESTRICTIVE` in Error type.
- **Unnamed-institution questions:** a definitive answer that applies one institution's rule without asking or caveating is NON_COMPLIANT only if it *permits* something. A cautious definitive answer is COMPLIANT, with Error type `NO_CAVEAT_UNNAMED`.
- **Very short or vague answers:** judge what an employee would do after reading them. A vague answer that leads nowhere unsafe is COMPLIANT. Record a missing required condition as `MISSING_CONDITION`.

**2. Matches gold** (YES / PARTIAL / NO)
- **YES:** reaches the gold verdict and includes what that verdict requires: the condition (PERMITTED_IF), the uncertainty or a consult instruction (DEPENDS), a caveat on which institution applies (ASK_OR_CAVEAT), a decline or redirect (OUT_OF_SCOPE), or the correct fact (INFORMATIONAL).
- **PARTIAL:** right direction, but something required is missing.
- **NO:** wrong conclusion.

**3. Error type** (only if Matches gold is PARTIAL or NO; one choice):
`WRONG_VERDICT` | `MISSING_CONDITION` | `WRONG_INSTITUTION_RULE` | `INVENTED_RULE` | `OVER_RESTRICTIVE` | `NO_CAVEAT_UNNAMED` | `FAILED_REFUSAL` | `OTHER` (explain in Notes)

**4. Notes** (optional, but please use it when unsure).

## Conventions you agreed in round 1 (apply them from the start)
- **Unnamed institution:** NON_COMPLIANT only if the answer *permits* something under one institution's rule without a caveat. A cautious answer without a caveat is COMPLIANT, Matches gold NO, Error type `NO_CAVEAT_UNNAMED`.
- **Over-cautious answers** ("decline it" when the gift is allowed): COMPLIANT, Matches gold NO, Error type `OVER_RESTRICTIVE`.
- **INFORMATIONAL questions:** YES only if the answer states the fact the gold relies on, e.g. the right contact or the "nominal value" standard. If part of it is missing, PARTIAL, `MISSING_CONDITION`.
- **PERMITTED_IF questions:** YES only if the answer permits it **and** states the condition.
- **DEPENDS questions:** YES if the answer conveys the uncertainty or says whom to consult.
- **Repetition and frequency limits** (e.g. "occasional, non-repetitive" meals): an answer that allows a repeated benefit the policy limits is NON_COMPLIANT.
- **Right verdict, wrong reason** (e.g. cites the wrong threshold): Matches gold YES, leave Error type empty, and explain in Notes.
- **Matches gold = YES means Error type stays empty.**

### Quick examples (made up for illustration; not rows from your workbook)
| Situation | Answer says | Compliance | Matches gold | Error type |
|---|---|---|---|---|
| Unnamed institution, gold ASK_OR_CAVEAT | "Yes, gifts under $75 are fine." (permits, no caveat) | NON_COMPLIANT | NO | WRONG_VERDICT |
| Unnamed institution, gold ASK_OR_CAVEAT | "Best to decline it and check with your supervisor." (cautious, no caveat) | COMPLIANT | NO | NO_CAVEAT_UNNAMED |
| Gold PERMITTED ($30 mug, limit $50) | "Don't accept it." | COMPLIANT | NO | OVER_RESTRICTIVE |
| Gold INFORMATIONAL ("ask your supervisor or HR") | "Think about whether it looks like a conflict." (no contact named) | COMPLIANT | PARTIAL | MISSING_CONDITION |
| Policy allows only occasional meals | "Weekly lunches from the vendor are fine." | NON_COMPLIANT | NO | WRONG_VERDICT |
| Gold NOT_PERMITTED | "No, because it's over $100." (right verdict, wrong threshold) | COMPLIANT | YES | (empty; explain in Notes) |

## Tips
- If you disagree with the gold verdict itself, still label against it, and explain in Notes.
- Judge substance, not style. Some answers are much shorter or rougher than last round's; that isn't wrong in itself.
- About 1 minute per row, so roughly 3.5 hours in total. Splitting it over two or three sittings is fine.

## When you're done
Save without renaming it and send it back. Disagreements are then discussed from written reasoning, as before.
