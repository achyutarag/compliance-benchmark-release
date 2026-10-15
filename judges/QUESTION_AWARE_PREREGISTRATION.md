# Question-aware judge control — pre-registration

Written 2026-10-12, BEFORE any question-aware call was made.

## Why
The paper recommends giving the auditor the employee's question, because rules such as role-conditioned ones turn on facts stated only in
the question. The compliance judges never saw the question. Role-conditioned violations (structure S6, 14 violations) were their weakest
case (GPT-5.5 2, Sonnet 5.5 4 detected, with retrieved context). This control tests the recommendation.

## Design (frozen in judges/JUDGE_CONFIG_Q.json with a SHA-256 before the first call)
- Same two judges and settings as the compliance judges (gpt-5.5-2026-04-23, claude-sonnet-5-5, low reasoning effort, structured output,
  Batch API, no server-side fallback, max tokens 4000); same 792 items per judge (416 retrieved + 376 oracle).
- The compliance rubric is unchanged except for one sentence: "You will be shown policy text and a response ..." becomes
  "You will be shown policy text, the employee's question, and the response that an HR assistant gave to that employee." The decision
  criteria, exact-numbers sentence and output format are identical. The user message adds the employee's question between the policy text and the response.
- The gold verdict, the answer-key clauses, labels and model names are NOT shown.

## Analysis plan (fixed now)
- Metrics: recall on violations, FPR on compliant answers, informedness J = recall - FPR with question-bootstrap 95% intervals, retrieved and oracle.
- Paired tests (same judge, with vs without the question): McNemar on violations and on compliant answers separately.
- Secondary: recall on role-conditioned violations (structure S6; 14 violations) and by other reasoning structures (descriptive).
- Reference (no question): informedness retrieved GPT 0.31 / Sonnet 0.31, oracle 0.46 / 0.43; S6 recall retrieved GPT 2/14, Sonnet 4/14.

## Interpretation rules (fixed now)
- If the question significantly improves recall or informedness, or S6 recall rises clearly, the recommendation is supported and reported as a result.
- If it does not, the recommendation is reported as untested-to-unsupported and the discussion is revised.
- Whatever the result, it is reported in the main text.
