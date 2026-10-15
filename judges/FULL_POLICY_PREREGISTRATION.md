# Full-policy + question judge baseline — pre-specification

Written 2026-10-14, BEFORE any full-policy call was made. (Pre-specified in a file with a creation time; not entered in an external registry.)

## Why
A reviewer-style objection: the main judges see only two retrieved chunks (80 words each) and not the question, while the corpus is small
enough to put a whole institution's policy in context together with the employee's question, which is the obvious deployment baseline.
This control tests whether LLM judges given the full policy and the question detect policy violations better than the judges in the main comparison.

## Design (frozen in judges/JUDGE_CONFIG_FULL.json with a SHA-256 before the first call)
- Same two judges and settings as the other judges (gpt-5.5-2026-04-23, claude-sonnet-5-5, low reasoning effort, structured output,
  Batch API, no server-side fallback, max tokens 4000).
- Same rubric and user template as the question-aware control (policy text, employee's question, response). The only change is the policy text:
  the complete cleaned text of every policy document of the question's institution (documents concatenated, each preceded by its file name),
  not retrieved or oracle chunks.
- Items: the 396 answers (99 questions x 4 answer models) whose question names an institution. The 5 questions about an unnamed institution
  (20 answers) are not run, because "the full policy" is undefined for them; they are reported as not run.
- The gold verdict, answer-key clauses, labels and model names are NOT shown.

## Analysis plan (fixed now)
- Metrics: recall on violations, FPR on compliant answers, informedness J = recall - FPR, precision; question-bootstrap 95% intervals.
- Comparisons on the same 396 answers: (a) the question-aware judge with retrieved context, (b) the compliance judge with retrieved context,
  (c) the best grounding check by informedness with retrieved context; tests are question-level sign-flip permutation tests on violations,
  on compliant answers and on informedness.
- Secondary: recall on role-conditioned violations (S6) and by answer model (descriptive).

## Interpretation rules (fixed now)
- Whatever the result, it is reported in the main text. If full-policy judges are clearly better than the retrieved-chunk judges, the paper says that the
  comparison with grounding checks is bounded by the 2-chunk setup; if not, it says that more context did not help these judges.
- No change to the main comparison table; this is an added baseline.

## Deviation log (2026-10-14, infrastructure only)
The single OpenAI batch (396 requests, about 2.2M input tokens) was rejected at validation by the provider's enqueued-token limit for the model
(1.35M tokens; status `failed` with 0 requests processed, nothing charged). The same 396 requests were therefore submitted as three sequential OpenAI batches
(each under about 1.1M tokens, next one submitted after the previous completed) by judges_full_openai_runner.py. Prompt, template, items, model, settings and
analysis plan are unchanged. The Anthropic batch (396 requests) ran as one batch without errors.
