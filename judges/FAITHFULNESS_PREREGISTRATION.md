# Faithfulness-prompt judge control — pre-registration

Written 2026-10-12, BEFORE any faithfulness-prompt call was made.

## Why
The compliance-rubric LLM judges differ from the grounding checks in two ways at once: model class (frontier LLM vs small encoder) and
task definition (compliance with the policy vs support by the text). This control keeps the model class and changes only the task
definition: the same two judges, the same inputs, a prompt that asks only whether the answer is SUPPORTED by the text (a
RAGAS-style faithfulness question). It also covers the claim that LLM-based faithfulness metrics behave like the grounding checks.

## Design (frozen in judges/JUDGE_CONFIG_FAITH.json with a SHA-256 before the first call)
- Judges and settings identical to the compliance judges: gpt-5.5-2026-04-23 and claude-sonnet-5-5, low reasoning effort, structured JSON
  output, Batch API, no server-side fallback, max tokens 4000.
- Items identical: 416 answers with retrieved context + 376 answers with oracle context, per judge (792 calls each).
  Input is the policy text and the response only (no question, gold, labels or model names); the user message template is unchanged.
- System prompt: identical in structure and wording to the compliance rubric except the task definition. A response is UNSUPPORTED if it
  contradicts the text or states something the text does not support; otherwise SUPPORTED (declining or referring the employee elsewhere
  is SUPPORTED). The sentence "Apply numbers and conditions exactly as written" is kept so the only difference is the task.
  Output: {"verdict": "SUPPORTED" or "UNSUPPORTED", "rationale": ...}. UNSUPPORTED is treated as flagging a violation.

## Analysis plan (fixed now)
- Metrics: recall on violations, FPR on compliant answers, and informedness J = recall - FPR, with question-bootstrap 95% intervals,
  for retrieved and oracle context. Compared with the compliance-rubric judges, the grounding checks and MSCG on the same answers.
- Paired tests: McNemar on violations and on compliant answers separately (faithfulness-prompt judge vs compliance-prompt judge, same model).
- Reference values from existing results (informedness): compliance judges, retrieved GPT 0.31 / Sonnet 0.30, oracle 0.46 / 0.43;
  grounding checks 0.01-0.19 (MiniCheck 0.01, DeBERTa not-entailed 0.08 retrieved / 0.19 oracle).

## Interpretation rules (fixed now, before results)
- If the faithfulness-prompt judges have J close to the grounding checks and clearly below the compliance-prompt judges, the gap is
  mainly due to the task definition, and the paper says so.
- If their J is close to the compliance-prompt judges, the gap is mainly due to model class, and the paper's framing ("grounding is not
  compliance") is revised accordingly.
- Anything in between is reported as partial. Whatever the result, it is reported in the main text.
