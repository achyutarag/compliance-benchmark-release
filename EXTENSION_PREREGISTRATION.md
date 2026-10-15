# Real-RAG extension: two additional answer models — pre-registration

Written 2026-09-30, BEFORE any answer from these models was generated.

## Goal
Increase the number of labelled real-RAG answers (and hence violations) on the SAME frozen design, so that
recall estimates for the auditors rest on more positive cases. Nothing in the existing study changes.

## Model choice and rationale (decided before any output)
- `meta-llama/Llama-3.2-3B-Instruct` (Meta), revision 0cb88a4f764b7a12671c53f0838cd831a0843b95
- `google/gemma-2-2b-it` (Google), revision 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8

Rationale: open-weight instruction-tuned models (a) from vendors outside BOTH judge families (OpenAI, Anthropic),
so neither LLM judge shares a family with these answers, and (b) of a size practical for self-hosted RAG in
compliance settings. Models were NOT chosen by expected violation rate; Qwen2.5-3B and Phi-3.5-mini were the
initially proposed alternatives, replaced by Llama/Gemma for vendor recognisability before any generation.

## Frozen inputs (identical to the first two answer models)
- Questions: answer_key_FROZEN_2026-09-30.csv (sha256 689571cb…), all 104.
- Retrieved context: index/retrieval_frozen.jsonl (the same 2 chunks per question).
- Prompt: rag_retrieval.RAG_SYSTEM_PROMPT and build_user_prompt, unchanged.

## Generation settings
- Each model's own chat template. Gemma 2's template has no system role, so the system prompt is prepended to the
  user turn (standard practice for this model); Llama uses a system turn.
- Sampling: temperature 0.7, top_p 1.0, top_k disabled, max_new_tokens 512 (longest existing answer: 187 words, 258 tokens),
  seed = 2026 + question index, CPU, float32.
- One answer per question. No answer is dropped, filtered, edited or regenerated for any reason, including
  off-format, truncated or refused answers (these are labelled like any other answer). Generation errors
  (crashes) are retried with the same seed and logged.

## Labelling (same protocol as the original 208; safest variant)
- Both annotators label ALL 208 new answers, blind to the answering model, in randomised order, with the same
  workbook columns and instructions. Disagreements reconciled from written reasoning as before.
- Deciding-clause presence is a property of the frozen retrieval and the question, already labelled; it is reused.

## Auditors (all frozen; no changes)
MSCG guard_v10, DNA v2 (sha a54255a4…), GPT-5.5 and Claude Sonnet 5.5 with the frozen JUDGE_CONFIG
(sha c2755615…), retrieved and oracle context.

## Analysis plan
- Report per answer model and pooled over all four (416 answers).
- Intervals for pooled metrics: bootstrap resampling whole QUESTIONS (answers to one question are correlated).
- Judge-family check: the original crossed design (2 answer models x 2 judges) is reported as before; the two
  new models form an independent-family stratum where neither judge shares a family. The four answer models are
  NOT treated as an independent cross-family sample.
- Primary comparison: violation recall of MSCG vs each judge (McNemar on violations), and accuracy (Holm across
  the six pairwise auditor comparisons, as before).
- Whatever the violation count turns out to be, it is reported; no further models are added to reach a target.
