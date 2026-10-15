# Established grounding checkers on real-RAG compliance — protocol

Written 2026-10-02, before any checker was run on any answer.

## Question
Do established grounding / factual-consistency checkers (not built by us) detect policy violations in real RAG answers?
Tests whether the weak real-RAG performance of MSCG's NLI signals is specific to our auditor or general to grounding checks.

## Checkers (frozen revisions recorded at download)
1. MiniCheck-RoBERTa-Large (lytang/MiniCheck-RoBERTa-Large; Tang et al., EMNLP 2024): document–claim grounding classifier.
2. DeBERTa-v3-large NLI (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli): strong general NLI model.
No model is trained, tuned or prompted by us. Weights loaded without remote code.

## Inputs (identical to the other auditors)
premise/document = policy text (retrieved chunks; oracle chunks in the oracle condition); hypothesis/claim = full answer.
Same 416 answers (retrieved) and 376 answers (oracle). Labels: frozen human labels (both rounds).

## Decision rules (fixed now)
- MiniCheck: violation if P(supported) < 0.5 (its documented default).
- DeBERTa-large NLI, two rules reported, both fixed now:
  (a) "contradiction" rule: violation if argmax = contradiction (as naive NLI);
  (b) "not entailed" rule: violation if P(entailment) < 0.5 (grounding-style use).
- Also answer-sentence-level variants (max over sentences of unsupported/contradiction), using the same sentence splitter as DNA,
  reported as secondary.

## Reported
Recall on violations, FPR on compliant answers, accuracy; AUROC (threshold-free) with bootstrap CI; retrieved and oracle
context; recall by reasoning structure. Compared with MSCG and the two LLM judges on the same answers.
Whatever the result, it is reported.

## Amendment (2026-10-02, before any score was analysed)
Weights are loaded in bfloat16 (softmax in float32) and items are scored in batches, because float32 inference swapped on this machine (6.6 s per item). 234 partial float32 MiniCheck scores were discarded unread (grounding/discarded_fp32_partial/); all reported scores come from the bf16 run.

## Scope note (2026-10-02)
The sentence-level variants (secondary) were not run: at ~6 h per checker on this CPU they did not fit the schedule. Only whole-answer scores are reported.

## Amendment 2 (2026-10-03, before running): third checker
Added FacebookAI/roberta-large-mnli (widely used NLI model), chosen by the same criteria fixed before any checker ran (published/widely used, open weights, no remote code, CPU-runnable). Same two NLI rules as DeBERTa-v3-large (argmax contradiction; P(entailment)<0.5), whole answers, bf16. Considered and excluded by those criteria: Vectara HHEM (requires repository code), AlignScore (requires its own codebase).


## Amendment 3 (2026-10-14, before running): fourth checker, HHEM-2.1-open
Added vectara/hallucination_evaluation_model (HHEM-2.1-open, a published factual-consistency model), previously excluded because it needs repository code (trust_remote_code).
Its two repository code files (configuration_hhem_v2.py, modeling_hhem_v2.py; revision de9a5e6b...) were read in full before running: they build a T5 token classifier and load the
tokenizer/config of google/flan-t5-base (revision 7bcac572...); no other code or network access. Code is copied to grounding/hhem_code/; weights are not redistributed.
Inputs identical to the other checkers: premise = policy text (retrieved chunks; oracle chunks in the oracle condition), hypothesis = full answer, 416 + 376 items, whole-answer only.
Decision rule fixed now: violation if the model's own two-class output assigns P(consistent) < 0.5 (its native decision; threshold-free AUROC on 1 - P(consistent) also reported).
float32 (small model), batch size 8. Whatever the result, it is reported with the other checkers; it was added after the other checkers' results were known, in response to a reviewer-style objection that the earlier exclusion was arbitrary.
AlignScore remains excluded (separate codebase and checkpoint; not run).

## Amendment 4 (2026-10-15, before running): question-plus-answer hypothesis
Reviewer-style objection: the grounding checks scored the answer alone, but many answers ("Yes, as long as ...") are not standalone claims, so this is not an equal-input comparison with the question-aware judges.
Added condition for MiniCheck, HHEM-2.1-open and DeBERTa-v3-large NLI (same models, revisions, premises, bf16/float32 settings and decision rules as before):
hypothesis = "Question: <question> Answer: <answer>" (the question verbatim from the benchmark, then the whole answer); premise = retrieved chunks (416 answers) or oracle chunks (376 answers).
No threshold, rule or model is changed; AUROC and the same decision rules are reported next to the answer-only rows. Whatever the result, it is reported. RoBERTa-large-MNLI and the modular auditors are not rerun (the NLI model for the contradiction/not-entailed rules is represented by DeBERTa-large).
