# Release: do grounding checks detect policy violations in RAG answers?

Benchmark, labels, model outputs and analysis code for the paper (anonymized). Annotators appear only as A, B and 3.

## Folder layout
- Top level: frozen answer key and labels (+ manifests), the pipeline scripts (`generate_answers*.py`, `judges*.py`, `run_*.py`, `build_index.py`, `rag_retrieval.py`, `analyze_*.py`) and protocol files.
- `analysis/`: audit, statistics and revision scripts (start with `analysis/audit_tables.py`). `results/`: their outputs.
- `provenance/`: scripts that built the annotation workbooks, reconciled and froze labels, and cleaned the policy texts (run from the repo root as `python provenance/<script>.py`); they document the process and are not needed to reproduce the results.
- `docs/`: annotator instructions for the answer key and the policy source list. Annotator instructions for later rounds are in `labelling*/`.
- `synthetic/`: the synthetic benchmark data and frozen-guard notebook used for Section 6.

## Quick check (no API keys, no GPU)
    python analysis/reproduce_paper_numbers.py
Regenerates 41 counts and statistics behind the text (clause table, reasoning structures, judge family, error types, agreement statistics) and compares each with the value written in the paper; it exits non-zero on any difference. With the paper's `.tex` path as an argument it also checks every cell of the main results table.

    python analysis/audit_tables.py
Recomputes the main detection table (recall, false-positive rate, informedness, AUROC; retrieved and oracle context) and the
quoting / MiniCheck-support statistics from the raw per-item files. Its output should match `results/audit_tables.txt` exactly
(verified before release). Needs numpy, scipy, scikit-learn.

## Where the paper's numbers come from
| Paper content | Files |
|---|---|
| Benchmark: 104 questions, answer key (kappa 0.82) | `answer_key_FROZEN_2026-09-30.csv` (+ MANIFEST), `annotations/`, `answer_key_review_annotator_*.csv` |
| 416 answers (4 answer models) | `answers/answers_*.jsonl`, `index/`, `policies/` (+ `docs/POLICY_SOURCES.md`) |
| Compliance labels (79 violations) | `labels_FROZEN_2026-09-30.csv`, `labels_ext_FROZEN_2026-10-01.csv` (+ MANIFESTs), `labelling*/` |
| Clause-presence labels (v2) | `clause_labels_v2_FROZEN_2026-10-11.csv`, `labelling/clause_recheck2/`, `analysis/retrieval_misses.py`, `results/clause_v2_tables.txt` |
| Main detection table, paired tests | `results/audit_tables.txt`, `results/analysis_416.txt`, `results/all416_per_answer.csv`, `grounding/` |
| Grounding checkers (MiniCheck, DeBERTa-L NLI, RoBERTa-MNLI) | `grounding/` (protocol, revisions, raw scores) |
| Full-policy + question judge baseline (396 named-institution answers) | `judges/FULL_POLICY_PREREGISTRATION.md` (incl. deviation log), `judges_full.py`, `judges_full_openai_runner.py`, `judges/JUDGE_CONFIG_FULL.json`, `judges/full_predictions.csv` and raw batch inputs/outputs, `analysis/full_policy_analysis.py` -> `results/full_policy_analysis.txt` |
| HHEM-2.1-open (Amendment 3 of the grounding protocol) | `grounding/run_hhem.py` (needs a local copy of the pinned model repo, see `grounding/CHECKER_REVISIONS.json`; weights not redistributed), `grounding/hhem_code/` (the two reviewed code files), `grounding/scores_hhem_whole.jsonl`, `analysis/hhem_results.py` -> `results/hhem_results.txt` |
| LLM judges and controls (compliance, faithfulness, question-aware) | `judges/` (configs, frozen prompts, predictions), `results/faith_analysis.txt`, `results/q_analysis.txt` |
| MSCG and DNA auditors | `run_mscg*.py`, `dna/`, `results/mscg_*.csv`, `results/ext_local_auditors.csv` |
| Failure analysis (38 failures, LLM + blind human codes) | `analysis/oracle_failures_coded.csv`, `analysis/failure_codes_llm_vs_human.csv`, `analysis/failure_code_agreement.py` |
| Section 6 synthetic table (naive NLI vs MSCG; gate recalls) | `synthetic/` (labelled synthetic sets, frozen-guard notebook, ablation tables), `python analysis/synthetic_audit.py` -> `results/synthetic_audit.txt` (needs the NLI/embedding models; no API) |
| Question-level (cluster) tests, precision / PR-AUC, error-type breakdown, per-answer-model recall, quoting-rate CI | `analysis/revision_analyses.py` -> `results/revision_analyses.txt`; controls: `analysis/controls_cluster_tests.py` -> `results/controls_cluster_tests.txt` |
| Label audit: the 17 author-approved items and sensitivity of the results to alternative labels | `analysis/label_audit.py` -> `results/label_audit.txt` |
| Cost / latency | `efficiency.py`, `results/efficiency.txt` |

## Things to know
- **Blinding and key files.** Files named `*_KEY_do_not_share.csv` mapped answers to models/annotators and were withheld from
  annotators during labelling. The names are kept as in the working directory so the scripts still run; the annotation rounds
  they belong to are finished, so they are released. One further round (an independent validation sample) is not part of this release.
- **Pre-specification.** Protocol files named `*PREREGISTRATION*` were written to disk before the runs they describe and carry
  file hashes / timestamps. They were **not** entered in an external registry; the paper calls them pre-specified.
- **Annotator B's round-2 labels** come from a table pasted back by B after a first returned file proved not to be our data
  (see `excluded/EXCLUDED_NOTE.md`; the file itself is withheld). The transcription is `labelling_ext/completed/annotator_B_labels_from_chat_paste.tsv`.
- **AI use.** An LLM assistant drafted the questions and initial key, wrote the code, drafted proposals for 17 disputed labels and
  coded the 38 failures (annotator A re-coded them blind; see paper). Annotators worked without AI assistants.
- **Included raw outputs:** `judges/*batch_input/output*.jsonl` are the raw OpenAI/Anthropic Batch API inputs and outputs behind the judge predictions.
- **Policy texts:** `policies/` holds the cleaned text used for retrieval (+ `CLEANING_LOG.md`); `docs/POLICY_SOURCES.md` gives each official URL and access date. Archived PDF/HTML copies are not redistributed; fetch them from the sources.
- **Not included:** API keys and the files of the independent validation sample.
- Re-running the generation, judge or checker scripts needs API keys / model downloads (OpenAI, Anthropic, Hugging Face) and costs money;
  stored outputs are provided so the analysis does not need them.
