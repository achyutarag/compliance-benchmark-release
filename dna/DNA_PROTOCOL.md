# Second modular system: Decomposed NLI Auditor (DNA) — protocol

Written 2026-09-30, BEFORE any DNA output on any evaluation set was computed.

## Purpose
A second modular auditor to which the component-level protocol is applied, built from the
alternative the related work points to: decomposition-based NLI checking (SummaC, FActScore,
AlignScore). It uses the SAME NLI model as MSCG (cross-encoder/nli-deberta-v3-base), so differences
from MSCG reflect granularity/aggregation, not model capacity.

## Components
1. **Claim segmentation**: the answer is split into sentences (regex on . ! ? and newlines; list
   markers stripped; fragments under 4 words dropped). Each sentence is one claim.
2. **Context segmentation**: the policy context is split into sentences; segments are every single
   sentence and every pair of adjacent sentences (window 2).
3. **Per-claim NLI**: for each claim i and segment j, (C_ij, E_ij, N_ij) from the NLI model with
   premise = segment, hypothesis = claim.
   - support_i = max_j E_ij ; contra_i = max_j C_ij
4. **Gate A — claim contradiction**: flag if max_i contra_i > tau_c.
5. **Gate B — unsupported claim**: flag if min_i support_i < tau_e.
   Verdict: violation if Gate A or Gate B fires (Gate A reported first for attribution).

## Calibration and freeze
- tau_c, tau_e chosen on the synthetic **fit split only** (output_v5.csv, split == "fit", 41 items),
  grid {0.05, 0.10, ..., 0.95} each, maximizing fit-split accuracy; ties broken toward fewest flags
  on the fit split (higher tau_c, then lower tau_e).
- The config (thresholds, segmentation parameters, model id) is written to DNA_CONFIG_FROZEN.json
  with a SHA-256, before any other set is scored. No changes afterwards.

## Evaluation (identical to MSCG's component protocol)
- Synthetic: development (77), fresh held-out (120), cross-generator Claude (119), three zero-shot
  domains (99/99/94). Real RAG: 208 answers, retrieved context; 188 answers, oracle policy context.
- Per set: accuracy; leave-one-gate-out (McNemar vs full); each gate in isolation (Gate A recall on
  Direct Failure, Gate B recall on Topic Drift and on Precedent Framing; real RAG: recall on all 28
  violations) with FPR on compliant items.
- Real RAG: comparison with MSCG / naive NLI / judges on the same answers (McNemar).
- Post-hoc, threshold-free (labelled as such): AUROC of the answer-level scores
  (max contra, 1 - min support) vs MSCG's whole-pair C, on real RAG retrieved and oracle.

## Predictions (written before running)
- P1 Synthetic: on single-sentence items decomposition is nearly trivial, so Gate A should behave like
  MSCG's contradiction gates (high Direct Failure recall). Gate B is expected to catch Topic Drift;
  whether it catches Precedent Framing is uncertain.
- P2 Real RAG, retrieved context: Gate B is expected to fire on many compliant answers, because real
  answers contain wording and advice not entailed by two retrieved chunks (high FPR). Gate A with
  max over many segments may also over-fire.
- P3 Oracle context: if MSCG's oracle-context failure is a granularity problem, DNA's recall should
  rise clearly above MSCG's 7/22. If it does not, the failure is not (only) granularity.

## Amendment 1 (2026-09-30, after fit-split calibration, BEFORE any evaluation set was scored)
Fit-split calibration was degenerate. On single-sentence synthetic items, max-segment entailment is
~0 (<= 0.013) for every non-paraphrase response, compliant Adversarial items included, so Gate B
cannot separate neutral-but-compliant wording from violations; the grid also could not disable
Gate B (tau_e >= 0.05 > all neutral E). The v1 optimum (tau_c 0.95, tau_e 0.05; fit accuracy 63.4%,
37/41 flagged, all 15 Adversarial flagged) is kept as DNA_CONFIG_v1_fit_iteration.json.
With Gate B disabled (not expressible in the v1 grid), Gate A alone reaches at most 27/41 (65.9%,
tau_c 0.05-0.10) and 26/41 at tau_c 0.5, i.e., no configuration separates the fit split well.

Decision: freeze v2 with the conventional zero-shot decision rule of support checkers, tau_c = 0.5
and tau_e = 0.5 (a claim is contradicted / unsupported when that class has probability > 0.5 /
entailment < 0.5 at its best segment). No parameter of DNA v2 is fitted to any data. All other
protocol elements, including the predictions above, are unchanged.
