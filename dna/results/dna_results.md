# DNA v2 component protocol results
config sha256 a54255a4df6ef3aa2685f8c433f6c9c3df5ca2b53a079328774d7db08333b4d3

## Leave-one-gate-out (accuracy; McNemar vs full DNA)
| Set | n | Full DNA | - Gate A (contradiction) | - Gate B (unsupported) |
| Development (gpt-4o-mini) | 77 | 80.5% | 80.5% (=) | 61.0% (p=0.036) |
| Fresh held-out (gpt-4o-mini) | 120 | 45.0% | 45.0% (=) | 74.2% (p<0.001) |
| Cross-generator (Claude) | 119 | 68.1% | 68.1% (=) | 63.0% (p=0.561) |
| Zero-shot: data retention | 99 | 67.7% | 67.7% (=) | 59.6% (p=0.410) |
| Zero-shot: COI disclosure | 99 | 59.6% | 59.6% (=) | 67.7% (p=0.403) |
| Zero-shot: AI data use | 94 | 78.7% | 78.7% (=) | 68.1% (p=0.203) |

## Each gate in isolation (recall on targeted category / FPR on label-0 items)
| Set | Gate A on Direct Failure | Gate B on Topic Drift | Gate B on Precedent Framing | Gate B on Adversarial (FP) |
| Development (gpt-4o-mini) | 16/16 (100%) / 0/29 (0%) | 16/16 (100%) / 15/29 (52%) | 16/16 (100%) | 15/15 (100%) |
| Fresh held-out (gpt-4o-mini) | 17/20 (85%) / 0/68 (0%) | 20/20 (100%) / 66/68 (97%) | 12/12 (100%) | 46/46 (100%) |
| Cross-generator (Claude) | 18/19 (95%) / 4/39 (10%) | 21/21 (100%) / 38/39 (97%) | 40/40 (100%) | 17/17 (100%) |
| Zero-shot: data retention | 19/21 (90%) / 0/40 (0%) | 20/20 (100%) / 32/40 (80%) | 18/18 (100%) | 15/15 (100%) |
| Zero-shot: COI disclosure | 20/20 (100%) / 1/43 (2%) | 20/20 (100%) / 40/43 (93%) | 16/16 (100%) | 23/23 (100%) |
| Zero-shot: AI data use | 20/21 (95%) / 0/40 (0%) | 20/20 (100%) / 20/40 (50%) | 13/13 (100%) | 20/20 (100%) |

## Full DNA: flagged per category
- Development (gpt-4o-mini): Adversarial 15/15, Direct Failure 16/16, Precedent Framing 16/16, Safe 0/14, Topic Drift 16/16
- Fresh held-out (gpt-4o-mini): Adversarial 46/46, Direct Failure 20/20, Precedent Framing 12/12, Safe 20/22, Topic Drift 20/20
- Cross-generator (Claude): Adversarial 17/17, Direct Failure 19/19, Precedent Framing 40/40, Safe 21/22, Topic Drift 21/21
- Zero-shot: data retention: Adversarial 15/15, Direct Failure 21/21, Precedent Framing 18/18, Safe 17/25, Topic Drift 20/20
- Zero-shot: COI disclosure: Adversarial 23/23, Direct Failure 20/20, Precedent Framing 16/16, Safe 17/20, Topic Drift 20/20
- Zero-shot: AI data use: Adversarial 20/20, Direct Failure 21/21, Precedent Framing 13/13, Safe 0/20, Topic Drift 20/20

## Real RAG, retrieved context (n=208, violations=28)
| Auditor | Acc | Recall (95% CI) | FPR |
| DNA (frozen) | 13.5% | 28/28 [88%, 100%] | 180/180 (100%) |
| DNA - Gate A | 16.3% | 25/28 [72%, 98%] | 171/180 (95%) |
| DNA - Gate B | 21.2% | 22/28 [59%, 92%] | 158/180 (88%) |
| MSCG (frozen) | 76.0% | 6/28 [8%, 41%] | 28/180 (16%) |
| Naive NLI | 80.8% | 2/28 [1%, 24%] | 14/180 (8%) |
| GPT-5.5 | 74.0% | 12/28 [24%, 63%] | 38/180 (21%) |
| Sonnet 5.5 | 66.8% | 11/28 [22%, 59%] | 52/180 (29%) |
- McNemar DNA vs MSCG (frozen): DNA-only-correct=22, other-only-correct=152, (p<0.001)
- McNemar DNA vs Naive NLI: DNA-only-correct=26, other-only-correct=166, (p<0.001)
- McNemar DNA vs GPT-5.5: DNA-only-correct=16, other-only-correct=142, (p<0.001)
- McNemar DNA vs Sonnet 5.5: DNA-only-correct=17, other-only-correct=128, (p<0.001)
- Isolated: Gate A recall 22/28, FPR 88%; Gate B recall 25/28, FPR 95%
- By clause in retrieved text (DNA recall / FPR):
  - YES: n=84, violations 2, recall 2/2, FPR 82/82
  - PARTIAL: n=20, violations 3, recall 3/3, FPR 17/17
  - NO: n=84, violations 17, recall 17/17, FPR 67/67
  - N/A: n=20, violations 6, recall 6/6, FPR 14/14
- Claims per answer: median 3; answers with >=1 unsupported claim: 196/208

## Oracle policy context (n=188, violations=22)
| Auditor | Acc | Recall (95% CI) | FPR |
| DNA (frozen) | 11.7% | 22/22 [85%, 100%] | 166/166 (100%) |
| DNA - Gate A | 17.0% | 21/22 [77%, 100%] | 155/166 (93%) |
| DNA - Gate B | 20.7% | 17/22 [55%, 92%] | 144/166 (87%) |
| DNA retrieved (same 188) | 11.7% | 22/22 [85%, 100%] | 166/166 (100%) |
| MSCG oracle | 79.3% | 7/22 [14%, 55%] | 24/166 (14%) |
| GPT-5.5 oracle | 58.5% | 20/22 [71%, 99%] | 76/166 (46%) |
| Sonnet 5.5 oracle | 58.5% | 20/22 [71%, 99%] | 76/166 (46%) |
- McNemar DNA vs MSCG oracle: DNA-only-correct=15, other-only-correct=142, (p<0.001)
- McNemar DNA vs GPT-5.5 oracle: DNA-only-correct=2, other-only-correct=90, (p<0.001)
- McNemar DNA vs Sonnet 5.5 oracle: DNA-only-correct=2, other-only-correct=90, (p<0.001)

## POST-HOC threshold-free signal quality: AUROC for violation (bootstrap 95% CI, 2000 resamples)
| Score | Retrieved (208) | Oracle (188) |
| DNA max claim contradiction | 0.40 [0.29, 0.51] | 0.33 [0.23, 0.43] |
| DNA 1 - min claim support | 0.49 [0.38, 0.60] | 0.65 [0.50, 0.78] |
| Whole-pair contradiction C (MSCG signal) | 0.53 [0.42, 0.65] | 0.43 [0.29, 0.57] |
| Whole-pair 1 - entailment E | 0.58 [0.46, 0.70] | 0.63 [0.50, 0.75] |
