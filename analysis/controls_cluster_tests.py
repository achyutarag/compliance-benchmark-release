"""Question-cluster sign-flip permutation tests for the two judge controls (replaces answer-level McNemar p-values)."""
import csv, os
import numpy as np
from collections import defaultdict
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: list(csv.DictReader(open(os.path.join(H, p), encoding="utf-8")))
A = {(r["qid"], r["answer_model"]): r for r in rd("results/all416_per_answer.csv")}
rng = np.random.default_rng(20261014)
def load(f, col):
    D = defaultdict(dict)
    for r in rd(f): D[(r["provider"], r["condition"])][(r["qid"], r["answer_model"])] = int(r[col])
    return D
F, Q = load("judges/faith_predictions.csv", "pred_unsupported"), load("judges/q_predictions.csv", "pred_violation")
C = load("judges/judge_predictions.csv", "pred_violation"); C2 = load("judges/judge_predictions_ext.csv", "pred_violation")
for d in (C2,):
    for k, v in d.items(): C[k].update(v)
def test(a, b, keys, which, n=20000):
    ks = [k for k in keys if (int(A[k]["y"]) == 1) == (which == "violations")]; tot = defaultdict(float)
    for k in ks: tot[k[0]] += a[k] - b[k]
    t = np.array(list(tot.values())); obs = t.sum(); sg = rng.choice([-1, 1], size=(n, len(t)))
    return int(round(obs)), len(ks), (1 + (np.abs((sg * t).sum(1)) >= abs(obs) - 1e-9).sum()) / (n + 1)
out = []
def P(s): print(s); out.append(s)
for prov, nm in (("openai", "GPT-5.5"), ("anthropic", "Sonnet 5.5")):
    for cond in ("retrieved", "oracle"):
        keys = sorted(Q[(prov, cond)]) if cond == "oracle" else sorted(A)
        for label, X in (("faithfulness prompt vs compliance prompt", F), ("question-aware vs question-free (compliance prompt)", Q)):
            for which in ("violations", "compliant"):
                if label.startswith("faith"): d, n, p = test(X[(prov, cond)], C[(prov, cond)], keys, which)
                else: d, n, p = test(X[(prov, cond)], C[(prov, cond)], keys, which)
                P(f"{nm:10} {cond:9} {label:55} {which:10} (n={n}) net flagged diff {d:+d}  cluster p = {p:.4f}")
open(os.path.join(H, "results", "controls_cluster_tests.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
