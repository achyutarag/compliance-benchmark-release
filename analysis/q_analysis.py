import csv, numpy as np
from collections import defaultdict
from scipy.stats import binomtest
HERE = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
A = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(HERE + "/results/all416_per_answer.csv", encoding="utf-8"))}
L = {(r["qid"], r["answer_model"]): r for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv") for r in csv.DictReader(open(HERE + "/" + f, encoding="utf-8"))}
Q = defaultdict(dict)
for r in csv.DictReader(open(HERE + "/judges/q_predictions.csv", encoding="utf-8")): Q[(r["provider"], r["condition"])][(r["qid"], r["answer_model"])] = int(r["pred_violation"])
rng = np.random.default_rng(20261012)
def ci(keys, f):
    byq = defaultdict(list)
    for i, k in enumerate(keys): byq[k[0]].append(i)
    qs = list(byq); v = []
    for _ in range(3000):
        idx = [i for q in rng.choice(qs, len(qs)) for i in byq[q]]; v.append(f(idx))
    return np.percentile(v, [2.5, 97.5])
pv = lambda a, b: binomtest(min(a, b), a + b).pvalue if a + b else 1.0
for cond in ("retrieved", "oracle"):
    keys = list(A) if cond == "retrieved" else [k for k in A if A[k].get("gpt55_oracle") not in (None, "")]
    y = np.array([int(A[k]["y"]) for k in keys]); print(f"\n{cond} (n={len(keys)}, violations={y.sum()})")
    for prov, col, nm in (("openai", f"gpt55_{cond}", "GPT-5.5"), ("anthropic", f"sonnet55_{cond}", "Sonnet 5.5")):
        pq = np.array([Q[(prov, cond)][k] for k in keys]); p0 = np.array([int(A[k][col]) for k in keys])
        for tag, p in (("with question   ", pq), ("without question", p0)):
            r = (p & y).sum() / y.sum(); f = (p & (1 - y)).sum() / (1 - y).sum()
            J = lambda idx, p=p: (p[idx] & y[idx]).sum() / max(1, y[idx].sum()) - (p[idx] & (1 - y[idx])).sum() / max(1, (1 - y[idx]).sum())
            lo, hi = ci(keys, J); print(f"  {nm:10} {tag}: recall {(p & y).sum()}/{y.sum()} ({r:.0%}) FPR {f:.0%} J {r - f:+.2f} [{lo:+.2f}, {hi:+.2f}] flags {p.sum()}")
        x = int(((pq == 1) & (p0 == 0) & (y == 1)).sum()); z = int(((pq == 0) & (p0 == 1) & (y == 1)).sum())
        x2 = int(((pq == 1) & (p0 == 0) & (y == 0)).sum()); z2 = int(((pq == 0) & (p0 == 1) & (y == 0)).sum())
        print(f"      McNemar with-vs-without question: violations caught only with {x} / only without {z} (p={pv(x, z):.2g}) | compliant flagged only with {x2} / only without {z2} (p={pv(x2, z2):.2g})")
print("\nROLE-CONDITIONED violations (structure S6), retrieved context:")
S6 = [k for k in A if A[k]["y"] == "1" and "S6" in L[k]["structures"].split(";")]
for prov, col, nm in (("openai", "gpt55_retrieved", "GPT-5.5"), ("anthropic", "sonnet55_retrieved", "Sonnet 5.5")):
    print(f"  {nm}: n={len(S6)}; without question {sum(int(A[k][col]) for k in S6)}, with question {sum(Q[(prov,'retrieved')][k] for k in S6)}")
print("BY STRUCTURE (violations, retrieved): structure n | GPT without->with | Sonnet without->with")
for s in ["S2", "S3", "S5", "S6", "S7", "S8", "S10", "S12", "S13"]:
    ks = [k for k in A if A[k]["y"] == "1" and s in L[k]["structures"].split(";")]
    if ks: print(f"  {s:4} {len(ks):2d} | GPT {sum(int(A[k]['gpt55_retrieved']) for k in ks)}->{sum(Q[('openai','retrieved')][k] for k in ks)} | Sonnet {sum(int(A[k]['sonnet55_retrieved']) for k in ks)}->{sum(Q[('anthropic','retrieved')][k] for k in ks)}")
