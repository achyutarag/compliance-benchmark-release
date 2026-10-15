import csv, numpy as np
from collections import defaultdict
from scipy.stats import binomtest
HERE = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
A = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(HERE + "/results/all416_per_answer.csv", encoding="utf-8"))}
F = defaultdict(dict)
for r in csv.DictReader(open(HERE + "/judges/faith_predictions.csv", encoding="utf-8")):
    F[(r["provider"], r["condition"])][(r["qid"], r["answer_model"])] = int(r["pred_unsupported"])
rng = np.random.default_rng(20261012)
def boot(keys, f):
    byq = defaultdict(list)
    for i, k in enumerate(keys): byq[k[0]].append(i)
    qs = list(byq); v = []
    for _ in range(3000):
        idx = [i for q in rng.choice(qs, len(qs)) for i in byq[q]]; v.append(f(idx))
    return np.percentile(v, [2.5, 97.5])
def stats(keys, p, y):
    tp = (p & y).sum(); fp = (p & (1 - y)).sum(); r = tp / y.sum(); f = fp / (1 - y).sum()
    J = lambda idx: ((p[idx] & y[idx]).sum() / max(1, y[idx].sum())) - ((p[idx] & (1 - y[idx])).sum() / max(1, (1 - y[idx]).sum()))
    lo, hi = boot(keys, J)
    return tp, fp, r, f, r - f, lo, hi
print("Informedness J = recall - FPR (question-bootstrap 95% CI)")
for cond in ("retrieved", "oracle"):
    keys = [k for k in A] if cond == "retrieved" else [k for k in A if A[k].get("gpt55_oracle") not in (None, "")]
    y = np.array([int(A[k]["y"]) for k in keys])
    print(f"\n{cond} (n={len(keys)}, violations={y.sum()})")
    for prov, col, nm in (("openai", f"gpt55_{cond}", "GPT-5.5"), ("anthropic", f"sonnet55_{cond}", "Sonnet 5.5")):
        pf = np.array([F[(prov, cond)][k] for k in keys]); pc = np.array([int(A[k][col]) for k in keys])
        for tag, p in (("faithfulness prompt", pf), ("compliance prompt  ", pc)):
            tp, fp, r, f, J, lo, hi = stats(keys, p, y)
            print(f"  {nm:10} {tag}: recall {tp}/{y.sum()} ({r:.0%})  FPR {f:.0%}  J {J:+.2f} [{lo:+.2f}, {hi:+.2f}]  flags {p.sum()}/{len(p)}")
        x = int(((pf == 1) & (pc == 0) & (y == 1)).sum()); z = int(((pf == 0) & (pc == 1) & (y == 1)).sum())
        x2 = int(((pf == 1) & (pc == 0) & (y == 0)).sum()); z2 = int(((pf == 0) & (pc == 1) & (y == 0)).sum())
        pv = lambda a, b: binomtest(min(a, b), a + b).pvalue if a + b else 1.0
        print(f"      McNemar faith vs compliance: violations {x} vs {z} (p={pv(x,z):.2g}) | compliant flagged {x2} vs {z2} (p={pv(x2,z2):.2g})")
print("\nReference J: MiniCheck 0.01/oracle ~0.03; DeBERTa-L not-entailed 0.08/0.19; RoBERTa not-ent 0.03/0.05; MSCG 0.14/0.19")
