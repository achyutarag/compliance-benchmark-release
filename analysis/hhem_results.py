"""HHEM-2.1-open results (grounding/GROUNDING_PROTOCOL.md Amendment 3): recall, FPR, informedness, precision, AUROC, PR-AUC with question-bootstrap CIs,
and question-level permutation tests against the two compliance judges."""
import contextlib, io, json, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, pred, oracle_keys = g["Y"], g["pred"], g["oracle_keys"]
S = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding", "scores_hhem_whole.jsonl"), encoding="utf-8")}
assert len(S) == 792, len(S)
rng = np.random.default_rng(20261014); out = []
def P(s): print(s); out.append(s)
def pcl(a, b, ks, stat, n=20000):
    nv = sum(Y[k] for k in ks); nc = len(ks) - nv; tot = defaultdict(float)
    for k in ks:
        d = a[k] - b[k]; tot[k[0]] += {"recall": d / nv if Y[k] else 0.0, "fpr": d / nc if not Y[k] else 0.0, "J": d / nv if Y[k] else -d / nc}[stat]
    t = np.array(list(tot.values())); obs = t.sum(); sg = rng.choice([-1, 1], size=(n, len(t)))
    return obs, (1 + (np.abs((sg * t).sum(1)) >= abs(obs) - 1e-12).sum()) / (n + 1)
for cond, ks in (("retrieved", sorted(Y)), ("oracle", oracle_keys)):
    y = np.array([Y[k] for k in ks]); p = np.array([int(S[(k[0], k[1], cond)] < 0.5) for k in ks]); sc = np.array([1 - S[(k[0], k[1], cond)] for k in ks])
    tp = int((p & y).sum()); fp = int((p & (1 - y)).sum())
    byq = defaultdict(list)
    for i, k in enumerate(ks): byq[k[0]].append(i)
    qs = list(byq); js = []
    for _ in range(3000):
        idx = [i for q in rng.choice(qs, len(qs)) for i in byq[q]]; yy, pp = y[idx], p[idx]
        js.append((pp & yy).sum() / max(1, yy.sum()) - (pp & (1 - yy)).sum() / max(1, (1 - yy).sum()))
    P(f"=== HHEM-2.1-open, {cond}: n={len(ks)}, violations={y.sum()}")
    P(f"  recall {tp}/{y.sum()} ({tp / y.sum():.0%})  FPR {fp}/{(1 - y).sum()} ({fp / (1 - y).sum():.0%})  J {tp / y.sum() - fp / (1 - y).sum():+.2f} [{np.percentile(js, 2.5):+.2f}, {np.percentile(js, 97.5):+.2f}]  precision {tp / max(1, p.sum()):.2f}  AUROC {roc_auc_score(y, sc):.2f}  PR-AUC {average_precision_score(y, sc):.2f}  flags {p.sum()}")
    H_ = {k: int(S[(k[0], k[1], cond)] < 0.5) for k in ks}
    for jn, pn in (("GPT-5.5", "gpt"), ("Sonnet 5.5", "sonnet")):
        J_ = {k: pred(pn, cond, k) for k in ks}
        for stat in ("recall", "fpr", "J"):
            o, pv = pcl(J_, H_, ks, stat); P(f"  {jn} minus HHEM, {stat}: {o:+.2f}, question-level p = {pv:.4f}")
open(os.path.join(H, "results", "hhem_results.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
