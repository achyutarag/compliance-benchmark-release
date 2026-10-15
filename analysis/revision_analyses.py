"""Revision analyses (no API calls): error-type breakdown of all 79 violations, quoting-rate CI, question-cluster permutation tests,
precision / PR-AUC, and per-answer-model recall. Reuses the audit's loaders (analysis/audit_tables.py, header only)."""
import contextlib, io, os, re, csv, json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read()
src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, pred, score, NAMES, oracle_keys = g["Y"], g["pred"], g["score"], g["NAMES"], g["oracle_keys"]
rd = g["rd"]
ET = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in rd(f): ET[(r["qid"], r["answer_model"])] = r["error_type"]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
rng = np.random.default_rng(20261014)
allk = sorted(Y); conds = {"retrieved": allk, "oracle": oracle_keys}
PRED = {(nm, c): {k: pred(pn, c, k) for k in ks} for c, ks in conds.items() for nm, pn, sn in NAMES for _ in [0]}

P("=== 1. Error type of all 79 violations (all answer models)")
P(dict(Counter(ET[k] for k in allk if Y[k])))
P("recall by error type, retrieved context (n per type in header)")
types = ["WRONG_VERDICT", "FAILED_REFUSAL", "INVENTED_RULE"]
P(f"{'auditor':26}" + "".join(f"{t:>20}" for t in types))
for nm, pn, sn in NAMES:
    row = ""
    for t in types:
        ks = [k for k in allk if Y[k] and ET[k] == t]; row += f"{sum(PRED[(nm, 'retrieved')][k] for k in ks):>14d}/{len(ks):<5d}"
    P(f"{nm:26}" + row)
P("\n=== 2. Quoting rate (5-word verbatim span) difference, violating minus compliant, question-cluster bootstrap")
ans, ret = {}, {}
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for r in map(json.loads, open(os.path.join(H, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r["answer"]; ret[(r["qid"], m)] = " ".join(c["text"] for c in r["retrieved"])
g5 = lambda t: {tuple(w[i:i + 5]) for w in [re.findall(r"[a-z0-9$]+", t.lower())] for i in range(len(w) - 4)}
has = {k: int(len(g5(ans[k]) & g5(ret[k])) > 0) for k in allk}
byq = defaultdict(list)
for k in allk: byq[k[0]].append(k)
qs = sorted(byq)
def qdiff(sample):
    v = [has[k] for q in sample for k in byq[q] if Y[k]]; c = [has[k] for q in sample for k in byq[q] if not Y[k]]
    return np.mean(v) - np.mean(c)
d0 = qdiff(qs); bs = [qdiff(list(rng.choice(qs, len(qs)))) for _ in range(4000)]
P(f"quoting violating {np.mean([has[k] for k in allk if Y[k]]):.3f} vs compliant {np.mean([has[k] for k in allk if not Y[k]]):.3f}; diff {d0:+.3f}, 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")

P("\n=== 3. Question-cluster sign-flip permutation tests (paired; clusters = questions); also exact McNemar for comparison")
from scipy.stats import binomtest
def cluster_p(a, b, ks, stat, nperm=20000):
    nv = sum(Y[k] for k in ks); nc = len(ks) - nv; tot = defaultdict(float)
    for k in ks:
        d = a[k] - b[k]
        w = {"recall": (d / nv if Y[k] else 0.0), "fpr": (d / nc if not Y[k] else 0.0), "J": (d / nv if Y[k] else -d / nc)}[stat]
        tot[k[0]] += w
    t = np.array(list(tot.values())); obs = t.sum()
    sg = rng.choice([-1, 1], size=(nperm, len(t))); perm = np.abs((sg * t).sum(1))
    return obs, (1 + (perm >= abs(obs) - 1e-12).sum()) / (nperm + 1)
def mcnemar(a, b, ks):
    n10 = sum(a[k] and not b[k] for k in ks); n01 = sum(b[k] and not a[k] for k in ks)
    return binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
for c, ks in conds.items():
    P(f"-- {c}")
    for jn in ("GPT-5.5", "Sonnet 5.5"):
        for cn in ("MiniCheck", "DeBERTa-L not entailed", "RoBERTa not entailed", "MSCG", "DNA"):
            a, b = PRED[(jn, c)], PRED[(cn, c)]
            vk = [k for k in ks if Y[k]]; ck = [k for k in ks if not Y[k]]
            ro, rp = cluster_p(a, b, ks, "recall"); fo, fp = cluster_p(a, b, ks, "fpr"); jo, jp = cluster_p(a, b, ks, "J")
            P(f"  {jn:10} vs {cn:24} recall diff {ro:+.2f} cluster p={rp:.4f} (McNemar viol p={mcnemar(a, b, vk):.1e}) | FPR diff {fo:+.2f} cluster p={fp:.4f} (McNemar compliant p={mcnemar(a, b, ck):.1e}) | J diff {jo:+.2f} cluster p={jp:.4f}")

P("\n=== 4. Precision and PR-AUC (prevalence in header)")
for c, ks in conds.items():
    y = np.array([Y[k] for k in ks]); P(f"-- {c}: n={len(ks)}, violations={y.sum()}, prevalence={y.mean():.3f}")
    for nm, pn, sn in NAMES:
        p = np.array([PRED[(nm, c)][k] for k in ks]); tp = int((p & y).sum()); prec = tp / p.sum() if p.sum() else float("nan")
        pr = f"{average_precision_score(y, [score(sn, c, k) for k in ks]):.2f}" if sn else " -- "
        P(f"  {nm:24} precision {prec:.2f} ({tp}/{p.sum()})  PR-AUC {pr}")

P("\n=== 5. Recall / FPR by answer model, retrieved context")
for nm in ("MiniCheck", "MSCG", "DNA", "GPT-5.5", "Sonnet 5.5"):
    row = []
    for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
        ks = [k for k in allk if k[1] == m]; nv = sum(Y[k] for k in ks); nc = len(ks) - nv
        tp = sum(PRED[(nm, "retrieved")][k] for k in ks if Y[k]); fp = sum(PRED[(nm, "retrieved")][k] for k in ks if not Y[k])
        row.append(f"{m.split('-')[0]}: {tp}/{nv} ({fp}/{nc} FP)")
    P(f"  {nm:12} " + " | ".join(row))
open(os.path.join(H, "results", "revision_analyses.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
