"""Full-policy + question judge baseline (judges/FULL_POLICY_PREREGISTRATION.md): results on the 396 named-institution answers."""
import contextlib, csv, io, os
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, pred, NAMES, rd = g["Y"], g["pred"], g["NAMES"], g["rd"]
F = defaultdict(dict); Q = defaultdict(dict)
for r in rd("judges/full_predictions.csv"): F[r["provider"]][(r["qid"], r["answer_model"])] = int(r["pred_violation"]) if r["pred_violation"] != "" else None
for r in rd("judges/q_predictions.csv"):
    if r["condition"] == "retrieved": Q[r["provider"]][(r["qid"], r["answer_model"])] = int(r["pred_violation"])
keys = sorted(set(F["openai"]) & set(F["anthropic"])); bad = [k for k in keys if F["openai"][k] is None or F["anthropic"][k] is None]
y_all = {k: Y[k] for k in keys}; nv = sum(y_all.values()); nc = len(keys) - nv
rng = np.random.default_rng(20261014); out = []
def P(s): print(s); out.append(s)
P(f"items: {len(keys)} answers, violations {nv}, compliant {nc}; unparseable/failed: {len(bad)}")
ks = [k for k in keys if k not in bad]
byq = defaultdict(list)
for k in ks: byq[k[0]].append(k)
qs = sorted(byq)
def stats(p, kk):
    v = [k for k in kk if Y[k]]; c = [k for k in kk if not Y[k]]
    tp = sum(p[k] for k in v); fp = sum(p[k] for k in c); fl = sum(p[k] for k in kk)
    return tp / len(v), fp / len(c), tp / max(1, fl), tp, len(v), fp, len(c), fl
def jboot(p):
    vals = []
    for _ in range(3000):
        kk = [k for q in rng.choice(qs, len(qs)) for k in byq[q]]; r, f, *_ = stats(p, kk); vals.append(r - f)
    return np.percentile(vals, [2.5, 97.5])
def cl(a, b, stat, n=20000):
    tot = defaultdict(float)
    for k in ks:
        d = a[k] - b[k]; tot[k[0]] += {"recall": d / nv if Y[k] else 0.0, "fpr": d / nc if not Y[k] else 0.0, "J": d / nv if Y[k] else -d / nc}[stat]
    t = np.array(list(tot.values())); obs = t.sum(); sg = rng.choice([-1, 1], size=(n, len(t)))
    return obs, (1 + (np.abs((sg * t).sum(1)) >= abs(obs) - 1e-12).sum()) / (n + 1)
P("\n=== Detection on the same answers (retrieved-chunk judges are restricted to these items)")
P(f"{'auditor':44}{'recall':>14}{'FPR':>14}{'J [95% CI]':>24}{'precision':>12}")
rows = {}
for prov, nm in (("openai", "GPT-5.5"), ("anthropic", "Sonnet 5.5")):
    A = {k: F[prov][k] for k in ks}
    B = {k: pred("gpt" if prov == "openai" else "sonnet", "retrieved", k) for k in ks}
    Qp = {k: Q[prov][k] for k in ks}
    for lab, p in ((f"{nm} full policy + question", A), (f"{nm} 2 chunks + question", Qp), (f"{nm} 2 chunks, no question (main)", B)):
        r, f, pr, tp, nvv, fp, ncc, fl = stats(p, ks); lo, hi = jboot(p); rows[lab] = p
        P(f"{lab:44}{tp:>6}/{nvv} {r:4.0%}{fp:>7}/{ncc} {f:4.0%}   {r - f:+.2f} [{lo:+.2f}, {hi:+.2f}]{pr:>12.2f}")
for nm, pn, sn in NAMES:
    if pn in ("gpt", "sonnet"): continue
    p = {k: pred(pn, "retrieved", k) for k in ks}; r, f, pr, tp, nvv, fp, ncc, fl = stats(p, ks); rows[nm] = p
    P(f"{nm + ' (retrieved)':44}{tp:>6}/{nvv} {r:4.0%}{fp:>7}/{ncc} {f:4.0%}   {r - f:+.2f}{'':>16}{pr:>12.2f}")
best = max((nm for nm, pn, sn in NAMES if pn not in ("gpt", "sonnet")), key=lambda n: sum(rows[n][k] for k in ks if Y[k]) / nv - sum(rows[n][k] for k in ks if not Y[k]) / nc)
P(f"best grounding check / modular auditor by informedness on these answers: {best}")
P("\n=== Question-level permutation tests (full policy + question minus comparator)")
for prov, nm in (("openai", "GPT-5.5"), ("anthropic", "Sonnet 5.5")):
    A = {k: F[prov][k] for k in ks}
    for cmpn in (f"{nm} 2 chunks + question", f"{nm} 2 chunks, no question (main)", best):
        for stat in ("recall", "fpr", "J"):
            o, pv = cl(A, rows[cmpn], stat); P(f"  {nm} full vs [{cmpn}] {stat}: {o:+.2f}, p = {pv:.4f}")
P("\n=== Recall / FPR by answer model (full policy + question)")
for prov, nm in (("openai", "GPT-5.5"), ("anthropic", "Sonnet 5.5")):
    row = []
    for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
        kk = [k for k in ks if k[1] == m]; r, f, pr, tp, nvv, fp, ncc, fl = stats({k: F[prov][k] for k in kk}, kk); row.append(f"{m.split('-')[0]}: {tp}/{nvv} ({fp}/{ncc} FP)")
    P(f"  {nm:11} " + " | ".join(row))
open(os.path.join(H, "results", "full_policy_analysis.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
