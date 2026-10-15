"""Auditor AUROC / informedness by answer-model group (stronger: Haiku, gpt-4o-mini; small: Llama, gemma), retrieved and oracle context, with question-bootstrap CIs."""
import contextlib, io, json, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, pred, score, NAMES, oracle_keys = g["Y"], g["pred"], g["score"], g["NAMES"], g["oracle_keys"]
HH = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding", "scores_hhem_whole.jsonl"), encoding="utf-8")}
GROUPS = {"stronger models (Haiku, gpt-4o-mini)": ("claude-haiku-4-5", "gpt-4o-mini"), "small models (Llama, gemma)": ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")}
SCORED = [(nm, sn) for nm, pn, sn in NAMES if sn] + [("HHEM", "hhem")]
rng = np.random.default_rng(20261015); out = []
def P(s=""): print(s); out.append(s)
def sc(nm, sn, cond, k): return 1 - HH[(k[0], k[1], cond)] if sn == "hhem" else score(sn, cond, k)
for cond, base in (("retrieved", sorted(Y)), ("oracle", oracle_keys)):
    P(f"=== {cond} context: AUROC [95% question-bootstrap CI] by answer-model group")
    P(f"{'auditor':26}" + "".join(f"{gn:>46}" for gn in list(GROUPS) + ["all (Table 1)"]))
    cache = {}
    for gn, ms in list(GROUPS.items()) + [("all", None)]:
        ks = [k for k in base if ms is None or k[1] in ms]; y = np.array([Y[k] for k in ks]); byq = defaultdict(list)
        for i, k in enumerate(ks): byq[k[0]].append(i)
        cache[gn] = (ks, y, byq)
    for nm, sn in SCORED:
        cells = []
        for gn in list(GROUPS) + ["all"]:
            ks, y, byq = cache[gn]; s_ = np.array([sc(nm, sn, cond, k) for k in ks]); au = roc_auc_score(y, s_); qs = list(byq); v = []
            for _ in range(2000):
                idx = [i for q in rng.choice(qs, len(qs)) for i in byq[q]]
                if 0 < y[idx].sum() < len(idx): v.append(roc_auc_score(y[idx], s_[idx]))
            cells.append(f"{au:.2f} [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}] ({int(y.sum())} viol, n={len(ks)})")
        P(f"{nm:26}" + "".join(f"{c:>46}" for c in cells))
    P("  informedness J (recall - FPR) of the decision rules:")
    for nm, pn, sn in NAMES:
        cells = []
        for gn in list(GROUPS) + ["all"]:
            ks, y, byq = cache[gn]; p = np.array([pred(pn, cond, k) for k in ks]); cells.append(f"{(p & y).sum() / y.sum() - (p & (1 - y)).sum() / (1 - y).sum():+.2f}")
        P(f"    {nm:24}" + "".join(f"{c:>46}" for c in cells))
    P()
open(os.path.join(H, "results", "auroc_by_answer_model.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
