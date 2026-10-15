"""Question-plus-answer hypothesis condition (grounding/GROUNDING_PROTOCOL.md, Amendment 4): recall, FPR, informedness, precision, AUROC with question-bootstrap CIs,
next to the answer-only values for the same checkers, retrieved and oracle context. Uses whichever of the three score files are complete."""
import contextlib, io, json, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE); G = os.path.join(H, "grounding")
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, oracle_keys = g["Y"], g["oracle_keys"]
load = lambda f: {tuple(json.loads(l)["id"].split("|")): json.loads(l) for l in open(os.path.join(G, f), encoding="utf-8")} if os.path.exists(os.path.join(G, f)) else {}
SRC = {"MiniCheck": ("scores_minicheck_whole.jsonl", "scores_minicheck_qa.jsonl", "support"), "HHEM": ("scores_hhem_whole.jsonl", "scores_hhem_qa.jsonl", "support"),
       "DeBERTa-L": ("scores_nli_large_whole.jsonl", "scores_nli_large_qa.jsonl", "nli")}
rng = np.random.default_rng(20261015); out = []
def P(s=""): print(s); out.append(s)
def rules(nm, rec):   # (score for violation, {rule: flag})
    if nm == "DeBERTa-L": return {"contradiction": rec["argmax_contra"], "not entailed": int(rec["E"] < 0.5)}, {"contradiction": rec["C"], "not entailed": -rec["E"]}
    sup = rec["support"] if isinstance(rec, dict) else rec; return {"support<0.5": int(sup < 0.5)}, {"support<0.5": -sup}
for nm, (fa, fq, kind) in SRC.items():
    A_, Q_ = load(fa), load(fq)
    if len(Q_) < 792: P(f"{nm}: question+answer scores incomplete ({len(Q_)}/792); skipped"); continue
    for cond, ks in (("retrieved", sorted(Y)), ("oracle", oracle_keys)):
        y = np.array([Y[k] for k in ks]); byq = defaultdict(list)
        for i, k in enumerate(ks): byq[k[0]].append(i)
        qs = list(byq)
        for rule in rules(nm, A_[(ks[0][0], ks[0][1], cond)])[0]:
            row = []
            for tag, D in (("answer only", A_), ("question + answer", Q_)):
                fl = np.array([rules(nm, D[(k[0], k[1], cond)])[0][rule] for k in ks]); sc = np.array([rules(nm, D[(k[0], k[1], cond)])[1][rule] for k in ks])
                tp = int((fl & y).sum()); fp = int((fl & (1 - y)).sum()); au = roc_auc_score(y, sc); v = []
                for _ in range(1500):
                    idx = [i for q in rng.choice(qs, len(qs)) for i in byq[q]]
                    if 0 < y[idx].sum() < len(idx): v.append(roc_auc_score(y[idx], sc[idx]))
                row.append(f"{tag}: recall {tp}/{y.sum()} ({tp / y.sum():.0%}) FPR {fp / (1 - y).sum():.0%} J {tp / y.sum() - fp / (1 - y).sum():+.2f} prec {tp / max(1, fl.sum()):.2f} AUROC {au:.2f} [{np.percentile(v, 2.5):.2f}, {np.percentile(v, 97.5):.2f}]")
            P(f"{nm:10} {rule:14} {cond:9} | " + " | ".join(row))
open(os.path.join(H, "results", "qa_hypothesis_results.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
