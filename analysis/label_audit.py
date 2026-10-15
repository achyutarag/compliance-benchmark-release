"""Issue 1 audit: what the 17 author-decided labels are, how they overlap the 79 violations / 38 coded failures, and how the headline results move under alternative labels.
Reads only frozen label files, the round-2 reconciliation records and stored auditor outputs. No API calls."""
import contextlib, csv, io, os
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y0, pred, score, NAMES, oracle_keys, rd = g["Y"], g["pred"], g["score"], g["NAMES"], g["oracle_keys"], g["rd"]
out = []
def P(s=""): print(s); out.append(s)
L1 = {(r["qid"], r["answer_model"]): r for r in rd("labels_FROZEN_2026-09-30.csv")}; L2 = {(r["qid"], r["answer_model"]): r for r in rd("labels_ext_FROZEN_2026-10-01.csv")}
key = {r["rec_id"]: (r["qid"], r["answer_model"]) for r in rd("labelling_ext/reconciliation/RECONCILIATION_EXT_KEY_do_not_share.csv")}
dis = {(r["qid"], r["answer_model"]): r for r in rd("labelling_ext/completed/disagreements_ext.csv")}
coded = {(r["qid"], r["answer_model"]) for r in rd("analysis/oracle_failures_coded.csv")}
nc = lambda x: int(x == "NON_COMPLIANT")
# --- the 17
src17 = {k: v for k, v in L2.items() if "approved final" in v["label_source"]}
P(f"=== The author-decided labels (round 2, 'approved final'): {len(src17)}")
rec_of = {v: k for k, v in key.items()}
rows17 = []
for k, v in sorted(src17.items()):
    d = dis.get(k, {}); rows17.append((rec_of.get(k, "?"), k[0], k[1], d.get("compliance_A", "?"), d.get("compliance_B", "?"), v["compliance_label"]))
P("rec  qid  answer_model              A              B              final")
for r in rows17: P(f"{r[0]:4} {r[1]:4} {r[2]:24} {r[3]:14} {r[4]:14} {r[5]}")
fin = Counter(r[5] for r in rows17); P(f"final compliance labels of the 17: {dict(fin)}")
P(f"compliance disagreement between A and B on the 17: {sum(r[3] != r[4] for r in rows17)} (the rest disagree only on match/error type)")
P(f"in the 79 violations: {sum(nc(L2[(r[1], r[2])]['compliance_label']) for r in rows17)}; in the 38 coded failures: {sum((r[1], r[2]) in coded for r in rows17)}")
P(f"answer models: {dict(Counter(r[2] for r in rows17))}; questions: {len({r[1] for r in rows17})}")
# --- resolution pattern of all round-2 compliance disagreements
cd = [(k, d) for k, d in dis.items() if d["compliance_A"] != d["compliance_B"]]
res = Counter()
for k, d in cd:
    f = L2[k]["compliance_label"]; res["final = A" if f == d["compliance_A"] else ("final = B" if f == d["compliance_B"] else "neither")] += 1
P(f"\n=== Round-2 A/B compliance disagreements: {len(cd)}; how the final label relates: {dict(res)}")
P(f"    of which final = violation: {sum(nc(L2[k]['compliance_label']) for k, d in cd)}")
P(f"=== Round 1: label sources of the {sum(nc(v['compliance_label']) for v in L1.values())} violations: {dict(Counter(v['label_source'].split(';')[0] for v in L1.values() if nc(v['compliance_label'])))}")
# --- sensitivity scenarios
def lab_round2(k, who):
    d = dis.get(k)
    if d is None: return nc(L2[k]["compliance_label"])          # both agreed
    return nc(d["compliance_" + who])
SC = {}
SC["base (final labels)"] = (dict(Y0), set())
Y = dict(Y0); k17 = set((r[1], r[2]) for r in rows17)
SC["drop the 17"] = (dict(Y0), k17)
Y = dict(Y0)
for r in rows17:
    k = (r[1], r[2]); Y[k] = int(Y0[k] or r[3] == "NON_COMPLIANT" or r[4] == "NON_COMPLIANT")
SC["17 -> violation if either annotator said so"] = (Y, set())
Y = dict(Y0)
for k in k17: Y[k] = 1
SC["17 all -> violation (extreme)"] = (Y, set())
for who, nm in (("A", "A"), ("B", "B")):
    Y = dict(Y0)
    for k in L2: Y[k] = lab_round2(k, who)
    SC[f"round 2 = annotator {nm}'s labels alone"] = (Y, set())
rec_all = {k for k, v in L2.items() if "reconciled" in v["label_source"] or "approved" in v["label_source"]}
SC["round 2: only items A and B agreed on from the start"] = (dict(Y0), rec_all)
single_v = {k for k, v in L1.items() if nc(v["compliance_label"]) and v["label_source"].startswith("single")}
SC[f"round 1: drop the {len(single_v)} single-annotator violations"] = (dict(Y0), single_v)
SC["drop all uncertain: 17 + reconciled round 2 + single-annotator round-1 violations"] = (dict(Y0), rec_all | single_v)
def metrics(Yd, drop, cond):
    ks = [k for k in (sorted(Y0) if cond == "retrieved" else oracle_keys) if k not in drop]; y = np.array([Yd[k] for k in ks]); res = {}
    for nm, pn, sn in NAMES:
        p = np.array([pred(pn, cond, k) for k in ks]); tp = (p & y).sum(); fp = (p & (1 - y)).sum()
        res[nm] = (tp / y.sum() - fp / (1 - y).sum(), roc_auc_score(y, [score(sn, cond, k) for k in ks]) if sn else None)
    return len(ks), int(y.sum()), res
show = ["MiniCheck", "DeBERTa-L not entailed", "MSCG", "DNA", "GPT-5.5", "Sonnet 5.5"]
for cond in ("retrieved", "oracle"):
    P(f"\n=== Sensitivity of the headline results, {cond} context: informedness J (AUROC for scored auditors)")
    P(f"{'scenario':84}{'n':>5}{'viol':>5}" + "".join(f"{s[:11]:>14}" for s in show))
    for nm, (Yd, drop) in SC.items():
        n, v, res = metrics(Yd, drop, cond)
        P(f"{nm[:83]:84}{n:>5}{v:>5}" + "".join(f"{res[s][0]:>+8.2f}" + (f" ({res[s][1]:.2f})" if res[s][1] is not None else "       ") for s in show))
open(os.path.join(H, "results", "label_audit.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
