"""Secondary numbers for the paper over all 416 real-RAG answers (ablation rows, DNA row, AUROC, precision, gate attribution, clause splits)."""
import csv, os
from collections import Counter
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
ORIG, EXT = ("claude-haiku-4-5", "gpt-4o-mini"), ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")
A = {(r["qid"], r["answer_model"]): r for r in rd(os.path.join(RES, "all416_per_answer.csv"))}
sig = {}
for r in rd(os.path.join(RES, "mscg_predictions.csv")):
    sig[(r["qid"], r["answer_model"])] = {k: float(r[k]) for k in ("C", "E", "N", "sim", "prec")} | {"gate": r["gate"]}
dna = {}
for r in rd(os.path.join(HERE, "dna", "results", "dna_real_rag_per_answer.csv")):
    dna[(r["qid"], r["answer_model"])] = {"mc": float(r["ret_max_contra"]), "ms": float(r["ret_min_support"]), "wC": float(r["ret_wholepair_C"]),
                                          "wE": float(r["ret_wholepair_E"]), "omc": r.get("orc_max_contra"), "oms": r.get("orc_min_support"),
                                          "owC": r.get("orc_wholepair_C"), "owE": r.get("orc_wholepair_E")}
for r in rd(os.path.join(RES, "ext_local_auditors.csv")):
    k = (r["qid"], r["answer_model"])
    sig[k] = {x: float(r[x]) for x in ("C", "E", "N", "sim", "prec")} | {"gate": r["mscg_gate"]}
    dna[k] = {"mc": float(r["dna_max_contra"]), "ms": float(r["dna_min_support"]), "wC": float(r["C"]), "wE": float(r["E"]),
              "omc": r.get("dna_oracle_max_contra"), "oms": r.get("dna_oracle_min_support"), "owC": r.get("C_oracle"), "owE": r.get("E_oracle")}
K = sorted(A); y = np.array([int(A[k]["y"]) for k in K]); n1, n0 = y.sum(), (1 - y).sum()
assert len(K) == 416 and set(sig) == set(K) == set(dna)


def mscg(s, use_prec=True, use_sim=True):
    if s["C"] > 0.35: return 1
    if s["N"] > 0.70:
        if s["C"] > 0.15: return 1
        if use_prec and s["prec"] > 0.50: return 1
        if use_sim and s["sim"] < 0.42: return 1
        return 0
    return 0 if s["E"] - s["C"] >= 0.50 else 1


full = np.array([mscg(sig[k]) for k in K]); assert (full == np.array([int(A[k]["mscg"]) for k in K])).all()
def mcn(a, b):
    x, z = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    return "(=)" if x + z == 0 else f"(p={binomtest(min(x, z), x + z).pvalue:.3f})"
print("LOO real RAG 416:", f"full {(full == y).mean():.1%}")
for nm, kw in (("-prec", dict(use_prec=False)), ("-sim", dict(use_sim=False)), ("NLI only", dict(use_prec=False, use_sim=False))):
    p = np.array([mscg(sig[k], **kw) for k in K]); print(f"   {nm}: {(p == y).mean():.1%} {mcn(full, p)}")
contra = np.array([(sig[k]["C"] > 0.35) or (sig[k]["N"] > 0.70 and sig[k]["C"] > 0.15) for k in K])
simg = np.array([sig[k]["sim"] < 0.42 for k in K]); precg = np.array([sig[k]["prec"] > 0.5 for k in K])
for nm, g in (("contradiction", contra), ("similarity", simg), ("precedent", precg)):
    print(f"ISO {nm}: recall {int((g & (y == 1)).sum())}/{n1} ({(g & (y == 1)).sum() / n1:.0%}), FPR {(g & (y == 0)).sum() / n0:.0%}")
print("precedent score range:", round(min(s["prec"] for s in sig.values()), 3), round(max(s["prec"] for s in sig.values()), 3))
missed = [sig[k]["gate"] for k in K if A[k]["y"] == "1" and not full[K.index(k)]]
print("MSCG gate on missed violations:", Counter(missed), "of", n1)

# DNA row
gA = np.array([dna[k]["mc"] > 0.5 for k in K]); gB = np.array([dna[k]["ms"] < 0.5 for k in K]); D = (gA | gB).astype(int)
print(f"DNA 416: acc {(D == y).mean():.1%}; -GateA acc {(gB.astype(int) == y).mean():.1%}; GateA alone recall {int((gA & (y == 1)).sum())}/{n1} "
      f"({(gA & (y == 1)).sum() / n1:.0%}) FPR {(gA & (y == 0)).sum() / n0:.0%}; GateB alone recall {int((gB & (y == 1)).sum())}/{n1} "
      f"({(gB & (y == 1)).sum() / n1:.0%}) FPR {(gB & (y == 0)).sum() / n0:.0%}; answers with >=1 unsupported claim {int(gB.sum())}/416; "
      f"flags {int(D.sum())}/416")
rng = np.random.default_rng(0)
def auc(yv, sv):
    yv, sv = np.asarray(yv), np.asarray(sv); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        if 0 < yv[i].sum() < len(i): bs.append(roc_auc_score(yv[i], sv[i]))
    return f"{roc_auc_score(yv, sv):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"
O = [k for k in K if dna[k]["omc"] not in (None, "")]; yo = [int(A[k]["y"]) for k in O]
print(f"AUROC (retrieved 416 | oracle {len(O)}):")
for nm, f_r, f_o in (("DNA max contra", lambda k: dna[k]["mc"], lambda k: float(dna[k]["omc"])),
                     ("DNA 1-min support", lambda k: -dna[k]["ms"], lambda k: -float(dna[k]["oms"])),
                     ("whole-pair C", lambda k: dna[k]["wC"], lambda k: float(dna[k]["owC"])),
                     ("whole-pair 1-E", lambda k: -dna[k]["wE"], lambda k: -float(dna[k]["owE"]))):
    print(f"   {nm:18} {auc(y, [f_r(k) for k in K])} | {auc(yo, [f_o(k) for k in O])}")

# precision, judges by clause, accuracy where clause present
for col in ("naive", "mscg", "gpt55_retrieved", "sonnet55_retrieved"):
    p = np.array([int(A[k][col]) for k in K]); print(f"precision {col}: {int((p & y).sum())}/{int(p.sum())}")
for col in ("gpt55_retrieved", "sonnet55_retrieved", "mscg", "naive"):
    for cl in ("YES", "NO"):
        s = [k for k in K if A[k]["clause"] == cl]; ys = np.array([int(A[k]["y"]) for k in s]); p = np.array([int(A[k][col]) for k in s])
        print(f"  {col:20} clause={cl}: acc {(p == ys).mean():.1%} (n={len(s)}), FPR {(p & (1 - ys)).sum()}/{(1 - ys).sum()} = {(p & (1 - ys)).sum() / (1 - ys).sum():.0%}")
lab = {(r["qid"], r["answer_model"]): r for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv") for r in rd(os.path.join(HERE, f))}
for m in ORIG + EXT:
    s = [lab[k] for k in lab if k[1] == m]
    print(f"matches gold {m}: YES {sum(r['matches_gold'] == 'YES' for r in s) / len(s):.0%}, violations {sum(int(r['y_violation']) for r in s)}")
# within-family / independent pooled recall for judges, MSCG on ext vs orig recall CIs printed in analysis_416
