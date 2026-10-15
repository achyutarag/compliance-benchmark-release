"""Independent audit of the paper's main numbers, recomputed from raw per-item files (not from the merged summary CSVs)."""
import csv, json, os, re
import numpy as np
from collections import Counter
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: list(csv.DictReader(open(os.path.join(H, p), encoding="utf-8")))
# labels
Y = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in rd(f): Y[(r["qid"], r["answer_model"])] = int(r["compliance_label"] == "NON_COMPLIANT")
assert len(Y) == 416 and sum(Y.values()) == 79
# judges (compliance prompt)
J = {}
for f in ("judges/judge_predictions.csv", "judges/judge_predictions_ext.csv"):
    for r in rd(f): J[(r["provider"], r["condition"], r["qid"], r["answer_model"])] = int(r["pred_violation"])
# MSCG + DNA raw
ms = {(r["qid"], r["answer_model"]): r for r in rd("results/mscg_predictions.csv")}
mo = {(r["qid"], r["answer_model"]): r for r in rd("results/mscg_oracle_context_diagnostic.csv")}
dn = {(r["qid"], r["answer_model"]): r for r in rd("dna/results/dna_real_rag_per_answer.csv")}
ex = {(r["qid"], r["answer_model"]): r for r in rd("results/ext_local_auditors.csv")}
MC = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding/scores_minicheck_whole.jsonl"), encoding="utf-8")}
NL = {tuple(json.loads(l)["id"].split("|")): json.loads(l) for l in open(os.path.join(H, "grounding/scores_nli_large_whole.jsonl"), encoding="utf-8")}
RB = {tuple(json.loads(l)["id"].split("|")): json.loads(l) for l in open(os.path.join(H, "grounding/scores_roberta_mnli_whole.jsonl"), encoding="utf-8")}
oracle_keys = sorted({(q, m) for (p, c, q, m) in J if c == "oracle" and p == "openai"})
assert len(oracle_keys) == 376
def pred(name, cond, k):
    q, m = k
    if name == "gpt": return J[("openai", cond, q, m)]
    if name == "sonnet": return J[("anthropic", cond, q, m)]
    if name == "mscg":
        if cond == "retrieved": return int(ms[k]["pred_mscg"]) if k in ms else int(ex[k]["mscg"])
        return int(mo[k]["pred_oracle"]) if k in mo else int(ex[k]["mscg_oracle"])
    if name == "dna":
        if cond == "retrieved": return int(dn[k]["dna_ret"]) if k in dn else int(ex[k]["dna"])
        return int(dn[k]["dna_orc"]) if k in dn else int(ex[k]["dna_oracle"])
    if name == "minicheck": return int(MC[(q, m, cond)] < 0.5)
    if name == "deb_contra": return NL[(q, m, cond)]["argmax_contra"]
    if name == "deb_notent": return int(NL[(q, m, cond)]["E"] < 0.5)
    if name == "rob_contra": return RB[(q, m, cond)]["argmax_contra"]
    if name == "rob_notent": return int(RB[(q, m, cond)]["E"] < 0.5)
def score(name, cond, k):
    q, m = k
    if name == "mscg_C": return float(ms[k]["C"]) if k in ms and cond == "retrieved" else (float(ex[k]["C"]) if cond == "retrieved" else (float(dn[k]["orc_wholepair_C"]) if k in dn else float(ex[k]["C_oracle"])))
    if name == "dna_supp": return -(float(dn[k]["ret_min_support"]) if k in dn else float(ex[k]["dna_min_support"])) if cond == "retrieved" else -(float(dn[k]["orc_min_support"]) if k in dn else float(ex[k]["dna_oracle_min_support"]))
    if name == "minicheck": return -MC[(q, m, cond)]
    if name == "deb_contra": return NL[(q, m, cond)]["C"]
    if name == "deb_notent": return -NL[(q, m, cond)]["E"]
    if name == "rob_contra": return RB[(q, m, cond)]["C"]
    if name == "rob_notent": return -RB[(q, m, cond)]["E"]
NAMES = [("MiniCheck", "minicheck", "minicheck"), ("DeBERTa-L contradiction", "deb_contra", "deb_contra"), ("DeBERTa-L not entailed", "deb_notent", "deb_notent"),
         ("RoBERTa contradiction", "rob_contra", "rob_contra"), ("RoBERTa not entailed", "rob_notent", "rob_notent"), ("MSCG", "mscg", "mscg_C"),
         ("DNA", "dna", "dna_supp"), ("GPT-5.5", "gpt", None), ("Sonnet 5.5", "sonnet", None)]
for cond, keys in (("retrieved", sorted(Y)), ("oracle", oracle_keys)):
    y = np.array([Y[k] for k in keys]); print(f"\n=== {cond}: n={len(keys)}, violations={y.sum()}")
    for nm, pn, sn in NAMES:
        p = np.array([pred(pn, cond, k) for k in keys]); tp = int((p & y).sum()); fp = int((p & (1 - y)).sum())
        au = f"{roc_auc_score(y, [score(sn, cond, k) for k in keys]):.2f}" if sn else " -- "
        print(f"  {nm:24} recall {tp:2d}/{y.sum()} ({tp/y.sum():4.0%}) FPR {fp/(1-y).sum():4.0%}  J {tp/y.sum()-fp/(1-y).sum():+.2f}  AUROC {au}  flags {p.sum()}")
# claim: violating answers are as grounded as compliant ones
ans = {}; ret = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for r in map(json.loads, open(os.path.join(H, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r["answer"]; ret[(r["qid"], m)] = " ".join(c["text"] for c in r["retrieved"])
g5 = lambda t: {tuple(w[i:i + 5]) for w in [re.findall(r"[a-z0-9$]+", t.lower())] for i in range(len(w) - 4)}
has = {k: int(len(g5(ans[k]) & g5(ret[k])) > 0) for k in Y}
v = [has[k] for k in Y if Y[k]]; c = [has[k] for k in Y if not Y[k]]
print(f"\nQUOTING: violations {sum(v)}/{len(v)} = {np.mean(v):.0%}, compliant {sum(c)}/{len(c)} = {np.mean(c):.0%}; Fisher exact p = {fisher_exact([[sum(v), len(v)-sum(v)], [sum(c), len(c)-sum(c)]])[1]:.3f}")
sv = [MC[(k[0], k[1], 'retrieved')] for k in Y if Y[k]]; sc = [MC[(k[0], k[1], 'retrieved')] for k in Y if not Y[k]]
print(f"MiniCheck support: median violating {np.median(sv):.3f} vs compliant {np.median(sc):.3f}; Mann-Whitney two-sided p = {mannwhitneyu(sv, sc).pvalue:.3f}")
# clause labels v2 vs round 1
v2 = {r["qid"]: r for r in rd("clause_labels_v2_FROZEN_2026-10-11.csv")}
print("\nCLAUSE v2: counts", Counter(r["clause_in_retrieved_v2"] for r in v2.values() if r["annotator3_label"]))
r1n = [q for q, r in v2.items() if r["round1_label"] == "NO"]; r1y = [q for q, r in v2.items() if r["round1_label"] == "YES"]
print(f"round-1 NO labels: {len(r1n)}; changed in final: {sum(v2[q]['clause_in_retrieved_v2'] != 'NO' for q in r1n)} | round-1 YES labels: {len(r1y)}; changed in final: {sum(v2[q]['clause_in_retrieved_v2'] != 'YES' for q in r1y)}")
print("stronger-model violations (in-scope) by clause:", Counter(v2[k[0]]["clause_in_retrieved_v2"] for k in Y if Y[k] and k[1] in ("claude-haiku-4-5", "gpt-4o-mini") and v2[k[0]]["clause_in_retrieved_v2"] != "N/A"))
print("small-model violations (in-scope) by clause:", Counter(v2[k[0]]["clause_in_retrieved_v2"] for k in Y if Y[k] and k[1] in ("Llama-3.2-3B-Instruct", "gemma-2-2b-it") and v2[k[0]]["clause_in_retrieved_v2"] != "N/A"))
