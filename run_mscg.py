"""
Frozen MSCG (guard_v10, fit once on the original 13-example fit split) on the 208 labelled real-RAG answers.
Same pairing as the pilot: policy = concatenated retrieved chunks, response = generated answer.
No refitting, no threshold changes. Outputs results/mscg_predictions.csv and prints the pre-planned analyses.
"""
import csv, io, json, os, contextlib
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
nb = json.load(open(os.path.join(ROOT, "solutions_final_v3.ipynb"), encoding="utf-8"))
cells = {c["metadata"].get("id"): "".join(c["source"]) for c in nb["cells"]}
g = {"__name__": "__nb__"}
cwd = os.getcwd(); os.chdir(ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    exec(cells["LshaAK9fJXIu"], g); exec(cells["gH3xCbjrXkpb"], g)
    s32 = cells["yOGAQ_CkXkpb"]; exec(s32[:s32.index("df_v10 = guard_v10.run_inference")], g)
os.chdir(cwd)
guard = g["guard_v10"]

labels = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(HERE, "labels_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
ans = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    for r in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r
tok = guard.nli_model.tokenizer

rows = []
for k, lab in sorted(labels.items()):
    a = ans[k]; policy = " ".join(c["text"] for c in a["retrieved"]); resp = a["answer"]
    C, E, N = (float(x) for x in guard.nli_model.predict([policy, resp], apply_softmax=True))
    ep, er = guard._embed(policy), guard._embed(resp); sim = float(ep @ er); prec = guard._precedent_score(policy, resp)
    if C > 0.35: pred, gate = 1, "gate1_contradiction"
    elif N > 0.70:
        if C > 0.15: pred, gate = 1, "gate2_background_leak"
        elif prec > 0.50: pred, gate = 1, "gate3_precedent"
        elif sim < 0.42: pred, gate = 1, "gate4_similarity"
        else: pred, gate = 0, "safe_high_neutral"
    else:
        pred, gate = (0 if E - C >= 0.50 else 1), "gate5_margin"
    n_tok = len(tok(policy, resp)["input_ids"])
    rows.append({**{x: lab[x] for x in ("qid", "answer_model", "institution", "structures", "gold_verdict", "compliance_label",
                                          "y_violation", "matches_gold", "error_type", "clause_in_retrieved")},
                 "C": C, "E": E, "N": N, "sim": sim, "prec": prec, "gate": gate, "pred_mscg": pred,
                 "pred_naive": int(np.argmax([C, E, N]) == 0), "nli_tokens": n_tok, "nli_truncated": int(n_tok > 512),
                 "answer_words": len(resp.split())})

# sanity: reconstructed gates must equal guard.run_inference on the same pairs
ref = guard.run_inference([(" ".join(c["text"] for c in ans[(r["qid"], r["answer_model"])]["retrieved"]),
                            ans[(r["qid"], r["answer_model"])]["answer"], r["y_violation"], "x", r["qid"], "x") for r in rows])
assert list(ref["pred_guard"]) == [r["pred_mscg"] for r in rows], "gate reconstruction mismatch"
with open(os.path.join(HERE, "results", "mscg_predictions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

y = np.array([int(r["y_violation"]) for r in rows])
def ci(k, n):
    if n == 0: return "n/a"
    c = binomtest(k, n).proportion_ci(method="exact"); return f"[{c.low:.0%}, {c.high:.0%}]"
def report(name, p, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask; yy, pp = y[m], p[m]
    tp, fn = int(((yy == 1) & (pp == 1)).sum()), int(((yy == 1) & (pp == 0)).sum())
    fp, tn = int(((yy == 0) & (pp == 1)).sum()), int(((yy == 0) & (pp == 0)).sum())
    n = len(yy); acc = (tp + tn) / n
    rec = f"{tp}/{tp+fn} {tp/(tp+fn):.0%} {ci(tp, tp+fn)}" if tp + fn else "n/a"
    fpr = f"{fp}/{fp+tn} {fp/(fp+tn):.0%}" if fp + tn else "n/a"
    prc = f"{tp}/{tp+fp}" if tp + fp else "-"
    print(f"  {name:28} n={n:3d} acc={acc:5.1%} {ci(tp+tn, n):12} recall(violations)={rec:22} FPR={fpr:12} precision={prc}")
P = np.array([r["pred_mscg"] for r in rows]); NV = np.array([r["pred_naive"] for r in rows]); AC = np.zeros(len(y), int)
print(f"\nMSCG (frozen guard_v10) on {len(rows)} real RAG answers | violations: {y.sum()} | NLI inputs >512 tokens (truncated): {sum(r['nli_truncated'] for r in rows)}")
print("\n=== Overall ===")
for nme, p in (("always-compliant", AC), ("naive NLI argmax", NV), ("MSCG", P)): report(nme, p)
b = int(((P == y) & (NV != y)).sum()); c = int(((P != y) & (NV == y)).sum())
print(f"  McNemar MSCG vs naive: MSCG-only-correct={b}, naive-only-correct={c}, p={binomtest(min(b,c), b+c).pvalue if b+c else float('nan'):.3f}")
print("\n=== By answer model ===")
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    mk = np.array([r["answer_model"] == m for r in rows]); print(f" {m}: violations {y[mk].sum()}/{mk.sum()}"); report("MSCG", P, mk); report("naive NLI", NV, mk)
print("\n=== By clause in retrieved text (retrieval vs generation) ===")
for cl in ("YES", "PARTIAL", "NO", "N/A"):
    mk = np.array([r["clause_in_retrieved"] == cl for r in rows]); print(f" clause={cl}: violations {y[mk].sum()}/{mk.sum()}"); report("MSCG", P, mk)
print("\n=== Gate that resolved each verdict ===")
from collections import Counter
for gname, cnt in Counter(r["gate"] for r in rows).most_common():
    gv = [r for r in rows if r["gate"] == gname]
    print(f"  {gname:22} {cnt:3d} | on violations {sum(int(r['y_violation']) for r in gv):2d} | on compliant {sum(1-int(r['y_violation']) for r in gv):3d}")
print("\n=== Component ablation (predictions with gates removed) ===")
def pred(r, use_prec=True, use_sim=True):
    if r["C"] > 0.35: return 1
    if r["N"] > 0.70:
        if r["C"] > 0.15: return 1
        if use_prec and r["prec"] > 0.50: return 1
        if use_sim and r["sim"] < 0.42: return 1
        return 0
    return 0 if r["E"] - r["C"] >= 0.50 else 1
for nme, kw in (("- precedent gate", dict(use_prec=False)), ("- similarity gate", dict(use_sim=False)), ("NLI gates only", dict(use_prec=False, use_sim=False))):
    report(nme, np.array([pred(r, **kw) for r in rows]))
print(f"  precedent score range: {min(r['prec'] for r in rows):.3f}-{max(r['prec'] for r in rows):.3f} | >0.5: {sum(r['prec'] > 0.5 for r in rows)}")
print("\n=== By reasoning structure (violations detected / violations; false positives / compliant) ===")
S = sorted({s for r in rows for s in r["structures"].split(";") if s}, key=lambda s: int(s[1:]))
for s in S:
    mk = np.array([s in r["structures"].split(";") for r in rows])
    v, cmp_ = (y == 1) & mk, (y == 0) & mk
    print(f"  {s:4} n={mk.sum():3d}  detected {int((P[v] == 1).sum())}/{v.sum():2d}   FP {int((P[cmp_] == 1).sum())}/{cmp_.sum()}")
