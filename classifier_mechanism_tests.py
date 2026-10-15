"""
Tests of the paper's hypothesised mechanism for the precedent classifier's failure:
  "it learned the surface style of its four positive training examples under strong regularization".

Predictions written BEFORE running (post-hoc analyses; reported as such):
  T1 Regularization: if strong regularization (C=0.1) is the cause, larger C should widen the score range AND
     raise held-out PF recall. If scores widen but recall stays low, regularization explains the compressed scores,
     not the failure.
  T2 Style proxy: training PF examples state no specific amount ("abstract"). If the classifier learned that style,
     held-out PF items WITHOUT an amount should score higher / be detected more often than PF items WITH an amount.
  T3 Proximity: if the classifier tracks surface similarity to its training examples, its score on held-out items
     should correlate with cosine similarity to the centroid of the 4 training PF embeddings, within each label
     (i.e., not only because PF items are similar to each other).
  T4 Style-diverse training: the recalibrated classifier (41-example fit split incl. concrete PF) is already reported;
     here we only repeat T1 for it to check whether its failure on the Claude set is also regularization-independent.
"""
import csv, io, json, os, re, contextlib
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")   # notebook and synthetic sets
nb = json.load(open(os.path.join(ROOT, "solutions_final_v3.ipynb"), encoding="utf-8"))
cells = {c["metadata"].get("id"): "".join(c["source"]) for c in nb["cells"]}
g = {"__name__": "__nb__"}; cwd = os.getcwd(); os.chdir(ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    exec(cells["LshaAK9fJXIu"], g); exec(cells["gH3xCbjrXkpb"], g)
    s32 = cells["yOGAQ_CkXkpb"]; exec(s32[:s32.index("df_v10 = guard_v10.run_inference")], g)
os.chdir(cwd)
guard = g["guard_v10"]; fit13 = g["fit_only"]

def rows(path, domain=None, splits=None):
    out = []
    for r in csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8")):
        if r["keep"].strip() != "1" or (domain and r.get("domain") != domain) or (splits and r["split"] not in splits): continue
        out.append((r["policy"], r["response"], int(r["verified_label"]), r["category"]))
    return out
SETS = {"fresh": rows("fresh_gift_final.csv"), "claude": rows("cross_gen_claude_final.csv"),
        "retention": rows("multi_domain_final_v3.csv", "data_retention"), "coi": rows("multi_domain_final_v3.csv", "coi_disclosure"),
        "ai_data": rows("multi_domain_final_v3.csv", "ai_data_use")}
fit41 = [(p, r, y, c) for p, r, y, c, n, s in g["load_verified_dataset_v2"]("output_v5.csv") if s == "fit"] if False else \
        [(r["policy"], r["response"], int(r["verified_label"]), r["category"]) for r in csv.DictReader(open(os.path.join(ROOT, "output_v5.csv"), encoding="utf-8"))
         if r["keep"].strip() == "1" and r["split"] == "fit"]
fit13r = [(p, r, y, c) for p, r, y, c, n, s in fit13]

E = {}
def feat(p, r):
    k = (p, r)
    if k not in E:
        er, ep = guard._embed(r), guard._embed(p); E[k] = (er, np.concatenate([er, er - ep]))
    return E[k]
def train(data, C):
    X = [feat(p, r)[1] for p, r, y, c in data if c in ("Precedent Framing", "Safe", "Adversarial")]
    Y = [y for p, r, y, c in data if c in ("Precedent Framing", "Safe", "Adversarial")]
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X, Y)
AMOUNT = re.compile(r"\$\s?\d|\b\d+(?:\.\d+)?\s*(?:dollars?|usd)\b|\b\d+\s?%|\b\d+\s*days?\b", re.I)

# sanity: C=0.1 on the 13-example fit set reproduces guard_v10's classifier scores
clf = train(fit13r, 0.1); p0, r0 = SETS["fresh"][0][:2]
assert abs(clf.predict_proba(feat(p0, r0)[1].reshape(1, -1))[0, 1] - guard._precedent_score(p0, r0)) < 1e-6

print("T1 / T4  regularization sweep: isolated PF recall | FPR on compliant | score range (all held-out items)")
for fname, fit in (("13-example fit (guard_v10)", fit13r), ("41-example fit (recalibrated)", fit41)):
    print(f"  {fname}")
    for C in (0.01, 0.1, 1, 10, 100):
        m = train(fit, C); rec, fp, allsc = {}, {}, []
        for sn, data in SETS.items():
            sc = m.predict_proba(np.array([feat(p, r)[1] for p, r, y, c in data]))[:, 1]; allsc += list(sc)
            pf = np.array([c == "Precedent Framing" for *_, c in data]); comp = np.array([y == 0 for _, _, y, _ in data])
            rec[sn] = f"{int((sc[pf] > .5).sum())}/{pf.sum()}"; fp[sn] = f"{int((sc[comp] > .5).sum())}/{comp.sum()}"
        print(f"    C={C:<6} recall " + " ".join(f"{k}:{v}" for k, v in rec.items()) +
              f" | FPR " + " ".join(f"{k}:{v}" for k, v in fp.items()) + f" | scores {min(allsc):.2f}-{max(allsc):.2f}")

print("\nT2  style proxy (frozen guard_v10): training PF examples with an amount:",
      sum(bool(AMOUNT.search(r)) for p, r, y, c in fit13r if c == "Precedent Framing"), "of",
      sum(c == "Precedent Framing" for *_, c in fit13r))
conc, abst = [], []
for sn, data in SETS.items():
    for p, r, y, c in data:
        if c == "Precedent Framing": (conc if AMOUNT.search(r) else abst).append(guard._precedent_score(p, r))
for nm, v in (("abstract (no amount)", abst), ("concrete (amount)", conc)):
    v = np.array(v); print(f"    {nm:22} n={len(v):3d} median score {np.median(v):.3f}  detected {int((v > .5).sum())}/{len(v)}")
if abst and conc:
    print(f"    Mann-Whitney abstract > concrete: p={mannwhitneyu(abst, conc, alternative='greater').pvalue:.4f}")

print("\nT3  proximity to the 4 training PF examples vs classifier score (frozen guard_v10)")
cent = np.mean([feat(p, r)[0] for p, r, y, c in fit13r if c == "Precedent Framing"], axis=0); cent /= np.linalg.norm(cent)
prox, score, lab = [], [], []
for data in SETS.values():
    for p, r, y, c in data:
        prox.append(float(feat(p, r)[0] @ cent)); score.append(guard._precedent_score(p, r)); lab.append(c)
prox, score, lab = np.array(prox), np.array(score), np.array(lab)
rho, pv = spearmanr(prox, score); print(f"    all held-out items (n={len(score)}): Spearman rho={rho:.2f} (p={pv:.1e})")
for c in ("Precedent Framing", "Safe", "Adversarial", "Direct Failure", "Topic Drift"):
    m = lab == c; r_, p_ = spearmanr(prox[m], score[m]); print(f"    within {c:18} n={m.sum():3d} rho={r_:.2f} (p={p_:.1e})")
