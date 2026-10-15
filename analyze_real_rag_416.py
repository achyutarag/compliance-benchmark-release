"""Pooled real-RAG analysis: 416 answers (4 answer models x 104 questions), all frozen auditors, retrieved + oracle context.
Pre-registered (EXTENSION_PREREGISTRATION.md): per-model and pooled results; intervals by bootstrap over QUESTIONS;
MSCG vs each judge on violations (McNemar); Holm across the six pairwise accuracy comparisons; judge-family strata.
Writes results/all416_per_answer.csv and results/analysis_416.txt."""
import csv, io, os, itertools, contextlib
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
ORIG, EXT = ("claude-haiku-4-5", "gpt-4o-mini"), ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))

rows = {}
for r in rd(os.path.join(RES, "all_auditors_per_answer.csv")):
    k = (r["qid"], r["answer_model"])
    rows[k] = {"qid": k[0], "answer_model": k[1], "y": int(r["y_violation"]), "clause": r["clause_in_retrieved"],
               "mscg": int(r["mscg"]), "naive": int(r["naive"]), "mscg_oracle": r["mscg_oracle"]}
for r in rd(os.path.join(HERE, "dna", "results", "dna_real_rag_per_answer.csv")):
    k = (r["qid"], r["answer_model"]); rows[k]["dna"] = int(r["dna_ret"]); rows[k]["dna_oracle"] = r.get("dna_orc", "")
for r in rd(os.path.join(RES, "ext_local_auditors.csv")):
    k = (r["qid"], r["answer_model"])
    rows[k] = {"qid": k[0], "answer_model": k[1], "y": int(r["y_violation"]), "clause": r["clause_in_retrieved"],
               "mscg": int(r["mscg"]), "naive": int(r["naive"]), "dna": int(r["dna"]),
               "mscg_oracle": r.get("mscg_oracle", ""), "dna_oracle": r.get("dna_oracle", "")}
JN = {"openai": "gpt55", "anthropic": "sonnet55"}
for f in ("judge_predictions.csv", "judge_predictions_ext.csv"):
    for r in rd(os.path.join(HERE, "judges", f)):
        rows[(r["qid"], r["answer_model"])][f"{JN[r['provider']]}_{r['condition']}"] = int(r["pred_violation"])
R = [rows[k] for k in sorted(rows)]
assert len(R) == 416 and all(all(c in r for c in ("mscg", "naive", "dna", "gpt55_retrieved", "sonnet55_retrieved")) for r in R)
with open(os.path.join(RES, "all416_per_answer.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(c for r in R for c in r))); w.writeheader(); w.writerows(R)

out = io.StringIO()
def P(*a): print(*a); print(*a, file=out)
rng = np.random.default_rng(20261001)
QIDS = sorted({r["qid"] for r in R})

def boot(sub, stat, B=4000):
    """Cluster bootstrap over questions: resample questions with replacement, keep all their answers."""
    byq = defaultdict(list)
    for r in sub: byq[r["qid"]].append(r)
    qs = list(byq); vals = []
    for _ in range(B):
        s = [r for q in rng.choice(qs, len(qs)) for r in byq[q]]
        v = stat(s)
        if v is not None: vals.append(v)
    return np.percentile(vals, [2.5, 97.5])

def metrics(sub, col):
    y = np.array([r["y"] for r in sub]); p = np.array([r[col] for r in sub])
    tp, fp = int((p & y).sum()), int((p & (1 - y)).sum())
    rec = lambda s: (lambda yy, pp: (pp & yy).sum() / yy.sum() if yy.sum() else None)(np.array([r["y"] for r in s]), np.array([r[col] for r in s]))
    lo, hi = boot(sub, rec)
    return (p == y).mean(), tp, int(y.sum()), lo, hi, fp, int((1 - y).sum())

AUD = [("Naive NLI", "naive"), ("MSCG", "mscg"), ("DNA", "dna"), ("GPT-5.5", "gpt55_retrieved"), ("Sonnet 5.5", "sonnet55_retrieved")]
def table(title, sub, auds=AUD):
    P(f"\n=== {title}: n={len(sub)}, violations={sum(r['y'] for r in sub)} ===")
    P(f"  {'auditor':12}{'acc':>7}  {'recall':>8}  {'95% CI (q-boot)':>16}  {'FPR':>10}")
    P(f"  {'always-comp':12}{1 - np.mean([r['y'] for r in sub]):>7.1%}  {'0':>8}")
    for name, col in auds:
        acc, tp, n1, lo, hi, fp, n0 = metrics(sub, col)
        P(f"  {name:12}{acc:>7.1%}  {f'{tp}/{n1}':>8}  {f'[{lo:.0%}, {hi:.0%}]':>16}  {f'{fp}/{n0} {fp / n0:.0%}':>10}")

table("POOLED, retrieved context (4 answer models)", R)
for m in ORIG + EXT: table(f"answer model {m}", [r for r in R if r["answer_model"] == m])
table("ORIGINAL two models (as in paper)", [r for r in R if r["answer_model"] in ORIG])
table("EXTENSION two models (independent family stratum)", [r for r in R if r["answer_model"] in EXT])

def mcnemar(sub, a, b):
    y = np.array([r["y"] for r in sub]); A = np.array([r[a] for r in sub]); B = np.array([r[b] for r in sub])
    x, z = int(((A == y) & (B != y)).sum()), int(((A != y) & (B == y)).sum())
    return x, z, (binomtest(min(x, z), x + z).pvalue if x + z else 1.0)

P("\n=== Pre-registered primary comparison: recall on the 79 violations (McNemar on violations only) ===")
V = [r for r in R if r["y"] == 1]
for name, col in AUD[3:] + [("Naive NLI", "naive")]:
    x, z, p = mcnemar(V, col, "mscg")
    P(f"  {name} vs MSCG: {name}-only detects {x}, MSCG-only detects {z}, exact p={p:.4f}")
P("\n=== Accuracy, six pairwise comparisons, Holm-corrected (pooled 416) ===")
pairs = list(itertools.combinations([a for a in AUD if a[1] != "dna"], 2)); res = []
for (n1, c1), (n2, c2) in pairs:
    x, z, p = mcnemar(R, c1, c2); res.append((n1, n2, x, z, p))
order = sorted(range(len(res)), key=lambda i: res[i][4]); adj = [0] * len(res); run = 0
for rnk, i in enumerate(order): run = max(run, min(1, (len(res) - rnk) * res[i][4])); adj[i] = run
for (n1, n2, x, z, p), a in zip(res, adj):
    P(f"  {n1:10} vs {n2:10}: {n1}-only correct {x:3d}, {n2}-only correct {z:3d}, p={p:.4f}, Holm p={a:.4f}")

P("\n=== Where violations arise: deciding clause in retrieved text x answer-model group ===")
for cl in ("YES", "PARTIAL", "NO", "N/A"):
    for grp, ms in (("orig", ORIG), ("ext", EXT)):
        s = [r for r in R if r["clause"] == cl and r["answer_model"] in ms]
        P(f"  clause={cl:7} {grp:4}: {sum(r['y'] for r in s):2d}/{len(s)} violations")

P("\n=== Recall by clause presence (pooled) ===")
for cl in ("YES", "PARTIAL", "NO", "N/A"):
    s = [r for r in R if r["clause"] == cl and r["y"] == 1]
    P(f"  clause={cl:7} violations={len(s):2d}: " + ", ".join(f"{n} {sum(r[c] for r in s)}" for n, c in AUD))

P("\n=== Judge-family check ===")
for jn, jc, fam in (("GPT-5.5", "gpt55_retrieved", "gpt-4o-mini"), ("Sonnet 5.5", "sonnet55_retrieved", "claude-haiku-4-5")):
    for m in ORIG + EXT:
        s = [r for r in R if r["answer_model"] == m]; y = [r["y"] for r in s]
        tp = sum(r[jc] & r["y"] for r in s); fp = sum(r[jc] & (1 - r["y"]) for r in s)
        tag = "same family" if m == fam else ("independent family" if m in EXT else "other family")
        P(f"  {jn:10} on {m:22} ({tag:18}): recall {tp}/{sum(y)}, FPR {fp}/{len(y) - sum(y)}")

P("\n=== Oracle policy context (in-scope named-institution questions, all 4 models) ===")
O = [r for r in R if r.get("mscg_oracle", "") != "" and r.get("gpt55_oracle") is not None]
for r in O:
    r["mscg_oracle"] = int(r["mscg_oracle"]); r["dna_oracle"] = int(r["dna_oracle"]) if r.get("dna_oracle", "") != "" else None
table("Oracle context, pooled", O, [("MSCG", "mscg_oracle"), ("GPT-5.5", "gpt55_oracle"), ("Sonnet 5.5", "sonnet55_oracle")])
table("Retrieved context, same answers", O, [("MSCG", "mscg"), ("GPT-5.5", "gpt55_retrieved"), ("Sonnet 5.5", "sonnet55_retrieved")])
x, z, p = mcnemar([r for r in O if r["y"]], "mscg_oracle", "mscg"); P(f"  MSCG oracle vs retrieved on violations: {x} vs {z}, p={p:.3f}")
for jn in ("gpt55", "sonnet55"):
    x, z, p = mcnemar([r for r in O if r["y"]], f"{jn}_oracle", "mscg_oracle"); P(f"  {jn} oracle vs MSCG oracle on violations: {x} vs {z}, p={p:.2e}")

P("\n=== Judge-judge agreement (pooled, retrieved) ===")
a = np.array([r["gpt55_retrieved"] for r in R]); b = np.array([r["sonnet55_retrieved"] for r in R]); po = (a == b).mean()
pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); P(f"  agreement {po:.1%}, kappa {(po - pe) / (1 - pe):.2f}")
open(os.path.join(RES, "analysis_416.txt"), "w", encoding="utf-8").write(out.getvalue())
