"""Pre-agreed analysis: LLM judges vs MSCG vs naive NLI vs always-compliant, on the frozen human labels."""
import csv, itertools, json, os
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
lab = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(HERE, "labels_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
ms = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(OUT, "mscg_predictions.csv"), encoding="utf-8"))}
mo = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(OUT, "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
jd = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(HERE, "judges", "judge_predictions.csv"), encoding="utf-8")):
    jd[(r["provider"], r["condition"])][(r["qid"], r["answer_model"])] = int(r["pred_violation"])

def ci(k, n):
    if not n: return "n/a"
    c = binomtest(k, n).proportion_ci(method="exact"); return f"[{c.low:.0%},{c.high:.0%}]"

def table(title, keys, methods):
    y = np.array([int(lab[k]["y_violation"]) for k in keys])
    print(f"\n=== {title} (n={len(keys)}, violations={y.sum()}) ===")
    print(f"  {'method':30}{'acc':>7} {'95% CI':>11} {'recall':>9} {'95% CI':>11} {'FPR':>9} {'prec':>7}")
    preds = {}
    for name, f in methods:
        p = np.array([f(k) for k in keys]); preds[name] = p
        tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
        fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
        print(f"  {name:30}{(tp+tn)/len(y):>7.1%} {ci(tp+tn, len(y)):>11} {f'{tp}/{tp+fn}':>9} {ci(tp, tp+fn):>11} {f'{fp}/{fp+tn}':>9} "
              f"{(f'{tp}/{tp+fp}' if tp+fp else '-'):>7}")
    return y, preds

def mcnemar(y, a, b):
    x, z = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    return x, z, (binomtest(min(x, z), x + z).pvalue if x + z else 1.0)

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj

K = sorted(lab)
GPT, SON = ("openai", "retrieved"), ("anthropic", "retrieved")
methods = [("always-compliant", lambda k: 0), ("naive NLI argmax", lambda k: int(ms[k]["pred_naive"])),
           ("MSCG (frozen)", lambda k: int(ms[k]["pred_mscg"])),
           ("judge gpt-5.5", lambda k: jd[GPT][k]), ("judge claude-sonnet-5-5", lambda k: jd[SON][k])]
y, P = table("PRIMARY: retrieved context, all 208 answers", K, methods)

comps = [("judge gpt-5.5", "MSCG (frozen)"), ("judge claude-sonnet-5-5", "MSCG (frozen)"),
         ("judge gpt-5.5", "naive NLI argmax"), ("judge claude-sonnet-5-5", "naive NLI argmax"),
         ("MSCG (frozen)", "naive NLI argmax"), ("judge gpt-5.5", "judge claude-sonnet-5-5")]
res = [(a, b, *mcnemar(y, P[a], P[b])) for a, b in comps]
adj = holm([r[4] for r in res])
print("\n  McNemar (accuracy, same items), Holm-adjusted across the 6 comparisons:")
for (a, b, x, z, p), pa in zip(res, adj):
    print(f"    {a:25} vs {b:25} {x:3d} vs {z:3d}  p={p:.4f}  p_holm={pa:.4f}")
gj, sj = P["judge gpt-5.5"], P["judge claude-sonnet-5-5"]
po = (gj == sj).mean(); pe = gj.mean() * sj.mean() + (1 - gj.mean()) * (1 - sj.mean())
print(f"\n  judge-judge agreement: {po:.1%}, kappa {(po - pe) / (1 - pe):.2f}")

print("\n=== CROSSED FAMILY CHECK (retrieved context): recall and FPR by answering model ===")
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    ks = [k for k in K if k[1] == m]; table(f"answers from {m}", ks, methods[2:])

OK = sorted(mo)
methods_o = [("MSCG retrieved", lambda k: int(mo[k]["pred_retrieved"])), ("MSCG oracle", lambda k: int(mo[k]["pred_oracle"])),
             ("gpt-5.5 retrieved", lambda k: jd[GPT][k]), ("gpt-5.5 oracle", lambda k: jd[("openai", "oracle")][k]),
             ("sonnet-5-5 retrieved", lambda k: jd[SON][k]), ("sonnet-5-5 oracle", lambda k: jd[("anthropic", "oracle")][k])]
yo, PO = table("RETRIEVED vs ORACLE POLICY CONTEXT (188 named, in-scope answers)", OK, methods_o)
print("  McNemar retrieved vs oracle (same auditor):")
for a in ("MSCG", "gpt-5.5", "sonnet-5-5"):
    x, z, p = mcnemar(yo, PO[f"{a} oracle"], PO[f"{a} retrieved"]); print(f"    {a:12} oracle-only-correct {x}, retrieved-only-correct {z}, p={p:.3f}")
for cl in ("YES", "PARTIAL", "NO"):
    ks = [k for k in OK if lab[k]["clause_in_retrieved"] == cl]
    table(f"clause in retrieved = {cl}", ks, methods_o)

print("\n=== By reasoning structure (retrieved context): violations detected / violations ; false positives / compliant ===")
S = sorted({s for k in K for s in lab[k]["structures"].split(";") if s}, key=lambda s: int(s[1:]))
print(f"  {'':5}{'n':>4} {'viol':>5} | " + " | ".join(f"{n[:12]:>16}" for n, _ in methods[2:]))
for s in S:
    ks = [k for k in K if s in lab[k]["structures"].split(";")]; yy = np.array([int(lab[k]["y_violation"]) for k in ks])
    cells = []
    for name, f in methods[2:]:
        p = np.array([f(k) for k in ks]); cells.append(f"{int(((yy==1)&(p==1)).sum())}/{yy.sum()} ; {int(((yy==0)&(p==1)).sum())}/{(yy==0).sum()}")
    print(f"  {s:5}{len(ks):>4} {yy.sum():>5} | " + " | ".join(f"{c:>16}" for c in cells))
rows = [{"qid": k[0], "answer_model": k[1], "y_violation": lab[k]["y_violation"], "clause_in_retrieved": lab[k]["clause_in_retrieved"],
         "mscg": ms[k]["pred_mscg"], "naive": ms[k]["pred_naive"], "gpt55_retrieved": jd[GPT][k], "sonnet55_retrieved": jd[SON][k],
         "gpt55_oracle": jd[("openai", "oracle")].get(k, ""), "sonnet55_oracle": jd[("anthropic", "oracle")].get(k, ""),
         "mscg_oracle": mo[k]["pred_oracle"] if k in mo else ""} for k in K]
with open(os.path.join(OUT, "all_auditors_per_answer.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
