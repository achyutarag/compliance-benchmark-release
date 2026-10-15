"""Recompute the synthetic rows of the paper's Table (tab:synthetic) from the raw labelled synthetic sets with the frozen guard_v10.
Per set: n, naive NLI accuracy (argmax contradiction), full MSCG accuracy, recall of the contradiction gates on Direct Failure
(gate 1 alone, and gates 1+2), FPR of the contradiction gates, recall of the precedent classifier (prec > 0.5) on Precedent Framing."""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); ROOT = os.path.join(R, "synthetic")  # release layout: data + notebook in synthetic/
sys.path.insert(0, R)
import run_mscg_core
run_mscg_core.ROOT = ROOT
from run_mscg_core import load_guard, signals, decide
g = load_guard()
def rows(f, flt=lambda r: True):
    return [r for r in csv.DictReader(open(os.path.join(ROOT, f), encoding="utf-8")) if r.get("keep", "1") == "1" and flt(r)]
sets = [("Fresh held-out (gpt-4o-mini)", rows("fresh_gift_final.csv")), ("Cross-generator (Claude)", rows("cross_gen_claude_final.csv"))]
md = rows("multi_domain_final_v3.csv")
for d in ("data_retention", "coi_disclosure", "ai_data_use"): sets.append((f"Zero-shot: {d}", [r for r in md if r["domain"] == d]))
out = []
for name, rs in sets:
    recs = []
    for r in rs:
        s = signals(g, r["policy"], r["response"]); y = int(r["verified_label"]); p, gate = decide(s)
        naive = int(max((s['C'], 0), (s['E'], 1), (s['N'], 2))[1] == 0)  # argmax = contradiction
        recs.append((r["category"], y, p, gate, naive, s))
    n = len(recs); acc = lambda i: 100 * sum(x[1] == x[i] for x in recs) / n
    df = [x for x in recs if x[0] == "Direct Failure"]; pf = [x for x in recs if x[0] == "Precedent Framing"]; neg = [x for x in recs if x[1] == 0]
    c1 = lambda x: x[5]["C"] > 0.35; c12 = lambda x: c1(x) or (x[5]["N"] > 0.70 and x[5]["C"] > 0.15)
    line = (f"{name:34} n={n:3d} naive={acc(4):5.1f} mscg={acc(2):5.1f} | contra recall g1 {sum(map(c1, df))}/{len(df)} ({100*sum(map(c1, df))/len(df):.0f}%)"
            f" g1+2 {sum(map(c12, df))}/{len(df)} ({100*sum(map(c12, df))/len(df):.0f}%) | contra FPR g1 {sum(map(c1, neg))}/{len(neg)} g1+2 {sum(map(c12, neg))}/{len(neg)}"
            f" | precedent clf recall {sum(x[5]['prec'] > 0.5 for x in pf)}/{len(pf)} ({100*sum(x[5]['prec'] > 0.5 for x in pf)/len(pf):.0f}%)")
    print(line, flush=True); out.append(line)
open(os.path.join(R, "results", "synthetic_audit.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
