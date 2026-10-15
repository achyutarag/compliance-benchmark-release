"""
POST-HOC DIAGNOSTIC (not pre-registered): MSCG with oracle-retrieved policy context (NOT the full policy).
For each named-institution, in-scope question, the policy context given to MSCG is the 2 chunks of that institution's
documents that best cover the clauses cited in the frozen answer key (chosen from the key only, independent of answers).
Compares MSCG on the same 188 answers with retrieved vs oracle policy text.
"""
import csv, json, os, re
import numpy as np
from scipy.stats import binomtest
import run_mscg_core as core   # helper below (guard loading + gate logic), shared with run_mscg.py semantics

HERE = os.path.dirname(os.path.abspath(__file__))
chunks = [json.loads(l) for l in open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8")]
key = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
prev = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(HERE, "results", "mscg_predictions.csv"), encoding="utf-8"))}
from rag_retrieval import INST_CODE, clause_coverage
ans = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    for r in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r

oracle = {}
for q, k in key.items():
    if q[0] in "NO": continue
    inst = INST_CODE[k["institution"]]
    cand = [c for c in chunks if c["institution"] == inst]
    def score(c):
        s = [clause_coverage(k[col], [c]) for col in ("clause_annotator_A", "clause_annotator_B")]
        return max([x for x in s if x is not None], default=0)
    best = sorted(cand, key=lambda c: -score(c))[:2]
    oracle[q] = (best, max(clause_coverage(k[c], best) or 0 for c in ("clause_annotator_A", "clause_annotator_B")))

guard = core.load_guard()
rows = []
for (q, m), p in sorted(prev.items()):
    if q not in oracle: continue
    pol = " ".join(c["text"] for c in oracle[q][0])
    sig = core.signals(guard, pol, ans[(q, m)]["answer"])
    rows.append({**p, "pred_retrieved": int(p["pred_mscg"]), "gate_retrieved": p["gate"],
                 "pred_oracle": core.decide(sig)[0], "gate_oracle": core.decide(sig)[1],
                 "oracle_chunks": "|".join(c["chunk_id"] for c in oracle[q][0]), "oracle_clause_coverage": round(oracle[q][1], 2),
                 "oracle_tokens": sig["n_tok"]})
with open(os.path.join(HERE, "results", "mscg_oracle_context_diagnostic.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

y = np.array([int(r["y_violation"]) for r in rows]); PR = np.array([r["pred_retrieved"] for r in rows]); PO = np.array([r["pred_oracle"] for r in rows])
cov = np.array([r["oracle_clause_coverage"] for r in rows])
print(f"rows: {len(rows)} | violations: {y.sum()} | oracle clause coverage median {np.median(cov):.2f}, >=0.6 for {np.mean(cov >= 0.6):.0%} "
      f"| oracle inputs >512 tokens: {sum(r['oracle_tokens'] > 512 for r in rows)}")
def rep(name, p, mk=None):
    mk = np.ones(len(y), bool) if mk is None else mk; yy, pp = y[mk], p[mk]
    tp, fn, fp, tn = [int(((yy == a) & (pp == b)).sum()) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0))]
    ci = lambda k, n: (lambda c: f"[{c.low:.0%}, {c.high:.0%}]")(binomtest(k, n).proportion_ci(method="exact")) if n else ""
    print(f"  {name:24} acc={(tp+tn)/len(yy):5.1%}  recall={tp}/{tp+fn} {ci(tp, tp+fn):11} FPR={fp}/{fp+tn} ({fp/(fp+tn) if fp+tn else 0:.0%})")
print("\nAll 188:"); rep("MSCG, retrieved text", PR); rep("MSCG, oracle context", PO)
b = int(((PO == y) & (PR != y)).sum()); c = int(((PO != y) & (PR == y)).sum())
print(f"  McNemar oracle vs retrieved: oracle-only-correct={b}, retrieved-only-correct={c}, p={binomtest(min(b,c), b+c).pvalue if b+c else float('nan'):.3f}")
for cl in ("YES", "PARTIAL", "NO"):
    mk = np.array([r["clause_in_retrieved"] == cl for r in rows])
    print(f"\nRetrieved clause = {cl} (n={mk.sum()}, violations={y[mk].sum()}):"); rep("retrieved", PR, mk); rep("oracle", PO, mk)
from collections import Counter
print("\nOracle-run gate on violations:", dict(Counter(r["gate_oracle"] for r in rows if int(r["y_violation"]))))
print("Oracle-run gate on compliant:  ", dict(Counter(r["gate_oracle"] for r in rows if not int(r["y_violation"]))))
print("Precedent score >0.5 in oracle run: see oracle CSV (prec column not stored separately)")
