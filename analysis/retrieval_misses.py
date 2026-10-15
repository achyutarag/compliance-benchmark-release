"""Why did institution-aware retrieval miss the deciding clause? (post-hoc, descriptive; rules fixed before running)
Unit: in-scope named-institution questions whose deciding clause was absent (NO) or partial in the frozen retrieval.
Clause location: chunk(s) of the named institution whose text best covers the annotators' cited clause (content-word coverage,
same measure as rag_retrieval.clause_coverage). Categories, applied in order:
  CHUNKING   no single chunk covers >= 0.6 of the clause, but two adjacent chunks together cover >= 0.8
  NEAR_MISS  best clause chunk ranked 3-5 in the institution-filtered ranking (top-2 retrieved)
  MISMATCH   best clause chunk ranked > 5
  NOT_FOUND  best single-chunk coverage < 0.6 and no adjacent pair reaches 0.8
"""
import csv, json, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from rag_retrieval import Retriever, INST_CODE, clause_coverage

R = Retriever()
key = {r["qid"]: r for r in csv.DictReader(open(os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv"), encoding="utf-8"))}
lab = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in csv.DictReader(open(os.path.join(HERE, f), encoding="utf-8")):
        lab.setdefault(r["qid"], []).append(r)
frozen = {json.loads(l)["qid"]: json.loads(l) for l in open(os.path.join(HERE, "index", "retrieval_frozen.jsonl"), encoding="utf-8")}
V2 = {r["qid"]: r["clause_in_retrieved_v2"] for r in csv.DictReader(open(os.path.join(HERE, "clause_labels_v2_FROZEN_2026-10-11.csv"), encoding="utf-8"))}
rows = []
for q, ls in sorted(lab.items()):
    cl = V2[q]   # clause labels v2 (frozen 2026-10-11); round-1 labels superseded
    if cl not in ("NO", "PARTIAL"): continue
    k = key[q]; inst = INST_CODE[k["institution"]]
    clauses = [k[c] for c in ("clause_annotator_A", "clause_annotator_B") if (k[c] or "").strip()]
    cand = [i for i, c in enumerate(R.chunks) if c["institution"] == inst]
    cov = lambda idxs: max((clause_coverage(t, [R.chunks[i] for i in idxs]) or 0) for t in clauses) if clauses else 0
    single = {i: cov([i]) for i in cand}
    best = max(single, key=single.get)
    pair = max((cov([i, i + 1]) for i in cand if i + 1 in single and R.chunks[i]["doc"] == R.chunks[i + 1]["doc"]), default=0)
    qv = R.model.encode(k["question"], convert_to_numpy=True, normalize_embeddings=True)
    sims = R.E @ qv
    order = sorted(cand, key=lambda i: -sims[i]); rank = order.index(best) + 1
    retrieved = [R.chunks.index(next(c for c in R.chunks if c["chunk_id"] == cid)) for cid in frozen[q]["chunk_ids"]]
    if single[best] < 0.6 and pair >= 0.8: cat = "CHUNKING"
    elif single[best] < 0.6: cat = "NOT_FOUND"
    elif rank <= 5: cat = "NEAR_MISS"
    else: cat = "MISMATCH"
    rows.append({"qid": q, "institution": k["institution"], "clause_label": cl, "category": cat, "best_chunk": R.chunks[best]["chunk_id"],
                 "best_cov": round(single[best], 2), "pair_cov": round(pair, 2), "rank": rank, "n_inst_chunks": len(cand),
                 "sim_best": round(float(sims[best]), 3), "sim_top2_min": round(float(min(sims[i] for i in retrieved)), 3),
                 "violations": sum(int(r["y_violation"]) for r in ls), "answers": len(ls), "question": k["question"]})
out = os.path.join(HERE, "results", "retrieval_misses_v2.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"questions: {len(rows)} ({Counter(r['clause_label'] for r in rows)})")
for cat in ("CHUNKING", "NEAR_MISS", "MISMATCH", "NOT_FOUND"):
    s = [r for r in rows if r["category"] == cat]
    if s: print(f"  {cat:10} {len(s):2d} questions | violations {sum(r['violations'] for r in s):2d}/{sum(r['answers'] for r in s)} answers | "
                f"median rank {np.median([r['rank'] for r in s]):.0f} | median sim gap {np.median([r['sim_top2_min'] - r['sim_best'] for r in s]):.3f}")
print("by institution:", Counter((r["institution"][:12], r["category"]) for r in rows))
print("rank distribution:", sorted(r["rank"] for r in rows))
