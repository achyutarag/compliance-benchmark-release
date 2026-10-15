"""Frozen local auditors (MSCG guard_v10, naive NLI, DNA v2) on the 208 extension answers (Llama-3.2-3B-Instruct, gemma-2-2b-it),
retrieved context and oracle policy context. Same pairing and code paths as run_mscg.py / run_mscg_oracle_context.py / dna_evaluate.py.
Writes results/ext_local_auditors.csv."""
import csv, json, os, sys
import numpy as np
import run_mscg_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dna")); import dna_core as dna
FROZEN = json.load(open(os.path.join(HERE, "dna", "DNA_CONFIG_FROZEN.json"))); CFG = FROZEN["config"]
assert dna.config_hash(CFG) == FROZEN["config_sha256"]
MODELS = ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")
labels = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(HERE, "labels_ext_FROZEN_2026-10-01.csv"), encoding="utf-8"))}
chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8"))}
oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(HERE, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
ans = {}
for m in MODELS:
    for r in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ans[(r["qid"], m)] = r

guard = core.load_guard()
rows = []
for k in sorted(labels):
    a = ans[k]; lab = labels[k]; ctx = " ".join(c["text"] for c in a["retrieved"])
    s = core.signals(guard, ctx, a["answer"]); p, gate = core.decide(s)
    d = dna.scores(ctx, a["answer"]); pd = dna.decide(d, CFG)
    row = {"qid": k[0], "answer_model": k[1], "y_violation": int(lab["y_violation"]), "clause_in_retrieved": lab["clause_in_retrieved"],
           "C": s["C"], "E": s["E"], "N": s["N"], "sim": s["sim"], "prec": s["prec"], "nli_tokens": s["n_tok"],
           "mscg": p, "mscg_gate": gate, "naive": int(np.argmax([s["C"], s["E"], s["N"]]) == 0),
           "dna": pd[0], "dna_gate": pd[1], "dna_max_contra": d["max_contra"], "dna_min_support": d["min_support"]}
    if k[0] in oracle:
        octx = " ".join(chunks[c] for c in oracle[k[0]])
        so = core.signals(guard, octx, a["answer"]); do = dna.scores(octx, a["answer"])
        row.update({"mscg_oracle": core.decide(so)[0], "mscg_oracle_gate": core.decide(so)[1], "C_oracle": so["C"], "E_oracle": so["E"],
                    "dna_oracle": dna.decide(do, CFG)[0], "dna_oracle_max_contra": do["max_contra"], "dna_oracle_min_support": do["min_support"]})
    rows.append(row)
dna.save_cache()

# sanity: reconstructed MSCG gates equal guard.run_inference on the same pairs (as in run_mscg.py)
ref = guard.run_inference([(" ".join(c["text"] for c in ans[(r["qid"], r["answer_model"])]["retrieved"]),
                            ans[(r["qid"], r["answer_model"])]["answer"], r["y_violation"], "x", r["qid"], "x") for r in rows])
assert list(ref["pred_guard"]) == [r["mscg"] for r in rows], "gate reconstruction mismatch"
fields = list(dict.fromkeys(f for r in rows for f in r))
with open(os.path.join(HERE, "results", "ext_local_auditors.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

y = np.array([r["y_violation"] for r in rows])
for name in ("mscg", "naive", "dna"):
    p = np.array([r[name] for r in rows]); tp = int((p & y).sum()); fp = int((p & (1 - y)).sum())
    print(f"{name:6} acc {(p == y).mean():.1%}  recall {tp}/{y.sum()}  FPR {fp}/{(1 - y).sum()}")
O = [r for r in rows if "mscg_oracle" in r]; yo = np.array([r["y_violation"] for r in O])
for name in ("mscg_oracle", "dna_oracle"):
    p = np.array([r[name] for r in O]); print(f"{name:12} recall {int((p & yo).sum())}/{yo.sum()}  FPR {int((p & (1 - yo)).sum())}/{(1 - yo).sum()}")
print("precedent classifier fired:", sum(r["prec"] > 0.5 for r in rows), "| prec range", round(min(r["prec"] for r in rows), 3), round(max(r["prec"] for r in rows), 3))
