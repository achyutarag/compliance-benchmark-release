"""Does the oracle condition's higher judge false-positive rate come from swapping out the chunks a compliant answer relied on?
Splits the compliant answers of the oracle subset by whether their retrieved chunks equal the two oracle chunks, and compares the judges' flag rate in the retrieved and oracle conditions."""
import csv, json, os
import numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda p: list(csv.DictReader(open(os.path.join(H, p), encoding="utf-8")))
Y = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in rd(f): Y[(r["qid"], r["answer_model"])] = int(r["compliance_label"] == "NON_COMPLIANT")
A = {(r["qid"], r["answer_model"]): r for r in rd("results/all416_per_answer.csv")}
orc = {r["qid"]: set(r["oracle_chunks"].split("|")) for r in rd("results/mscg_oracle_context_diagnostic.csv")}
ret = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for r in map(json.loads, open(os.path.join(H, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        ret[(r["qid"], m)] = {c["chunk_id"] for c in r["retrieved"]}
out = []
def P(s=""): print(s); out.append(s)
ks = [k for k in sorted(Y) if k[0] in orc and not Y[k]]          # compliant answers in the oracle subset
same = [k for k in ks if ret[k] == orc[k[0]]]; part = [k for k in ks if ret[k] != orc[k[0]] and ret[k] & orc[k[0]]]; none = [k for k in ks if not (ret[k] & orc[k[0]])]
P(f"compliant answers in the oracle subset: {len(ks)}; retrieved chunks equal to the oracle chunks: {len(same)}; overlap in one chunk: {len(part)}; no overlap: {len(none)}")
P(f"{'group':34}{'n':>5}   " + "   ".join(f"{j:>22}" for j in ("GPT-5.5", "Sonnet 5.5")))
for nm, grp in (("retrieved = oracle chunks", same), ("one chunk shared", part), ("no chunk shared", none), ("all compliant", ks)):
    if not grp: continue
    cells = []
    for j in ("gpt55", "sonnet55"):
        r_ = np.mean([int(A[k][f"{j}_retrieved"]) for k in grp]); o_ = np.mean([int(A[k][f"{j}_oracle"]) for k in grp]); cells.append(f"retrieved {r_:4.0%} -> oracle {o_:4.0%}")
    P(f"{nm:34}{len(grp):>5}   " + "   ".join(f"{c:>22}" for c in cells))
open(os.path.join(H, "results", "oracle_swap_test.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
