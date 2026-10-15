"""HHEM-2.1-open scores (GROUNDING_PROTOCOL.md, Amendment 3). Usage: python run_hhem.py   (HHEM_DIR = local copy of the pinned model repo; default below)
Appends {"id","support"} (P(consistent), whole answer) to grounding/scores_hhem_whole.jsonl; resumable."""
import csv, json, os, sys, time
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForSequenceClassification
HERE = os.path.dirname(os.path.abspath(__file__)); RAG = os.path.dirname(HERE)
HHEM_DIR = os.environ.get("HHEM_DIR", "hhem_repo")
torch.set_num_threads(os.cpu_count())
out_path = os.path.join(HERE, "scores_hhem_whole.jsonl")
done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(RAG, "index", "chunks.jsonl"), encoding="utf-8"))}
oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(RAG, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
items = []
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for a in map(json.loads, open(os.path.join(RAG, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        items.append((f"{a['qid']}|{m}|retrieved", " ".join(c["text"] for c in a["retrieved"]), a["answer"]))
        if a["qid"] in oracle: items.append((f"{a['qid']}|{m}|oracle", " ".join(chunks[c] for c in oracle[a["qid"]]), a["answer"]))
todo = [it for it in items if it[0] not in done]; print(f"hhem: {len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(HHEM_DIR, trust_remote_code=True).eval()
t0 = time.time()
with open(out_path, "a", encoding="utf-8") as f:
    for b0 in range(0, len(todo), 8):
        batch = todo[b0:b0 + 8]; sc = model.predict([(pol, ans) for _, pol, ans in batch]).tolist()
        for (iid, _, _), s in zip(batch, sc): f.write(json.dumps({"id": iid, "support": s}) + "\n")
        f.flush(); print(f"  {min(b0 + 8, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
print("done", f"{time.time() - t0:.0f}s", flush=True)
