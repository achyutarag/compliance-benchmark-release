"""Resumable version of run_checkers.py (same models, inputs and scores; GROUNDING_PROTOCOL.md).
Usage: python run_checkers_resumable.py <minicheck|nli_large> <whole|sent>
Appends one JSON line per item to grounding/scores_<checker>_<granularity>.jsonl; skips items already scored."""
import csv, json, os, sys, time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); RAG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RAG, "dna")); from dna_core import sentences, MIN_WORDS
os.environ["HF_HUB_OFFLINE"] = "1"
REVS = json.load(open(os.path.join(HERE, "CHECKER_REVISIONS.json")))
MID = {"minicheck": "lytang/MiniCheck-RoBERTa-Large", "nli_large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "roberta_mnli": "FacebookAI/roberta-large-mnli"}
checker, gran = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count())
out_path = os.path.join(HERE, f"scores_{checker}_{gran}.jsonl")
done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()

chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(RAG, "index", "chunks.jsonl"), encoding="utf-8"))}
oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(RAG, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
items = []
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for a in map(json.loads, open(os.path.join(RAG, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        items.append((f"{a['qid']}|{m}|retrieved", " ".join(c["text"] for c in a["retrieved"]), a["answer"]))
        if a["qid"] in oracle:
            items.append((f"{a['qid']}|{m}|oracle", " ".join(chunks[c] for c in oracle[a["qid"]]), a["answer"]))
todo = [it for it in items if it[0] not in done]
print(f"{checker} {gran}: {len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
tok = AutoTokenizer.from_pretrained(MID[checker], revision=REVS[MID[checker]])
mod = AutoModelForSequenceClassification.from_pretrained(MID[checker], revision=REVS[MID[checker]], use_safetensors=(checker != "minicheck"), dtype=torch.bfloat16).eval()   # bf16: memory (see protocol)
lab = {v.lower(): k for k, v in mod.config.id2label.items()}


@torch.no_grad()
def probs(pairs):
    enc = tok([p for p, _ in pairs], [h for _, h in pairs], truncation="only_first", max_length=512, padding=True, return_tensors="pt")
    return torch.softmax(mod(**enc).logits.float(), -1).tolist()


t0 = time.time()
B = 8   # items per batch
with open(out_path, "a", encoding="utf-8") as f:
    for b0 in range(0, len(todo), B):
        batch = todo[b0:b0 + B]; pairs, owner = [], []
        for j, (iid, pol, ans) in enumerate(batch):
            hyps = [ans] if gran == "whole" else ([s for s in sentences(ans) if len(s.split()) >= MIN_WORDS] or [ans])
            for h in hyps: pairs.append((pol, h)); owner.append(j)
        Pall = []
        for k in range(0, len(pairs), 16): Pall += probs(pairs[k:k + 16])
        for j, (iid, _, _) in enumerate(batch):
            P = [Pall[i] for i in range(len(owner)) if owner[i] == j]
            if checker == "minicheck":   # index 1 = supported (MiniCheck reference implementation)
                rec = {"id": iid, "support": min(p[1] for p in P)}
            else:
                e, c = lab["entailment"], lab["contradiction"]
                rec = {"id": iid, "E": min(p[e] for p in P), "C": max(p[c] for p in P),
                       "argmax_contra": int(any(max(range(3), key=lambda k: p[k]) == c for p in P))}
            f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"  {min(b0 + B, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
print("done", checker, gran, f"{time.time() - t0:.0f}s", flush=True)
