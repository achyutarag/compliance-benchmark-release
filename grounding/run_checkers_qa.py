"""Question-plus-answer hypothesis condition (GROUNDING_PROTOCOL.md, Amendment 4). Usage: python run_checkers_qa.py <hhem|minicheck|nli_large>
Appends one JSON line per item to grounding/scores_<checker>_qa.jsonl; resumable. Same models, revisions, inputs and rules as the answer-only runs; only the hypothesis changes."""
import csv, json, os, sys, time
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); RAG = os.path.dirname(HERE)
checker = sys.argv[1]; torch.set_num_threads(os.cpu_count())
out_path = os.path.join(HERE, f"scores_{checker}_qa.jsonl")
done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(RAG, "index", "chunks.jsonl"), encoding="utf-8"))}
oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(RAG, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
hyp = lambda a: f"Question: {a['question']} Answer: {a['answer']}"
items = []
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for a in map(json.loads, open(os.path.join(RAG, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        items.append((f"{a['qid']}|{m}|retrieved", " ".join(c["text"] for c in a["retrieved"]), hyp(a)))
        if a["qid"] in oracle: items.append((f"{a['qid']}|{m}|oracle", " ".join(chunks[c] for c in oracle[a["qid"]]), hyp(a)))
todo = [it for it in items if it[0] not in done]; print(f"{checker} qa: {len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
REVS = json.load(open(os.path.join(HERE, "CHECKER_REVISIONS.json")))
t0 = time.time(); B = 8
if checker == "hhem":
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(os.environ.get("HHEM_DIR", "hhem_repo"), trust_remote_code=True).eval()
    with open(out_path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), B):
            batch = todo[b0:b0 + B]; sc = model.predict([(p, h) for _, p, h in batch]).tolist()
            for (iid, _, _), s in zip(batch, sc): f.write(json.dumps({"id": iid, "support": s}) + "\n")
            f.flush(); print(f"  {min(b0 + B, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
else:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    MID = {"minicheck": "lytang/MiniCheck-RoBERTa-Large", "nli_large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"}[checker]
    tok = AutoTokenizer.from_pretrained(MID, revision=REVS[MID])
    mod = AutoModelForSequenceClassification.from_pretrained(MID, revision=REVS[MID], use_safetensors=(checker != "minicheck"), dtype=torch.bfloat16).eval()
    lab = {v.lower(): k for k, v in mod.config.id2label.items()}
    @torch.no_grad()
    def probs(pairs):
        enc = tok([p for p, _ in pairs], [h for _, h in pairs], truncation="only_first", max_length=512, padding=True, return_tensors="pt")
        return torch.softmax(mod(**enc).logits.float(), -1).tolist()
    with open(out_path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), B):
            batch = todo[b0:b0 + B]; P_ = probs([(p, h) for _, p, h in batch])
            for (iid, _, _), p in zip(batch, P_):
                if checker == "minicheck": rec = {"id": iid, "support": p[1]}
                else:
                    e, c = lab["entailment"], lab["contradiction"]; rec = {"id": iid, "E": p[e], "C": p[c], "argmax_contra": int(max(range(3), key=lambda k: p[k]) == c)}
                f.write(json.dumps(rec) + "\n")
            f.flush(); print(f"  {min(b0 + B, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
print("done", checker, f"{time.time() - t0:.0f}s", flush=True)
