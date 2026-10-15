"""Run the two established grounding checkers (GROUNDING_PROTOCOL.md) on all real-RAG answers, retrieved + oracle context.
Writes grounding/checker_scores.csv (raw scores; decisions applied in analysis)."""
import csv, json, os, sys, time, winreg
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); RAG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RAG, "dna")); from dna_core import sentences, MIN_WORDS
os.environ.setdefault("HF_TOKEN", winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "HF_TOKEN")[0])
os.environ["HF_HUB_OFFLINE"] = "1"
REVS = json.load(open(os.path.join(HERE, "CHECKER_REVISIONS.json")))
torch.set_num_threads(os.cpu_count())

chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(RAG, "index", "chunks.jsonl"), encoding="utf-8"))}
oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(RAG, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
items = []
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    for a in map(json.loads, open(os.path.join(RAG, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
        items.append((a["qid"], m, "retrieved", " ".join(c["text"] for c in a["retrieved"]), a["answer"]))
        if a["qid"] in oracle:
            items.append((a["qid"], m, "oracle", " ".join(chunks[c] for c in oracle[a["qid"]]), a["answer"]))
print(len(items), "items", flush=True)


def load(mid):
    tok = AutoTokenizer.from_pretrained(mid, revision=REVS[mid])
    mod = AutoModelForSequenceClassification.from_pretrained(mid, revision=REVS[mid], use_safetensors=("Moritz" in mid)).eval()
    print(mid, "labels:", mod.config.id2label, flush=True)
    return tok, mod


@torch.no_grad()
def probs(tok, mod, pairs, bs=8):
    out = []
    for i in range(0, len(pairs), bs):
        b = pairs[i:i + bs]
        enc = tok([p for p, _ in b], [h for _, h in b], truncation="only_first", max_length=512, padding=True, return_tensors="pt")
        out += torch.softmax(mod(**enc).logits, -1).tolist()
    return out


rows = {(q, m, c): {"qid": q, "answer_model": m, "condition": c} for q, m, c, _, _ in items}
for mid, short in (("lytang/MiniCheck-RoBERTa-Large", "minicheck"), ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "nli_large")):
    tok, mod = load(mid); lab = {v.lower(): k for k, v in mod.config.id2label.items()}
    t0 = time.time()
    whole = probs(tok, mod, [(p, a) for _, _, _, p, a in items])
    sent_pairs, owner = [], []
    for j, (_, _, _, p, a) in enumerate(items):
        ss = [s for s in sentences(a) if len(s.split()) >= MIN_WORDS] or [a]
        for s in ss: sent_pairs.append((p, s)); owner.append(j)
    sent = probs(tok, mod, sent_pairs)
    for j, (q, m, c, _, _) in enumerate(items):
        r = rows[(q, m, c)]; ps = [sent[i] for i in range(len(owner)) if owner[i] == j]
        if short == "minicheck":   # MiniCheck: index 1 = supported (as in the MiniCheck reference implementation)
            r["minicheck_support"] = whole[j][1]; r["minicheck_min_sent_support"] = min(x[1] for x in ps)
        else:
            e, cidx = lab["entailment"], lab["contradiction"]
            r["nli_large_E"] = whole[j][e]; r["nli_large_C"] = whole[j][cidx]; r["nli_large_argmax_contra"] = int(max(range(3), key=lambda k: whole[j][k]) == cidx)
            r["nli_large_min_sent_E"] = min(x[e] for x in ps); r["nli_large_max_sent_C"] = max(x[cidx] for x in ps)
    print(short, f"done in {time.time() - t0:.0f}s", flush=True)
    del mod
with open(os.path.join(HERE, "checker_scores.csv"), "w", newline="", encoding="utf-8") as f:
    rr = list(rows.values()); w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
print("written", len(rows))
