"""
Build the retrieval index for the scaled real-RAG study (Option A: pilot-equivalent settings).

Settings (fixed before any answer was generated):
  embedding model   all-MiniLM-L6-v2, L2-normalised (same as the pilot and as MSCG)
  chunking          sentence-packed chunks of <= 80 words, no overlap (pilot chunks were ~50-80 words)
  retrieval         cosine similarity, top_k = 2, single index over all institutions (no metadata filter)
  chunk label       "[<institution source>]" prefix is shown to the generator, as in the pilot

Outputs in index/: chunks.jsonl, embeddings.npy, INDEX_MANIFEST.json
"""
import hashlib, json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
P, OUT = os.path.join(HERE, "policies"), os.path.join(HERE, "index")
os.makedirs(OUT, exist_ok=True)
SEP = "=" * 70 + "\n"
MAX_WORDS, TOP_K, MODEL = 80, 2, "all-MiniLM-L6-v2"

SOURCES = {  # doc name -> (institution code, source label shown to the generator)
    "hartwick_6-42": ("H", "Hartwick College, Gifts, Loans & Favors Policy (#6.42)"),
    "upenn_VI-D": ("P", "University of Pennsylvania, Faculty Handbook VI.D Gift Policy"),
    "utk_FI0717": ("T", "University of Tennessee, FI0717 Employee Gift Acceptance Policy"),
    "utk_FI0717_guidance": ("T", "University of Tennessee, FI0717 Guidance: Acceptable and Unacceptable Gifts"),
    "depaul_gifts_to_employees": ("D", "DePaul University, Gifts to Employees"),
    "uva_FIN-019": ("V", "University of Virginia, FIN-019 Acceptance of Gifts and Special Benefits from Suppliers"),
    "uthealth_HOOP-226": ("U", "UTHealth Houston, HOOP 226 Gifts to University Employees from Third Parties"),
    "ma_930CMR5": ("M", "Massachusetts State Ethics Commission, 930 CMR 5.00 (gift regulations)"),
    "ma_advisory_19-1": ("M", "Massachusetts State Ethics Commission, Advisory 19-1: Gifts and Gratuities"),
}

man = json.load(open(os.path.join(P, "MANIFEST.json"), encoding="utf-8"))
assert man.get("frozen"), "corpus is not frozen"
for d in man["documents"]:  # refuse to build from a modified corpus
    assert hashlib.sha256(open(os.path.join(P, d["text"]), "rb").read()).hexdigest() == d["text_sha256"], d["name"]

SENT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(“\"•])|\s+(?=•\s)")
chunks = []
for d in man["documents"]:
    body = open(os.path.join(P, d["text"]), encoding="utf-8").read().split(SEP, 1)[1]
    text = " ".join(body.split())                      # undo PDF hard line breaks
    sents = [s.strip() for s in SENT.split(text) if s.strip()]
    cur, n = [], 0
    def flush():
        if cur:
            inst, label = SOURCES[d["name"]]
            chunks.append({"chunk_id": f"{d['name']}_{len([c for c in chunks if c['doc'] == d['name']]) + 1:03d}",
                           "doc": d["name"], "institution": inst, "source": label, "text": " ".join(cur)})
    for s in sents:
        w = len(s.split())
        if cur and n + w > MAX_WORDS:
            flush(); cur, n = [], 0
        cur.append(s); n += w
    flush()

model = SentenceTransformer(MODEL, device="cpu")
emb = model.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
np.save(os.path.join(OUT, "embeddings.npy"), emb)
with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
lens = [len(c["text"].split()) for c in chunks]
per_doc = {d: sum(1 for c in chunks if c["doc"] == d) for d in SOURCES}
json.dump({"option": "A (pilot-equivalent)", "embedding_model": MODEL, "normalized": True, "max_words_per_chunk": MAX_WORDS,
           "overlap": 0, "top_k": TOP_K, "metadata_filter": None, "corpus_sha256": man["corpus_sha256"],
           "n_chunks": len(chunks), "chunks_per_doc": per_doc,
           "chunk_words": {"min": min(lens), "median": int(np.median(lens)), "max": max(lens)},
           "embeddings_sha256": hashlib.sha256(open(os.path.join(OUT, "embeddings.npy"), "rb").read()).hexdigest()},
          open(os.path.join(OUT, "INDEX_MANIFEST.json"), "w", encoding="utf-8"), indent=2)
print(f"{len(chunks)} chunks | words min/median/max {min(lens)}/{int(np.median(lens))}/{max(lens)}")
print(per_doc)
