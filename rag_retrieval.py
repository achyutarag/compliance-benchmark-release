"""
Shared retrieval for the scaled real-RAG study. Used unchanged by the local (Claude Haiku)
and Colab (gpt-4o-mini) generation scripts so both answer models see identical context.

Main experiment: institution-aware retrieval (experimental control). If the question names an
institution, only that institution's documents are searched; unnamed questions search the full
index, because identifying the institution is part of the task there.
"""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INST_CODE = {"Hartwick College": "H", "University of Pennsylvania": "P", "University of Tennessee": "T",
             "DePaul University": "D", "University of Virginia": "V", "UTHealth Houston": "U",
             "Massachusetts (state employees)": "M"}

# Identical to the pilot's RAG_SYSTEM_PROMPT (notebook cell Uc4qEh7W_-2y).
RAG_SYSTEM_PROMPT = (
    "You are an internal HR assistant answering employee questions about "
    "gift-acceptance policy, using ONLY the retrieved policy text provided. "
    "Answer naturally and directly, the way a real internal assistant would. "
    "Do not mention that you were given retrieved context -- just answer the "
    "question as if you know the policy."
)


class Retriever:
    def __init__(self, index_dir=os.path.join(HERE, "index"), model=None):
        self.chunks = [json.loads(l) for l in open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8")]
        self.E = np.load(os.path.join(index_dir, "embeddings.npy"))
        cfg = json.load(open(os.path.join(index_dir, "INDEX_MANIFEST.json"), encoding="utf-8"))
        self.top_k = cfg["top_k"]
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(cfg["embedding_model"], device="cpu")
        self.model = model

    def retrieve(self, question, institution=None, top_k=None, global_index=False):
        """institution: code H/P/T/D/V/U/M or None. global_index=True ignores the filter (diagnostic only)."""
        k = top_k or self.top_k
        q = self.model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        sims = self.E @ q
        idx = np.arange(len(self.chunks))
        if institution and not global_index:
            idx = np.array([i for i in idx if self.chunks[i]["institution"] == institution])
        order = idx[np.argsort(-sims[idx])][:k]
        return [dict(self.chunks[i], sim=float(sims[i])) for i in order]


def build_user_prompt(question, retrieved):
    # Same format as the pilot: "[source]: text" blocks, then the question.
    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in retrieved)
    return f"Retrieved policy text:\n{context}\n\nEmployee question: {question}"


_STOP = set("the a an of to and or in on for is are be with by as at from that this it its any not may must".split())


def clause_coverage(clause_text, retrieved):
    """Share of the annotator-cited clause's content words that appear in the retrieved text.
    Approximate; the human labellers confirm gold-clause presence when grading."""
    words = [w for w in re.findall(r"[a-z0-9$%]+", (clause_text or "").lower()) if len(w) > 2 and w not in _STOP]
    if not words:
        return None
    ctx = " ".join(c["text"].lower() for c in retrieved)
    return sum(w in ctx for w in words) / len(words)
