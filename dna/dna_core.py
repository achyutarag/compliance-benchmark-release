"""Decomposed NLI Auditor (DNA): segmentation, cached per-claim NLI, gate logic. See DNA_PROTOCOL.md."""
import hashlib, json, os, pickle, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "cross-encoder/nli-deberta-v3-base"
MIN_WORDS = 4
WINDOW = 2
CACHE = os.path.join(HERE, "nli_cache.pkl")
_model, _cache = None, None

SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\([a-z0-9]+\))\s*")


def sentences(text):
    out = []
    for s in SENT.split(text or ""):
        s = MARK.sub("", s.strip()).strip()
        if s: out.append(s)
    return out


def claims(answer):
    return [s for s in sentences(answer) if len(s.split()) >= MIN_WORDS]


def segments(context):
    s = sentences(context)
    return s + [" ".join(s[i:i + WINDOW]) for i in range(len(s) - WINDOW + 1)] if len(s) > 1 else s


def _load():
    global _model, _cache
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_ID)
        lab = {v.lower(): k for k, v in _model.model.config.id2label.items()}
        assert (lab["contradiction"], lab["entailment"], lab["neutral"]) == (0, 1, 2), lab
        _cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


def nli(pairs):
    """pairs: list of (premise, hypothesis) -> array (n, 3) of softmax [C, E, N]; cached."""
    _load()
    todo = list({p for p in pairs if p not in _cache})
    if todo:
        probs = _model.predict(todo, apply_softmax=True, batch_size=32, show_progress_bar=False)
        for p, pr in zip(todo, probs): _cache[p] = np.asarray(pr, dtype=np.float32)
    return np.array([_cache[p] for p in pairs]) if pairs else np.zeros((0, 3))


def save_cache():
    if _cache is not None: pickle.dump(_cache, open(CACHE, "wb"))


def scores(context, answer):
    cl, sg = claims(answer), segments(context)
    if not cl:  # answer with no claim of >= MIN_WORDS words: score the whole answer as one claim
        cl = [answer.strip()]
    P = nli([(s, c) for c in cl for s in sg]).reshape(len(cl), len(sg), 3)
    support, contra = P[:, :, 1].max(1), P[:, :, 0].max(1)
    return {"max_contra": float(contra.max()), "min_support": float(support.min()),
            "n_claims": len(cl), "n_segments": len(sg),
            "weakest_claim": cl[int(support.argmin())], "most_contradicted_claim": cl[int(contra.argmax())]}


def decide(s, cfg, use_a=True, use_b=True):
    if use_a and s["max_contra"] > cfg["tau_c"]: return 1, "gateA_contradiction"
    if use_b and s["min_support"] < cfg["tau_e"]: return 1, "gateB_unsupported"
    return 0, "pass"


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
