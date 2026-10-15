"""Calibrate DNA thresholds on the synthetic fit split ONLY (output_v5.csv, split=fit) and freeze the config."""
import csv, json, os, datetime
import numpy as np
import dna_core as core

ROOT = os.path.dirname(os.path.dirname(core.HERE))
OUT = os.path.join(core.HERE, "DNA_CONFIG_FROZEN.json")
assert not os.path.exists(OUT), "config already frozen"

fit = [r for r in csv.DictReader(open(os.path.join(ROOT, "output_v5.csv"), encoding="utf-8"))
       if r["keep"].strip() == "1" and r["split"] == "fit"]
S = [core.scores(r["policy"], r["response"]) for r in fit]; core.save_cache()
y = np.array([int(r["verified_label"]) for r in fit])
grid = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
best = None
for tc in grid:
    for te in grid:
        p = np.array([core.decide(s, {"tau_c": tc, "tau_e": te})[0] for s in S])
        key = ((p == y).mean(), -p.sum(), tc, -te)
        if best is None or key > best[0]: best = (key, tc, te)
(acc, negflags, _, _), tc, te = best
cfg = {"system": "DNA (Decomposed NLI Auditor)", "model_id": core.MODEL_ID, "min_claim_words": core.MIN_WORDS,
       "context_window": core.WINDOW, "tau_c": tc, "tau_e": te,
       "rule": "violation if max_claim max_segment C > tau_c, else if min_claim max_segment E < tau_e",
       "calibration": f"output_v5.csv split=fit, n={len(fit)}, grid 0.05..0.95 step 0.05, max accuracy, ties -> fewest flags"}
frozen = {"config": cfg, "fit_accuracy": float(acc), "fit_flags": int(-negflags), "fit_n": len(fit),
          "config_sha256": core.config_hash(cfg), "frozen_at": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(frozen, open(OUT, "w"), indent=2)
os.chmod(OUT, 0o444)
print(json.dumps(frozen, indent=2))
print("fit-split categories x prediction:")
for c in sorted({r["category"] for r in fit}):
    idx = [i for i, r in enumerate(fit) if r["category"] == c]
    print(f"  {c:18} flagged {sum(core.decide(S[i], cfg)[0] for i in idx)}/{len(idx)}")
