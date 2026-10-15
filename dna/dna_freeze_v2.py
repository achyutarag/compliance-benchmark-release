"""Freeze DNA v2 (Amendment 1 in DNA_PROTOCOL.md): fixed 0.5 thresholds, no fitting."""
import csv, datetime, json, os
import numpy as np
import dna_core as core
OUT = os.path.join(core.HERE, "DNA_CONFIG_FROZEN.json"); assert not os.path.exists(OUT)
cfg = {"system": "DNA v2 (Decomposed NLI Auditor)", "model_id": core.MODEL_ID, "min_claim_words": core.MIN_WORDS,
       "context_window": core.WINDOW, "tau_c": 0.5, "tau_e": 0.5,
       "rule": "violation if max_claim max_segment C > tau_c, else if min_claim max_segment E < tau_e",
       "calibration": "none (fixed 0.5; Amendment 1)"}
ROOT = os.path.dirname(os.path.dirname(core.HERE))
fit = [r for r in csv.DictReader(open(os.path.join(ROOT, "output_v5.csv"), encoding="utf-8")) if r["keep"].strip() == "1" and r["split"] == "fit"]
p = np.array([core.decide(core.scores(r["policy"], r["response"]), cfg)[0] for r in fit]); y = np.array([int(r["verified_label"]) for r in fit])
fr = {"config": cfg, "config_sha256": core.config_hash(cfg), "fit_accuracy_for_reference": float((p == y).mean()),
      "frozen_at": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(fr, open(OUT, "w"), indent=2); os.chmod(OUT, 0o444); print(json.dumps(fr, indent=2))
