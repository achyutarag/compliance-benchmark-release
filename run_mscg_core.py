"""Frozen guard_v10 loading and gate logic, shared by the MSCG diagnostics (identical to run_mscg.py)."""
import contextlib, io, json, os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
last_prec = {}


def load_guard():
    nb = json.load(open(os.path.join(ROOT, "solutions_final_v3.ipynb"), encoding="utf-8"))
    cells = {c["metadata"].get("id"): "".join(c["source"]) for c in nb["cells"]}
    g = {"__name__": "__nb__"}; cwd = os.getcwd(); os.chdir(ROOT)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells["LshaAK9fJXIu"], g); exec(cells["gH3xCbjrXkpb"], g)
        s32 = cells["yOGAQ_CkXkpb"]; exec(s32[:s32.index("df_v10 = guard_v10.run_inference")], g)
    os.chdir(cwd)
    return g["guard_v10"]


def signals(guard, policy, response):
    C, E, N = (float(x) for x in guard.nli_model.predict([policy, response], apply_softmax=True))
    return {"C": C, "E": E, "N": N, "sim": float(guard._embed(policy) @ guard._embed(response)),
            "prec": guard._precedent_score(policy, response),
            "n_tok": len(guard.nli_model.tokenizer(policy, response)["input_ids"])}


def decide(s):
    if s["C"] > 0.35: return 1, "gate1_contradiction"
    if s["N"] > 0.70:
        if s["C"] > 0.15: return 1, "gate2_background_leak"
        if s["prec"] > 0.50: return 1, "gate3_precedent"
        if s["sim"] < 0.42: return 1, "gate4_similarity"
        return 0, "safe_high_neutral"
    return (0 if s["E"] - s["C"] >= 0.50 else 1), "gate5_margin"
