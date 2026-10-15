"""Evaluate frozen DNA v2 with the component protocol on all synthetic sets and on real RAG (retrieved + oracle context).
Writes results to dna/results/. See DNA_PROTOCOL.md (incl. Amendment 1)."""
import csv, json, os
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
import dna_core as core

HERE = core.HERE; RAG = os.path.dirname(HERE); ROOT = os.path.dirname(RAG); OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
FROZEN = json.load(open(os.path.join(HERE, "DNA_CONFIG_FROZEN.json")))
CFG = FROZEN["config"]; assert core.config_hash(CFG) == FROZEN["config_sha256"]
print(f"DNA config sha256 {FROZEN['config_sha256'][:12]}  tau_c={CFG['tau_c']} tau_e={CFG['tau_e']}")


def load(path, splits=None, domain=None):
    return [(r["policy"], r["response"], int(r["verified_label"]), r["category"])
            for r in csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8"))
            if r["keep"].strip() == "1" and (splits is None or r["split"] in splits) and (domain is None or r["domain"] == domain)]


SYN = {"Development (gpt-4o-mini)": load("output_v5.csv", splits=("held_out", "adversarial")),
       "Fresh held-out (gpt-4o-mini)": load("fresh_gift_final.csv"),
       "Cross-generator (Claude)": load("cross_gen_claude_final.csv")}
for d, nm in (("data_retention", "data retention"), ("coi_disclosure", "COI disclosure"), ("ai_data_use", "AI data use")):
    SYN[f"Zero-shot: {nm}"] = load("multi_domain_final_v3.csv", domain=d)


def mcn(y, a, b):
    x, z = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    return (binomtest(min(x, z), x + z).pvalue if x + z else None), x, z


def fmt_p(p): return "(=)" if p is None else (f"(p<0.001)" if p < 0.001 else f"(p={p:.3f})")


def rate(num, den): d = int(den.sum()); return f"{int((num & den).sum())}/{d} ({(num & den).sum() / d:.0%})" if d else "--"


def ci(k, n): c = binomtest(k, n).proportion_ci(method="exact"); return f"[{c.low:.0%}, {c.high:.0%}]"


def variants(S):
    A = np.array([s["max_contra"] > CFG["tau_c"] for s in S]); B = np.array([s["min_support"] < CFG["tau_e"] for s in S])
    return {"full": (A | B).astype(int), "-A": B.astype(int), "-B": A.astype(int)}, A, B


lines = ["# DNA v2 component protocol results", f"config sha256 {FROZEN['config_sha256']}", ""]
lines.append("## Leave-one-gate-out (accuracy; McNemar vs full DNA)")
lines.append("| Set | n | Full DNA | - Gate A (contradiction) | - Gate B (unsupported) |")
iso = ["", "## Each gate in isolation (recall on targeted category / FPR on label-0 items)",
       "| Set | Gate A on Direct Failure | Gate B on Topic Drift | Gate B on Precedent Framing | Gate B on Adversarial (FP) |"]
percat = ["", "## Full DNA: flagged per category"]
for name, data in SYN.items():
    S = [core.scores(p, r) for p, r, y, c in data]
    y = np.array([d[2] for d in data]); cat = np.array([d[3] for d in data]); V, A, B = variants(S)
    row = [name, str(len(y)), f"{(V['full'] == y).mean():.1%}"]
    for k in ("-A", "-B"):
        p, *_ = mcn(y, V["full"], V[k]); row.append(f"{(V[k] == y).mean():.1%} {fmt_p(p if (V[k] != V['full']).any() else None)}")
    lines.append("| " + " | ".join(row) + " |")
    comp = y == 0
    iso.append(f"| {name} | {rate(A, cat == 'Direct Failure')} / {rate(A, comp)} | {rate(B, cat == 'Topic Drift')} / {rate(B, comp)} | "
               f"{rate(B, cat == 'Precedent Framing')} | {rate(B, cat == 'Adversarial')} |")
    percat.append(f"- {name}: " + ", ".join(f"{c} {int(V['full'][cat == c].sum())}/{int((cat == c).sum())}" for c in sorted(set(cat))))
    core.save_cache()
lines += iso + percat

# ---------------- real RAG ----------------
aud = {(r["qid"], r["answer_model"]): r for r in csv.DictReader(open(os.path.join(RAG, "results", "all_auditors_per_answer.csv"), encoding="utf-8"))}
orc = {(r["qid"], r["answer_model"]): r["oracle_chunks"].split("|")
       for r in csv.DictReader(open(os.path.join(RAG, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(RAG, "index", "chunks.jsonl"), encoding="utf-8"))}
ans = {}
for m in ("claude-haiku-4-5", "gpt-4o-mini"):
    for r in map(json.loads, open(os.path.join(RAG, "answers", f"answers_{m}.jsonl"), encoding="utf-8")): ans[(r["qid"], m)] = r

rows = []
for k in sorted(aud):
    a = ans[k]; ctx = " ".join(c["text"] for c in a["retrieved"])
    s = core.scores(ctx, a["answer"]); C, E, _ = core.nli([(ctx, a["answer"])])[0]
    row = {"qid": k[0], "answer_model": k[1], "y": int(aud[k]["y_violation"]), "clause": aud[k]["clause_in_retrieved"],
           "ret_max_contra": s["max_contra"], "ret_min_support": s["min_support"], "n_claims": s["n_claims"],
           "ret_weakest_claim": s["weakest_claim"], "ret_wholepair_C": float(C), "ret_wholepair_E": float(E),
           "dna_ret": core.decide(s, CFG)[0], "dna_ret_gate": core.decide(s, CFG)[1]}
    if k in orc:
        octx = " ".join(chunks[c] for c in orc[k]); so = core.scores(octx, a["answer"]); Co, Eo, _ = core.nli([(octx, a["answer"])])[0]
        row.update({"orc_max_contra": so["max_contra"], "orc_min_support": so["min_support"], "orc_weakest_claim": so["weakest_claim"],
                    "orc_wholepair_C": float(Co), "orc_wholepair_E": float(Eo), "dna_orc": core.decide(so, CFG)[0],
                    "dna_orc_gate": core.decide(so, CFG)[1]})
    rows.append(row)
core.save_cache()
with open(os.path.join(OUT, "dna_real_rag_per_answer.csv"), "w", newline="", encoding="utf-8") as f:
    fields = list(dict.fromkeys(x for r in rows for x in r)); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def summary(title, R, preds):
    y = np.array([r["y"] for r in R]); out = [f"", f"## {title} (n={len(R)}, violations={y.sum()})",
                                              "| Auditor | Acc | Recall (95% CI) | FPR |"]
    for nm, p in preds.items():
        p = np.asarray(p); tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        out.append(f"| {nm} | {(p == y).mean():.1%} | {tp}/{y.sum()} {ci(tp, int(y.sum()))} | {fp}/{(y == 0).sum()} ({fp / (y == 0).sum():.0%}) |")
    base = np.asarray(preds["DNA (frozen)"])
    for nm, p in preds.items():
        if nm.startswith("DNA"): continue
        pv, x, z = mcn(y, base, np.asarray(p)); out.append(f"- McNemar DNA vs {nm}: DNA-only-correct={x}, other-only-correct={z}, {fmt_p(pv)}")
    return out, y


g = lambda R, col: [int(aud[(r["qid"], r["answer_model"])][col]) for r in R]
d_ret = np.array([r["dna_ret"] for r in rows])
A = np.array([r["ret_max_contra"] > CFG["tau_c"] for r in rows]); B = np.array([r["ret_min_support"] < CFG["tau_e"] for r in rows])
out, y = summary("Real RAG, retrieved context", rows, {"DNA (frozen)": d_ret, "DNA - Gate A": B.astype(int), "DNA - Gate B": A.astype(int),
                 "MSCG (frozen)": g(rows, "mscg"), "Naive NLI": g(rows, "naive"), "GPT-5.5": g(rows, "gpt55_retrieved"),
                 "Sonnet 5.5": g(rows, "sonnet55_retrieved")})
lines += out
lines.append(f"- Isolated: Gate A recall {int((A & (y == 1)).sum())}/{y.sum()}, FPR {(A & (y == 0)).sum() / (y == 0).sum():.0%}; "
             f"Gate B recall {int((B & (y == 1)).sum())}/{y.sum()}, FPR {(B & (y == 0)).sum() / (y == 0).sum():.0%}")
lines.append("- By clause in retrieved text (DNA recall / FPR):")
for cl in ("YES", "PARTIAL", "NO", "N/A"):
    mk = np.array([r["clause"] == cl for r in rows])
    lines.append(f"  - {cl}: n={mk.sum()}, violations {int(y[mk].sum())}, recall {int((d_ret[mk] & y[mk]).sum())}/{int(y[mk].sum())}, "
                 f"FPR {int((d_ret[mk] & (1 - y[mk])).sum())}/{int((1 - y[mk]).sum())}")
lines.append(f"- Claims per answer: median {np.median([r['n_claims'] for r in rows]):.0f}; "
             f"answers with >=1 unsupported claim: {int(B.sum())}/{len(rows)}")

O = [r for r in rows if "dna_orc" in r]
d_orc = np.array([r["dna_orc"] for r in O])
Ao = np.array([r["orc_max_contra"] > CFG["tau_c"] for r in O]); Bo = np.array([r["orc_min_support"] < CFG["tau_e"] for r in O])
out, yo = summary("Oracle policy context", O, {"DNA (frozen)": d_orc, "DNA - Gate A": Bo.astype(int), "DNA - Gate B": Ao.astype(int),
                  "DNA retrieved (same 188)": [r["dna_ret"] for r in O], "MSCG oracle": g(O, "mscg_oracle"),
                  "GPT-5.5 oracle": g(O, "gpt55_oracle"), "Sonnet 5.5 oracle": g(O, "sonnet55_oracle")})
lines += out

lines += ["", "## POST-HOC threshold-free signal quality: AUROC for violation (bootstrap 95% CI, 2000 resamples)",
          "| Score | Retrieved (208) | Oracle (188) |"]
rng = np.random.default_rng(0)


def auc(yv, sv):
    yv, sv = np.asarray(yv), np.asarray(sv); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        if 0 < yv[i].sum() < len(i): bs.append(roc_auc_score(yv[i], sv[i]))
    return f"{roc_auc_score(yv, sv):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


for nm, kr, ko, sign in (("DNA max claim contradiction", "ret_max_contra", "orc_max_contra", 1),
                         ("DNA 1 - min claim support", "ret_min_support", "orc_min_support", -1),
                         ("Whole-pair contradiction C (MSCG signal)", "ret_wholepair_C", "orc_wholepair_C", 1),
                         ("Whole-pair 1 - entailment E", "ret_wholepair_E", "orc_wholepair_E", -1)):
    lines.append(f"| {nm} | {auc(y, [sign * r[kr] for r in rows])} | {auc(yo, [sign * r[ko] for r in O])} |")

txt = "\n".join(lines); print(txt); open(os.path.join(OUT, "dna_results.md"), "w", encoding="utf-8").write(txt + "\n")
