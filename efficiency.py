"""Cost and latency of the real-RAG auditors (post-hoc, descriptive).
  local:  wall-clock per answer for MSCG (all 416, retrieved context) and DNA (first 60 answers, NLI cache bypassed), CPU only
  judges: tokens per call from the saved batch outputs (both runs, 792 calls per judge); cost at batch and standard prices;
          synchronous latency from 10 direct calls per judge on fixed items (same frozen prompt).
Writes results/efficiency.txt."""
import csv, json, os, platform, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); J = os.path.join(HERE, "judges")
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

# ---------- local auditors
import torch, run_mscg_core as core
sys.path.insert(0, os.path.join(HERE, "dna")); import dna_core as dna
P(f"hardware: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, CUDA={torch.cuda.is_available()}, torch threads={torch.get_num_threads()}")
ans = []
for m in ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it"):
    ans += [json.loads(l) for l in open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")]
pairs = [(" ".join(c["text"] for c in a["retrieved"]), a["answer"]) for a in ans]
t0 = time.perf_counter(); guard = core.load_guard(); P(f"MSCG model load: {time.perf_counter() - t0:.1f}s (once)")
core.signals(guard, *pairs[0])  # warm-up
ts = []
for p, r in pairs:
    t = time.perf_counter(); core.decide(core.signals(guard, p, r)); ts.append(time.perf_counter() - t)
P(f"MSCG per answer (n={len(ts)}): median {statistics.median(ts) * 1000:.0f} ms, p95 {sorted(ts)[int(.95 * len(ts))] * 1000:.0f} ms, total {sum(ts):.1f}s")
dna._load(); dna._cache = {}  # bypass cache: time real NLI work
FROZEN = json.load(open(os.path.join(HERE, "dna", "DNA_CONFIG_FROZEN.json")))["config"]
ts = []; npairs = []
for p, r in pairs[:60]:
    t = time.perf_counter(); s = dna.scores(p, r); dna.decide(s, FROZEN); ts.append(time.perf_counter() - t); npairs.append(s["n_claims"] * s["n_segments"])
P(f"DNA per answer (n=60, uncached): median {statistics.median(ts):.2f} s, p95 {sorted(ts)[int(.95 * 60)]:.2f} s, median NLI pairs {statistics.median(npairs):.0f}")

# ---------- judges: tokens from saved batch outputs
PRICES = {"openai": (5.00, 30.00), "anthropic": (2.00, 10.00)}  # $/1M (in, out), standard; batch = 50%
tok = {"openai": [], "anthropic": []}
for sfx in ("", "_ext"):
    for l in open(os.path.join(J, f"openai_batch_output{sfx}.jsonl"), encoding="utf-8"):
        u = ((json.loads(l).get("response") or {}).get("body") or {}).get("usage") or {}
        if u: tok["openai"].append((u["prompt_tokens"], u["completion_tokens"]))
    for l in open(os.path.join(J, f"anthropic_batch_output{sfx}.jsonl"), encoding="utf-8"):
        r = json.loads(l)["result"]
        if r.get("type") == "succeeded": tok["anthropic"].append((r["message"]["usage"]["input_tokens"], r["message"]["usage"]["output_tokens"]))
for prov, tl in tok.items():
    i = statistics.mean(t[0] for t in tl); o = statistics.mean(t[1] for t in tl); pi, po = PRICES[prov]
    std = (i * pi + o * po) / 1e6
    P(f"{prov} judge: {len(tl)} calls, mean tokens in/out {i:.0f}/{o:.0f}; cost per answer ${std:.4f} standard, ${std / 2:.4f} batch; "
      f"per 1,000 answers ${std * 1000:.2f} standard, ${std * 500:.2f} batch")

# ---------- judges: synchronous latency, 10 fixed items each
if os.environ.get("SKIP_LIVE") != "1":
    sys.argv = ["judges.py", "x", "orig"]; g = {"__file__": os.path.join(HERE, "judges.py"), "__name__": "x"}
    exec(open(os.path.join(HERE, "judges.py"), encoding="utf-8").read().split("cmd = sys.argv[1]")[0], g)
    g["check_frozen"](); its = [i for i in g["items"]() if i[1] == "retrieved"][::21][:10]
    from openai import OpenAI
    import anthropic
    oc, ac = OpenAI(), anthropic.Anthropic()
    for prov in ("openai", "anthropic"):
        lat = []
        for _, _, q, m, pol, resp in its:
            t = time.perf_counter()
            if prov == "openai": oc.chat.completions.create(**g["openai_body"](pol, resp))
            else: ac.messages.create(**g["anthropic_params"](pol, resp))
            lat.append(time.perf_counter() - t)
        P(f"{prov} judge synchronous latency (n=10): median {statistics.median(lat):.1f} s, max {max(lat):.1f} s")
open(os.path.join(HERE, "results", "efficiency.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
