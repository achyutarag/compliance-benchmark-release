"""Question-aware judge control (judges/QUESTION_AWARE_PREREGISTRATION.md). Same judges/items/settings/rubric as judges.py; the employee's question is added.
  python judges_q.py freeze | submit | status | collect
Keys read from the environment / user registry, never printed; clients pinned to the providers' own base URLs."""
import csv, datetime, hashlib, json, os, sys, winreg
HERE = os.path.dirname(os.path.abspath(__file__)); J = os.path.join(HERE, "judges")
CFG = os.path.join(J, "JUDGE_CONFIG_Q.json"); STATE = os.path.join(J, "batch_state_q.json")
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    if not os.environ.get(k):
        os.environ[k] = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), k)[0]

COMP = json.load(open(os.path.join(J, "JUDGE_CONFIG.json"), encoding="utf-8"))
OLD = "You will be shown policy text and a response that an HR assistant gave to an employee."
NEW = "You will be shown policy text, the employee's question, and the response that an HR assistant gave to that employee."
assert COMP["system_prompt"].count(OLD) == 1
SYSTEM = COMP["system_prompt"].replace(OLD, NEW)
USER = "Policy text:\n<<<\n{policy}\n>>>\n\nEmployee question:\n<<<\n{question}\n>>>\n\nResponse to audit:\n<<<\n{response}\n>>>"
SCHEMA = COMP["output_schema"]; MODELS = COMP["models"]; MAX_TOKENS = COMP["settings"]["max_tokens"]
PRICES = {"openai": (5.00, 30.00), "anthropic": (2.00, 10.00)}
ANSWER_MODELS = ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it")
safe_id = lambda c: c.replace(".", "_d_"); unsafe_id = lambda c: c.replace("_d_", ".")


def items():
    chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8"))}
    oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(HERE, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
    out = []
    for m in ANSWER_MODELS:
        for a in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
            out.append((f"retrieved__{a['qid']}__{m}", a["qid"], m, "\n\n".join(c["text"] for c in a["retrieved"]), a["question"], a["answer"]))
            if a["qid"] in oracle:
                out.append((f"oracle__{a['qid']}__{m}", a["qid"], m, "\n\n".join(chunks[c] for c in oracle[a["qid"]]), a["question"], a["answer"]))
    return out


def user_msg(pol, q, resp): return USER.format(policy=pol, question=q, response=resp)
def openai_body(pol, q, resp):
    return {"model": MODELS["openai"], "reasoning_effort": "low", "max_completion_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(pol, q, resp)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}}}
def anthropic_params(pol, q, resp):
    return {"model": MODELS["anthropic"], "max_tokens": MAX_TOKENS, "system": SYSTEM, "messages": [{"role": "user", "content": user_msg(pol, q, resp)}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}
def digest(): return hashlib.sha256(json.dumps({"system_prompt": SYSTEM, "user_template": USER, "output_schema": SCHEMA, "models": MODELS, "max_tokens": MAX_TOKENS}, sort_keys=True).encode()).hexdigest()
def check_frozen():
    cfg = json.load(open(CFG, encoding="utf-8")); assert cfg["sha256"] == digest(), "config changed since freezing"; return cfg


cmd = sys.argv[1]
if cmd == "freeze":
    assert not os.path.exists(CFG), "already frozen"
    diff = [l for l in SYSTEM.splitlines() if l not in COMP["system_prompt"].splitlines()]
    print("system-prompt lines that differ from the compliance rubric:", len(diff))
    json.dump({"system_prompt": SYSTEM, "user_template": USER, "output_schema": SCHEMA, "models": MODELS, "max_tokens": MAX_TOKENS,
               "preregistration": "judges/QUESTION_AWARE_PREREGISTRATION.md", "sha256": digest(), "frozen_at": datetime.datetime.now().isoformat(timespec="seconds")},
              open(CFG, "w", encoding="utf-8"), indent=2)
    its = items(); print("frozen", digest()[:16], "| items per judge:", len(its))
    # token estimate for cost
    est = sum(len(user_msg(p, q, r)) + len(SYSTEM) for _, _, _, p, q, r in its) / 4
    print(f"approx input tokens per judge: {est:,.0f} -> openai ~${est * 5 / 1e6 / 2 + 792 * 95 * 30 / 1e6 / 2:.2f}, anthropic ~${est * 2 / 1e6 / 2 + 792 * 85 * 10 / 1e6 / 2:.2f} (batch)")

elif cmd == "submit":
    check_frozen(); its = items()
    from openai import OpenAI
    import anthropic
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {"submitted_at": datetime.datetime.now().isoformat(timespec="seconds"), "n_requests": len(its)}
    assert "anthropic_batch_id" not in state, "already submitted"
    if "openai_batch_id" not in state:
        jl = os.path.join(J, "openai_batch_input_q.jsonl")
        with open(jl, "w", encoding="utf-8") as f:
            for cid, _, _, pol, q, resp in its: f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": openai_body(pol, q, resp)}) + "\n")
        oc = OpenAI(base_url="https://api.openai.com/v1"); fo = oc.files.create(file=open(jl, "rb"), purpose="batch")
        ob = oc.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"study": "question-aware judge control"})
        state["openai_batch_id"] = ob.id; json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)
    ab = anthropic.Anthropic(base_url="https://api.anthropic.com").messages.batches.create(
        requests=[{"custom_id": safe_id(cid), "params": anthropic_params(pol, q, resp)} for cid, _, _, pol, q, resp in its])
    state["anthropic_batch_id"] = ab.id; json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)
    print("submitted: openai", state["openai_batch_id"][:12] + "...", "| anthropic", state["anthropic_batch_id"][:12] + "...", "| n", len(its))

elif cmd == "status":
    st = json.load(open(STATE, encoding="utf-8"))
    from openai import OpenAI
    import anthropic
    ob = OpenAI(base_url="https://api.openai.com/v1").batches.retrieve(st["openai_batch_id"]); ab = anthropic.Anthropic(base_url="https://api.anthropic.com").messages.batches.retrieve(st["anthropic_batch_id"])
    print(f"openai:    {ob.status} | completed {ob.request_counts.completed}/{ob.request_counts.total}, failed {ob.request_counts.failed}")
    print(f"anthropic: {ab.processing_status} | succeeded {ab.request_counts.succeeded}, errored {ab.request_counts.errored}, processing {ab.request_counts.processing}")

elif cmd == "collect":
    st = json.load(open(STATE, encoding="utf-8"))
    from openai import OpenAI
    import anthropic
    rows, usage = [], {"openai": [0, 0], "anthropic": [0, 0]}
    oc = OpenAI(base_url="https://api.openai.com/v1"); ob = oc.batches.retrieve(st["openai_batch_id"]); assert ob.status == "completed", ob.status
    raw = oc.files.content(ob.output_file_id).text; open(os.path.join(J, "openai_batch_output_q.jsonl"), "w", encoding="utf-8").write(raw)
    for line in raw.splitlines():
        r = json.loads(line); b = (r.get("response") or {}).get("body") or {}; u = b.get("usage", {})
        usage["openai"][0] += u.get("prompt_tokens", 0); usage["openai"][1] += u.get("completion_tokens", 0)
        rows.append(("openai", r["custom_id"], b.get("model", ""), b["choices"][0].get("finish_reason", "") if b.get("choices") else "error", b["choices"][0]["message"]["content"] if b.get("choices") else ""))
    ac = anthropic.Anthropic(base_url="https://api.anthropic.com")
    with open(os.path.join(J, "anthropic_batch_output_q.jsonl"), "w", encoding="utf-8") as f:
        for res in ac.messages.batches.results(st["anthropic_batch_id"]):
            f.write(res.model_dump_json() + "\n")
            if res.result.type == "succeeded":
                m = res.result.message; usage["anthropic"][0] += m.usage.input_tokens; usage["anthropic"][1] += m.usage.output_tokens
                rows.append(("anthropic", unsafe_id(res.custom_id), m.model, m.stop_reason, next((x.text for x in m.content if x.type == "text"), "")))
            else: rows.append(("anthropic", unsafe_id(res.custom_id), "", res.result.type, ""))
    out, bad = [], []
    for prov, cid, served, stop, txt in rows:
        cond, q, m = cid.split("__")
        try: v = json.loads(txt); verdict, why = v["verdict"], v["rationale"]; assert verdict in ("COMPLIANT", "VIOLATION")
        except Exception: verdict, why = "", ""; bad.append((prov, cid, stop))
        out.append({"judge": MODELS[prov], "provider": prov, "served_model": served, "condition": cond, "qid": q, "answer_model": m, "verdict": verdict,
                    "pred_violation": "" if not verdict else int(verdict == "VIOLATION"), "rationale": why, "stop_reason": stop})
    with open(os.path.join(J, "q_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print("per provider/condition:", dict(Counter((o["provider"], o["condition"]) for o in out)), "| unparseable/failed:", len(bad), bad[:5])
    print("served:", dict(Counter((o["provider"], o["served_model"]) for o in out)))
    for p, (i_, o_) in usage.items(): print(f"{p}: tokens in/out {i_}/{o_} -> batch cost ~${(i_ * PRICES[p][0] + o_ * PRICES[p][1]) / 1e6 * 0.5:.2f}")
