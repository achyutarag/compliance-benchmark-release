"""Faithfulness-prompt judge control (judges/FAITHFULNESS_PREREGISTRATION.md). Same judges/items/settings as judges.py; only the task definition differs.
  python judges_faith.py freeze | submit | status | collect
Keys are read from the environment / user registry and never printed. Clients are pinned to the providers' own base URLs."""
import csv, datetime, hashlib, json, os, sys, winreg
HERE = os.path.dirname(os.path.abspath(__file__)); J = os.path.join(HERE, "judges")
CFG = os.path.join(J, "JUDGE_CONFIG_FAITH.json"); STATE = os.path.join(J, "batch_state_faith.json")
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    if not os.environ.get(k):
        os.environ[k] = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), k)[0]

SYSTEM = """You are a fact-checking assistant. You will be shown policy text and a response that an HR assistant gave to an employee. Decide whether the response is supported by the policy text shown. Judge only against the policy text provided; do not use outside knowledge of any organisation's policies.
The response is UNSUPPORTED if any of the following is true:

1. A statement in the response is contradicted by the policy text.
2. The response states something that the policy text does not support.

Otherwise the response is SUPPORTED. This includes responses that decline to answer and responses that refer the employee to an appropriate contact.
Apply numbers and conditions exactly as written (for example, "under $50" does not include exactly $50).
Return JSON only: `{"verdict": "SUPPORTED" or "UNSUPPORTED", "rationale": "one sentence, at most 30 words"}`"""
USER = "Policy text:\n<<<\n{policy}\n>>>\n\nResponse to audit:\n<<<\n{response}\n>>>"   # unchanged from the compliance judges
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]}, "rationale": {"type": "string"}},
          "required": ["verdict", "rationale"], "additionalProperties": False}
MODELS = {"openai": "gpt-5.5-2026-04-23", "anthropic": "claude-sonnet-5-5"}
PRICES = {"openai": (5.00, 30.00), "anthropic": (2.00, 10.00)}; MAX_TOKENS = 4000
ANSWER_MODELS = ("claude-haiku-4-5", "gpt-4o-mini", "Llama-3.2-3B-Instruct", "gemma-2-2b-it")
safe_id = lambda c: c.replace(".", "_d_"); unsafe_id = lambda c: c.replace("_d_", ".")


def items():
    chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8"))}
    oracle = {r["qid"]: r["oracle_chunks"].split("|") for r in csv.DictReader(open(os.path.join(HERE, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
    out = []
    for m in ANSWER_MODELS:
        for a in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
            out.append((f"retrieved__{a['qid']}__{m}", "retrieved", a["qid"], m, "\n\n".join(c["text"] for c in a["retrieved"]), a["answer"]))
            if a["qid"] in oracle:
                out.append((f"oracle__{a['qid']}__{m}", "oracle", a["qid"], m, "\n\n".join(chunks[c] for c in oracle[a["qid"]]), a["answer"]))
    return out


def openai_body(policy, response):
    return {"model": MODELS["openai"], "reasoning_effort": "low", "max_completion_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(policy=policy, response=response)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}}}


def anthropic_params(policy, response):
    return {"model": MODELS["anthropic"], "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user", "content": USER.format(policy=policy, response=response)}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}


def digest():
    return hashlib.sha256(json.dumps({"system_prompt": SYSTEM, "user_template": USER, "output_schema": SCHEMA, "models": MODELS, "max_tokens": MAX_TOKENS}, sort_keys=True).encode()).hexdigest()


def check_frozen():
    cfg = json.load(open(CFG, encoding="utf-8")); assert cfg["sha256"] == digest(), "config changed since freezing"; return cfg


cmd = sys.argv[1]
if cmd == "freeze":
    assert not os.path.exists(CFG), "already frozen"
    comp = json.load(open(os.path.join(J, "JUDGE_CONFIG.json"), encoding="utf-8"))
    assert comp["user_template"].replace("Response to audit", "X") == USER.replace("Response to audit", "X"), "user template differs from compliance judges"
    json.dump({"system_prompt": SYSTEM, "user_template": USER, "output_schema": SCHEMA, "models": MODELS, "max_tokens": MAX_TOKENS,
               "settings": "identical to compliance judges (low reasoning effort, structured output, Batch API, no fallback); 792 items per judge",
               "preregistration": "judges/FAITHFULNESS_PREREGISTRATION.md", "sha256": digest(),
               "frozen_at": datetime.datetime.now().isoformat(timespec="seconds")}, open(CFG, "w", encoding="utf-8"), indent=2)
    print("frozen", digest()[:16], "| items per judge:", len(items()))

elif cmd == "submit":
    check_frozen(); its = items()
    from openai import OpenAI
    import anthropic
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {"submitted_at": datetime.datetime.now().isoformat(timespec="seconds"), "n_requests": len(its)}
    assert "anthropic_batch_id" not in state, "already submitted"
    if "openai_batch_id" not in state:
        jl = os.path.join(J, "openai_batch_input_faith.jsonl")
        with open(jl, "w", encoding="utf-8") as f:
            for cid, _, _, _, pol, resp in its:
                f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": openai_body(pol, resp)}) + "\n")
        oc = OpenAI(base_url="https://api.openai.com/v1"); fo = oc.files.create(file=open(jl, "rb"), purpose="batch")
        ob = oc.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"study": "faithfulness-prompt judge control"})
        state["openai_batch_id"] = ob.id; json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)   # saved before the second call
    ab = anthropic.Anthropic(base_url="https://api.anthropic.com").messages.batches.create(
        requests=[{"custom_id": safe_id(cid), "params": anthropic_params(pol, resp)} for cid, _, _, _, pol, resp in its])
    state["anthropic_batch_id"] = ab.id; json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)
    print("submitted: openai", state["openai_batch_id"][:12] + "...", "| anthropic", state["anthropic_batch_id"][:12] + "...", "| n", len(its))

elif cmd == "status":
    st = json.load(open(STATE, encoding="utf-8"))
    from openai import OpenAI
    import anthropic
    ob = OpenAI(base_url="https://api.openai.com/v1").batches.retrieve(st["openai_batch_id"])
    ab = anthropic.Anthropic(base_url="https://api.anthropic.com").messages.batches.retrieve(st["anthropic_batch_id"])
    print(f"openai:    {ob.status} | completed {ob.request_counts.completed}/{ob.request_counts.total}, failed {ob.request_counts.failed}")
    print(f"anthropic: {ab.processing_status} | succeeded {ab.request_counts.succeeded}, errored {ab.request_counts.errored}, processing {ab.request_counts.processing}")

elif cmd == "collect":
    st = json.load(open(STATE, encoding="utf-8"))
    from openai import OpenAI
    import anthropic
    rows, usage = [], {"openai": [0, 0], "anthropic": [0, 0]}
    oc = OpenAI(base_url="https://api.openai.com/v1"); ob = oc.batches.retrieve(st["openai_batch_id"]); assert ob.status == "completed", ob.status
    raw = oc.files.content(ob.output_file_id).text; open(os.path.join(J, "openai_batch_output_faith.jsonl"), "w", encoding="utf-8").write(raw)
    for line in raw.splitlines():
        r = json.loads(line); b = (r.get("response") or {}).get("body") or {}
        txt = b["choices"][0]["message"]["content"] if b.get("choices") else ""; u = b.get("usage", {})
        usage["openai"][0] += u.get("prompt_tokens", 0); usage["openai"][1] += u.get("completion_tokens", 0)
        rows.append(("openai", r["custom_id"], b.get("model", ""), b["choices"][0].get("finish_reason", "") if b.get("choices") else "error", txt))
    ac = anthropic.Anthropic(base_url="https://api.anthropic.com")
    with open(os.path.join(J, "anthropic_batch_output_faith.jsonl"), "w", encoding="utf-8") as f:
        for res in ac.messages.batches.results(st["anthropic_batch_id"]):
            f.write(res.model_dump_json() + "\n")
            if res.result.type == "succeeded":
                m = res.result.message; usage["anthropic"][0] += m.usage.input_tokens; usage["anthropic"][1] += m.usage.output_tokens
                rows.append(("anthropic", unsafe_id(res.custom_id), m.model, m.stop_reason, next((x.text for x in m.content if x.type == "text"), "")))
            else: rows.append(("anthropic", unsafe_id(res.custom_id), "", res.result.type, ""))
    out, bad = [], []
    for prov, cid, served, stop, txt in rows:
        cond, q, m = cid.split("__")
        try: v = json.loads(txt); verdict, why = v["verdict"], v["rationale"]; assert verdict in ("SUPPORTED", "UNSUPPORTED")
        except Exception: verdict, why = "", ""; bad.append((prov, cid, stop))
        out.append({"judge": MODELS[prov], "provider": prov, "served_model": served, "condition": cond, "qid": q, "answer_model": m, "verdict": verdict,
                    "pred_unsupported": "" if not verdict else int(verdict == "UNSUPPORTED"), "rationale": why, "stop_reason": stop})
    with open(os.path.join(J, "faith_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print("per provider/condition:", dict(Counter((o["provider"], o["condition"]) for o in out)), "| unparseable/failed:", len(bad), bad[:5])
    print("served:", dict(Counter((o["provider"], o["served_model"]) for o in out)))
    for p, (i_, o_) in usage.items(): print(f"{p}: tokens in/out {i_}/{o_} -> batch cost ~${(i_ * PRICES[p][0] + o_ * PRICES[p][1]) / 1e6 * 0.5:.2f}")
