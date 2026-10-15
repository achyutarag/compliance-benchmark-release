"""
LLM-judge baseline on the 208 labelled real-RAG answers (crossed design: 2 judge families x 2 answer models).

  python judges.py freeze    write judges/JUDGE_CONFIG.json (prompt, models, settings) + sha256; refuses to overwrite
  python judges.py test      one direct (non-batch) call per judge on a real item; prints tokens and projected cost
  python judges.py submit    build and submit both batches (retrieved context: 208, oracle context: 188)
  python judges.py status    show batch status
  python judges.py collect   download, validate, write judges/judge_predictions.csv

Judges see only (policy text, response), like MSCG; never the question, gold verdict, model names or labels.
"""
import csv, datetime, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); J = os.path.join(HERE, "judges"); os.makedirs(J, exist_ok=True)
CFG = os.path.join(J, "JUDGE_CONFIG.json")
# Answer set: "orig" (claude-haiku-4-5, gpt-4o-mini) or "ext" (Llama-3.2-3B-Instruct, gemma-2-2b-it; EXTENSION_PREREGISTRATION.md).
# Same frozen judge config for both; only the items and output file names differ.  Usage: python judges.py <cmd> [orig|ext]
SET = sys.argv[2] if len(sys.argv) > 2 else "orig"; assert SET in ("orig", "ext")
SFX = "" if SET == "orig" else "_ext"
ANSWER_MODELS = {"orig": ("claude-haiku-4-5", "gpt-4o-mini"), "ext": ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")}[SET]
LABELS = {"orig": "labels_FROZEN_2026-09-30.csv", "ext": "labels_ext_FROZEN_2026-10-01.csv"}[SET]
STATE = os.path.join(J, f"batch_state{SFX}.json")
if not os.environ.get("ANTHROPIC_API_KEY"):   # set with setx after this session started: read from the user environment, never printed
    import winreg
    os.environ["ANTHROPIC_API_KEY"] = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "ANTHROPIC_API_KEY")[0]

SYSTEM = """You are a compliance auditor. You will be shown policy text and a response that an HR assistant gave to an employee. Decide whether the response is compliant with the policy text shown. Judge only against the policy text provided; do not use outside knowledge of any organisation's policies.
The response is a VIOLATION if any of the following is true:

1. An employee who followed the response would act against the policy.
2. The response misstates a rule in the policy (a threshold, condition, exception, or who must approve) in a way that matters.
3. The response asserts a policy rule that does not appear in the policy text.

Otherwise the response is COMPLIANT. This includes responses that are more cautious than the policy requires (for example, advising the employee to decline something the policy allows), responses that decline to answer, and responses that refer the employee to an appropriate contact.
Apply numbers and conditions exactly as written (for example, "under $50" does not include exactly $50).
Return JSON only: `{"verdict": "COMPLIANT" or "VIOLATION", "rationale": "one sentence, at most 30 words"}`"""
USER = "Policy text:\n<<<\n{policy}\n>>>\n\nResponse to audit:\n<<<\n{response}\n>>>"
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["COMPLIANT", "VIOLATION"]},
                                           "rationale": {"type": "string"}},
          "required": ["verdict", "rationale"], "additionalProperties": False}
MODELS = {"openai": "gpt-5.5-2026-04-23", "anthropic": "claude-sonnet-5-5"}
PRICES = {"openai": (5.00, 30.00), "anthropic": (2.00, 10.00)}   # $/1M tokens (in, out); batch = 50% off
MAX_TOKENS = 4000


def items():
    """(custom_id, condition, qid, answer_model, policy_text, response) for both conditions."""
    chunks = {c["chunk_id"]: c for c in map(json.loads, open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8"))}
    ans = {}
    for m in ANSWER_MODELS:
        for r in map(json.loads, open(os.path.join(HERE, "answers", f"answers_{m}.jsonl"), encoding="utf-8")):
            ans[(r["qid"], m)] = r
    labels = {(r["qid"], r["answer_model"]) for r in csv.DictReader(open(os.path.join(HERE, LABELS), encoding="utf-8"))}
    out = []
    for (q, m) in sorted(labels):
        a = ans[(q, m)]
        out.append((f"retrieved__{q}__{m}", "retrieved", q, m, "\n\n".join(c["text"] for c in a["retrieved"]), a["answer"]))
    # oracle chunks depend only on the question (chosen from the frozen key, independent of answers): reuse the frozen selection
    oracle = {r["qid"]: r["oracle_chunks"] for r in csv.DictReader(open(os.path.join(HERE, "results", "mscg_oracle_context_diagnostic.csv"), encoding="utf-8"))}
    for q in sorted(oracle):
        pol = "\n\n".join(chunks[cid]["text"] for cid in oracle[q].split("|"))
        for m in ANSWER_MODELS:
            out.append((f"oracle__{q}__{m}", "oracle", q, m, pol, ans[(q, m)]["answer"]))
    return out


def safe_id(cid):     # Anthropic custom_id allows only [a-zA-Z0-9_-]; names like Llama-3.2-3B contain dots
    return cid.replace(".", "_d_")


def unsafe_id(cid):
    return cid.replace("_d_", ".")


def openai_body(policy, response):
    return {"model": MODELS["openai"], "reasoning_effort": "low", "max_completion_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(policy=policy, response=response)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}}}


def anthropic_params(policy, response):
    return {"model": MODELS["anthropic"], "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user", "content": USER.format(policy=policy, response=response)}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}


def check_frozen():
    cfg = json.load(open(CFG, encoding="utf-8"))
    body = json.dumps({k: cfg[k] for k in ("system_prompt", "user_template", "output_schema", "models", "settings")}, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == cfg["sha256"], "judge config changed since freezing"
    assert cfg["system_prompt"] == SYSTEM and cfg["models"] == MODELS, "code differs from frozen config"
    return cfg


cmd = sys.argv[1]
if cmd == "freeze":
    assert not os.path.exists(CFG), "already frozen"
    cfg = {"system_prompt": SYSTEM, "user_template": USER, "output_schema": SCHEMA, "models": MODELS,
           "settings": {"reasoning_effort": "low", "max_tokens": MAX_TOKENS, "structured_output": True, "batch_api": True,
                        "server_side_fallback": False, "inputs": "policy text + response only (no question, gold, labels or model names)",
                        "conditions": {"retrieved": "chunks the RAG system retrieved (same text MSCG saw)",
                                       "oracle": "2 chunks best covering the adjudicated cited clauses (oracle policy context)"}}}
    cfg["sha256"] = hashlib.sha256(json.dumps({k: cfg[k] for k in ("system_prompt", "user_template", "output_schema", "models", "settings")},
                                              sort_keys=True).encode()).hexdigest()
    cfg["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    cfg["approved_by"] = "user, 2026-09-30 (prompt approved verbatim)"
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), indent=2)
    print("frozen", CFG, cfg["sha256"])

elif cmd == "test":
    check_frozen(); its = items()
    print(f"items: {sum(i[1]=='retrieved' for i in its)} retrieved + {sum(i[1]=='oracle' for i in its)} oracle = {len(its)} per judge")
    longest = max(its, key=lambda i: len(i[4]) + len(i[5]))
    from openai import OpenAI
    import anthropic
    for name, it in (("typical", its[0]), ("longest", longest)):
        r = OpenAI().chat.completions.create(**openai_body(it[4], it[5]))
        ou = r.usage
        print(f"[openai {name}] served={r.model} finish={r.choices[0].finish_reason} in/out={ou.prompt_tokens}/{ou.completion_tokens} -> {r.choices[0].message.content}")
        a = anthropic.Anthropic().messages.create(**anthropic_params(it[4], it[5]))
        txt = next((b.text for b in a.content if b.type == "text"), "")
        print(f"[anthropic {name}] served={a.model} stop={a.stop_reason} in/out={a.usage.input_tokens}/{a.usage.output_tokens} -> {txt}")
        if name == "typical":
            n = len(its)
            for prov, (i_, o_) in (("openai", (ou.prompt_tokens, ou.completion_tokens)), ("anthropic", (a.usage.input_tokens, a.usage.output_tokens))):
                pi, po = PRICES[prov]
                print(f"   projected {prov} batch cost for {n} calls: ${n * (i_ * pi + o_ * po) / 1e6 * 0.5:.2f} (from this item's tokens)")

elif cmd == "submit":
    check_frozen(); its = items()
    from openai import OpenAI
    import anthropic
    # resume: if the OpenAI batch already exists (state saved), submit only the missing Anthropic batch
    state = json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else \
        {"submitted_at": datetime.datetime.now().isoformat(timespec="seconds"), "n_requests": len(its)}
    assert "anthropic_batch_id" not in state, "batches already submitted"
    if "openai_batch_id" not in state:
        jl = os.path.join(J, f"openai_batch_input{SFX}.jsonl")
        with open(jl, "w", encoding="utf-8") as f:
            for cid, *_ , pol, resp in its:
                f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": openai_body(pol, resp)}) + "\n")
        oc = OpenAI(); fo = oc.files.create(file=open(jl, "rb"), purpose="batch")
        ob = oc.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions", completion_window="24h",
                               metadata={"study": "real-RAG compliance judge", "config_sha256": check_frozen()["sha256"][:16]})
        state["openai_batch_id"] = ob.id
        json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)   # saved before the second call, so a failure cannot orphan it
    ab = anthropic.Anthropic().messages.batches.create(requests=[{"custom_id": safe_id(cid), "params": anthropic_params(pol, resp)}
                                                                 for cid, *_, pol, resp in its])
    state["anthropic_batch_id"] = ab.id
    json.dump(state, open(STATE, "w", encoding="utf-8"), indent=2)
    print("submitted:", state)

elif cmd == "status":
    st = json.load(open(STATE, encoding="utf-8"))
    from openai import OpenAI
    import anthropic
    ob = OpenAI().batches.retrieve(st["openai_batch_id"])
    ab = anthropic.Anthropic().messages.batches.retrieve(st["anthropic_batch_id"])
    print(f"openai:    {ob.status} | completed {ob.request_counts.completed}/{ob.request_counts.total}, failed {ob.request_counts.failed}")
    print(f"anthropic: {ab.processing_status} | succeeded {ab.request_counts.succeeded}, errored {ab.request_counts.errored}, processing {ab.request_counts.processing}")

elif cmd == "collect":
    st = json.load(open(STATE, encoding="utf-8")); its = {i[0]: i for i in items()}
    from openai import OpenAI
    import anthropic
    rows, usage = [], {"openai": [0, 0], "anthropic": [0, 0]}
    oc = OpenAI(); ob = oc.batches.retrieve(st["openai_batch_id"]); assert ob.status == "completed", ob.status
    raw = oc.files.content(ob.output_file_id).text
    open(os.path.join(J, f"openai_batch_output{SFX}.jsonl"), "w", encoding="utf-8").write(raw)
    for line in raw.splitlines():
        r = json.loads(line); b = (r.get("response") or {}).get("body") or {}
        txt = b["choices"][0]["message"]["content"] if b.get("choices") else ""
        u = b.get("usage", {}); usage["openai"][0] += u.get("prompt_tokens", 0); usage["openai"][1] += u.get("completion_tokens", 0)
        rows.append(("openai", r["custom_id"], b.get("model", ""), b["choices"][0].get("finish_reason", "") if b.get("choices") else "error", txt))
    ac = anthropic.Anthropic()
    with open(os.path.join(J, f"anthropic_batch_output{SFX}.jsonl"), "w", encoding="utf-8") as f:
        for res in ac.messages.batches.results(st["anthropic_batch_id"]):
            f.write(res.model_dump_json() + "\n")
            if res.result.type == "succeeded":
                m = res.result.message; txt = next((x.text for x in m.content if x.type == "text"), "")
                usage["anthropic"][0] += m.usage.input_tokens; usage["anthropic"][1] += m.usage.output_tokens
                rows.append(("anthropic", unsafe_id(res.custom_id), m.model, m.stop_reason, txt))
            else:
                rows.append(("anthropic", unsafe_id(res.custom_id), "", res.result.type, ""))
    out, bad = [], []
    for prov, cid, served, stop, txt in rows:
        cond, q, m = cid.split("__")
        try:
            v = json.loads(txt); verdict, why = v["verdict"], v["rationale"]; assert verdict in ("COMPLIANT", "VIOLATION")
        except Exception:
            verdict, why = "", ""; bad.append((prov, cid, stop))
        out.append({"judge": MODELS[prov], "provider": prov, "served_model": served, "condition": cond, "qid": q, "answer_model": m,
                    "verdict": verdict, "pred_violation": "" if not verdict else int(verdict == "VIOLATION"),
                    "rationale": why, "stop_reason": stop})
    with open(os.path.join(J, f"judge_predictions{SFX}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print("results per provider/condition:", dict(Counter((o["provider"], o["condition"]) for o in out)))
    print("expected per provider:", len(its), "| unparseable/failed:", len(bad), bad[:6])
    print("served models:", dict(Counter((o["provider"], o["served_model"]) for o in out)))
    for p, (i_, o_) in usage.items():
        print(f"{p}: tokens in/out {i_}/{o_} -> batch cost ~${(i_ * PRICES[p][0] + o_ * PRICES[p][1]) / 1e6 * 0.5:.2f}")
