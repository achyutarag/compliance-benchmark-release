"""Runs the full-policy OpenAI requests as sequential batches under the provider's enqueued-token limit (judges/FULL_POLICY_PREREGISTRATION.md, deviation log).
Resumable; state in judges/batch_state_full_openai.json. Keys read from the environment/registry, never printed."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "judges_full.py"), encoding="utf-8").read(); src = src[:src.index("cmd = sys.argv[1]")]
g = {"__name__": "jf", "__file__": os.path.join(HERE, "judges_full.py")}; exec(compile(src, "jf", "exec"), g)
from openai import OpenAI
g["check_frozen"](); its = g["items"](); J = g["J"]; SP = os.path.join(J, "batch_state_full_openai.json")
oc = OpenAI(base_url="https://api.openai.com/v1")
CAP = 0.85e6   # estimated tokens (chars/4) per part; real tokens are about 1.3x
parts, cur, tot = [], [], 0
for it in its:
    est = (len(g["user_msg"](it[3], it[4], it[5])) + len(g["SYSTEM"])) / 4
    if cur and tot + est > CAP: parts.append(cur); cur, tot = [], 0
    cur.append(it); tot += est
parts.append(cur); print("parts:", [len(p) for p in parts], flush=True)
st = json.load(open(SP)) if os.path.exists(SP) else {"batches": []}
for i, part in enumerate(parts):
    if i < len(st["batches"]): bid = st["batches"][i]
    else:
        jl = os.path.join(J, f"openai_batch_input_full_part{i + 1}.jsonl")
        with open(jl, "w", encoding="utf-8") as f:
            for cid, _, _, pol, q, resp in part: f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": g["openai_body"](pol, q, resp)}) + "\n")
        fo = oc.files.create(file=open(jl, "rb"), purpose="batch")
        b = oc.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"study": f"full-policy judge baseline part {i + 1}"})
        bid = b.id; st["batches"].append(bid); json.dump(st, open(SP, "w"), indent=2); print(f"part {i + 1}: submitted ({len(part)} requests)", flush=True)
    while True:
        b = oc.batches.retrieve(bid); print(f"part {i + 1}: {b.status} {b.request_counts.completed}/{b.request_counts.total}", flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"): break
        time.sleep(90)
    if b.status != "completed":
        print("STOP:", b.status, [e.code for e in (b.errors.data if b.errors else [])], flush=True); sys.exit(1)
print("ALL OPENAI PARTS COMPLETED", flush=True)
