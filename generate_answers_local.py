"""
Generate RAG answers with the two pre-registered open-weight models (EXTENSION_PREREGISTRATION.md).

    python generate_answers_local.py llama
    python generate_answers_local.py gemma

Same frozen questions, frozen retrieved chunks and prompt as generate_answers.py. Writes
answers/answers_<model>.jsonl in the same record format. Resumes after a crash (same seeds); never regenerates
an answer that already exists.
"""
import csv, datetime, hashlib, json, os, sys, winreg
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_retrieval import Retriever, INST_CODE, RAG_SYSTEM_PROMPT, build_user_prompt, clause_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv")
KEY_SHA = "689571cb91e110968b02db51c5223434cbea2719790934c10d7fea928c494420"
PREREG = os.path.join(HERE, "EXTENSION_PREREGISTRATION.md")
AMEND = os.path.join(HERE, "EXTENSION_AMENDMENT_1.md")
MODELS = {"llama": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95", True),
          "gemma": ("google/gemma-2-2b-it", "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8", False)}  # (id, revision, system role?)
TEMPERATURE, TOP_P, MAX_NEW, SEED0 = 0.7, 1.0, 512, 2026

if not os.environ.get("HF_TOKEN"):  # set with setx after this session started: read from the user environment, never printed
    os.environ["HF_TOKEN"] = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "HF_TOKEN")[0]
mid, rev, has_system = MODELS[sys.argv[1]]
name = mid.split("/")[1]
assert hashlib.sha256(open(KEY, "rb").read()).hexdigest() == KEY_SHA, "answer key changed since freezing"
prereg_sha = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
out = os.path.join(HERE, "answers", f"answers_{name}.jsonl")
done = {json.loads(l)["qid"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()

torch.set_num_threads(os.cpu_count())
tok = AutoTokenizer.from_pretrained(mid, revision=rev)
model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.bfloat16).eval()


def generate(user, seed):
    msgs = ([{"role": "system", "content": RAG_SYSTEM_PROMPT}, {"role": "user", "content": user}] if has_system
            else [{"role": "user", "content": RAG_SYSTEM_PROMPT + "\n\n" + user}])
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        o = model.generate(**ids, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=0,
                           max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
    new = o[0, ids["input_ids"].shape[1]:]
    stop = "length" if len(new) >= MAX_NEW and new[-1].item() not in (tok.eos_token_id,) else "stop"
    return tok.decode(new, skip_special_tokens=True).strip(), stop, ids["input_ids"].shape[1], len(new)


R = Retriever(); by_id = {c["chunk_id"]: c for c in R.chunks}
frozen = {}
for l in open(os.path.join(HERE, "index", "retrieval_frozen.jsonl"), encoding="utf-8"):
    x = json.loads(l); frozen[x["qid"]] = [dict(by_id[cid], sim=s) for cid, s in zip(x["chunk_ids"], x["sims"])]
key = list(csv.DictReader(open(KEY, encoding="utf-8")))
print(f"{mid}@{rev[:8]}: {len(done)} already done, {len(key) - len(done)} to go", flush=True)
with open(out, "a", encoding="utf-8") as f:
    for i, row in enumerate(key):
        if row["qid"] in done: continue
        ret = frozen[row["qid"]]; t0 = datetime.datetime.now()
        ans, stop, a, b = generate(build_user_prompt(row["question"], ret), SEED0 + i)
        cov = [clause_coverage(row[c], ret) for c in ("clause_annotator_A", "clause_annotator_B")]
        rec = {"qid": row["qid"], "question": row["question"], "institution": row["institution"],
               "retrieval_filter": INST_CODE.get(row["institution"]),
               "retrieved": [{k: c[k] for k in ("chunk_id", "doc", "institution", "sim", "text")} for c in ret],
               "retrieved_institutions": [c["institution"] for c in ret],
               "clause_coverage_auto": max([x for x in cov if x is not None], default=None),
               "answer": ans, "generator": name, "served_model": f"{mid}@{rev}", "temperature": TEMPERATURE,
               "top_p": TOP_P, "seed": SEED0 + i, "stop_reason": stop, "input_tokens": a, "output_tokens": b,
               "preregistration_sha256": prereg_sha, "amendment1_sha256": hashlib.sha256(open(AMEND, "rb").read()).hexdigest(), "dtype": "bfloat16", "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"  {row['qid']} ({i + 1}/{len(key)}) {b} tok, {(datetime.datetime.now() - t0).seconds}s, stop={stop}", flush=True)
print("done ->", out)
