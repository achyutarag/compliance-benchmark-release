"""
Generate RAG answers for the 104 frozen questions with the frozen retrieval configuration.

    python generate_answers.py claude     # claude-haiku-4-5, temperature 0.7 (run locally)
    python generate_answers.py openai     # gpt-4o-mini,     temperature 0.7 (run in Colab)

Writes answers/answers_<model>.jsonl, one record per question with the exact retrieved chunks
(ids, similarity, text), the answer, stop reason and token usage. Refuses to overwrite.
"""
import csv, datetime, hashlib, json, os, sys
from rag_retrieval import Retriever, INST_CODE, RAG_SYSTEM_PROMPT, build_user_prompt, clause_coverage

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "answer_key_FROZEN_2026-09-30.csv")
KEY_SHA = "689571cb91e110968b02db51c5223434cbea2719790934c10d7fea928c494420"
TEMPERATURE = 0.7                       # same as the pilot
MODELS = {"claude": "claude-haiku-4-5", "openai": "gpt-4o-mini"}

which = sys.argv[1]
model = MODELS[which]
assert hashlib.sha256(open(KEY, "rb").read()).hexdigest() == KEY_SHA, "answer key changed since freezing"
os.makedirs(os.path.join(HERE, "answers"), exist_ok=True)
out = os.path.join(HERE, "answers", f"answers_{model}.jsonl")
assert not os.path.exists(out), f"{out} exists; refusing to overwrite"

if which == "claude":
    import anthropic
    client = anthropic.Anthropic()
    def generate(user):
        r = client.messages.create(model=model, max_tokens=2048, system=RAG_SYSTEM_PROMPT,
                                   messages=[{"role": "user", "content": user}],
                                   extra_body={"temperature": TEMPERATURE})  # SDK 1.x: sampling params via extra_body
        return ("".join(b.text for b in r.content if b.type == "text").strip(), r.stop_reason,
                r.usage.input_tokens, r.usage.output_tokens, r.model)
else:
    from openai import OpenAI
    client = OpenAI()
    def generate(user):
        r = client.chat.completions.create(model=model, temperature=TEMPERATURE,
                                           messages=[{"role": "system", "content": RAG_SYSTEM_PROMPT},
                                                     {"role": "user", "content": user}])
        return (r.choices[0].message.content.strip(), r.choices[0].finish_reason,
                r.usage.prompt_tokens, r.usage.completion_tokens, r.model)

R = Retriever()
key = list(csv.DictReader(open(KEY, encoding="utf-8")))
# Reuse the exact chunks fixed from the first run, so every answer model sees identical context.
FROZEN = os.path.join(HERE, "index", "retrieval_frozen.jsonl")
frozen = {}
if os.path.exists(FROZEN):
    by_id = {c["chunk_id"]: c for c in R.chunks}
    for l in open(FROZEN, encoding="utf-8"):
        x = json.loads(l)
        frozen[x["qid"]] = [dict(by_id[cid], sim=s) for cid, s in zip(x["chunk_ids"], x["sims"])]
    print(f"using frozen retrieval for {len(frozen)} questions")
tin = tout = 0
with open(out, "w", encoding="utf-8") as f:
    for i, row in enumerate(key, 1):
        inst = INST_CODE.get(row["institution"])              # None for unnamed (N) questions
        ret = frozen[row["qid"]] if frozen else R.retrieve(row["question"], inst)
        if frozen:  # sanity check: recomputed retrieval should match; report (not fail) on near-tie differences
            live = [c["chunk_id"] for c in R.retrieve(row["question"], inst)]
            if live != [c["chunk_id"] for c in ret]:
                print(f"  [note] {row['qid']}: live retrieval differs from frozen ({live}); frozen chunks used")
        ans, stop, a, b, served = generate(build_user_prompt(row["question"], ret))
        tin += a; tout += b
        cov = [clause_coverage(row[c], ret) for c in ("clause_annotator_A", "clause_annotator_B")]
        rec = {"qid": row["qid"], "question": row["question"], "institution": row["institution"],
               "retrieval_filter": inst, "retrieved": [{k: c[k] for k in ("chunk_id", "doc", "institution", "sim", "text")} for c in ret],
               "retrieved_institutions": [c["institution"] for c in ret],
               "clause_coverage_auto": max([x for x in cov if x is not None], default=None),
               "answer": ans, "generator": model, "served_model": served, "temperature": TEMPERATURE,
               "stop_reason": stop, "input_tokens": a, "output_tokens": b,
               "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if stop not in ("end_turn", "stop"):
            print(f"  [warn] {row['qid']}: stop_reason={stop}")
        if i % 20 == 0:
            print(f"  {i}/{len(key)}")
print(f"done: {len(key)} answers -> {out} | tokens in/out {tin}/{tout}")
