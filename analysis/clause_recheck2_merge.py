"""Merge annotator 3's blind clause re-check (94 in-scope questions) with the round-1 clause labels.
Checks every quote verbatim against the frozen retrieved text, computes agreement, lists disagreements, and re-tabulates violations by clause presence."""
import csv, json, os, re, shutil
from collections import Counter
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = r"<downloads>\recheck2.csv results.csv"
DST = os.path.join(HERE, "labelling", "clause_recheck2", "clause_recheck2_annotator3_returned.csv")
if os.path.exists(SRC) and not os.path.exists(DST): shutil.copy(SRC, DST)
norm = lambda t: re.sub(r"\s+", " ", t.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').lower()).strip()
key = {r["row_id"]: r["qid"] for r in csv.DictReader(open(os.path.join(HERE, "labelling", "clause_recheck2", "CLAUSE_RECHECK2_KEY_do_not_share.csv"), encoding="utf-8"))}
ret = {json.loads(l)["qid"]: json.loads(l)["chunk_ids"] for l in open(os.path.join(HERE, "index", "retrieval_frozen.jsonl"), encoding="utf-8")}
ch = {json.loads(l)["chunk_id"]: json.loads(l)["text"] for l in open(os.path.join(HERE, "index", "chunks.jsonl"), encoding="utf-8")}
lab = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in csv.DictReader(open(os.path.join(HERE, f), encoding="utf-8")): lab.setdefault(r["qid"], []).append(r)
rows = list(csv.DictReader(open(DST, encoding="utf-8")))
assert len(rows) == 94 and {r["Row ID"] for r in rows} == set(key)
a3, bad_q = {}, []
for r in rows:
    q = key[r["Row ID"]]; a3[q] = r
    if r["Clause present"] in ("YES", "PARTIAL"):
        ctx = norm(" ".join(ch[c] for c in ret[q]))
        if norm(r["Quote from retrieved text"]) not in ctx: bad_q.append((q, r["Quote from retrieved text"][:80]))
    elif r["Quote from retrieved text"].strip(): bad_q.append((q, "quote given for NO"))
print("annotator 3 labels:", dict(Counter(r["Clause present"] for r in rows)))
print("quotes failing verbatim check:", bad_q)
qs = sorted(a3); r1 = {q: lab[q][0]["clause_in_retrieved"] for q in qs}
assert all(len({x["clause_in_retrieved"] for x in lab[q]}) == 1 for q in qs)
print("round-1 labels on same questions:", dict(Counter(r1.values())))
y1 = [r1[q] for q in qs]; y3 = [a3[q]["Clause present"] for q in qs]
print(f"\nAGREEMENT (3 categories): raw {sum(a == b for a, b in zip(y1, y3))}/94 = {sum(a == b for a, b in zip(y1, y3)) / 94:.1%}, kappa {cohen_kappa_score(y1, y3):.2f}")
b1 = [x == "YES" for x in y1]; b3 = [x == "YES" for x in y3]
print(f"AGREEMENT (YES vs not-YES): raw {sum(a == b for a, b in zip(b1, b3))}/94 = {sum(a == b for a, b in zip(b1, b3)) / 94:.1%}, kappa {cohen_kappa_score(b1, b3):.2f}")
print("confusion (round1 -> annotator3):", dict(Counter(zip(y1, y3))))
cand = {"C06", "C07", "D14", "H07", "H08", "T01", "T06", "U04", "U10", "U11", "U14", "V09"}
print("\nthe 12 automated false-NO candidates -> annotator 3:", {q: a3[q]["Clause present"] for q in sorted(cand)})
print("\nDISAGREEMENTS (round1 vs annotator3):")
dis = [q for q in qs if r1[q] != a3[q]["Clause present"]]
for q in dis:
    print(f"  {q}: round1={r1[q]:7} A3={a3[q]['Clause present']:7} | A3 quote/note: {(a3[q]['Quote from retrieved text'] or a3[q]['Notes'])[:90]!r} | r1 quote: {lab[q][0]['clause_quote'][:70]!r}")
with open(os.path.join(HERE, "results", "clause_recheck2_comparison.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["qid", "round1", "annotator3", "agree", "violations_orig_models", "violations_added_models"])
    for q in qs:
        v = lambda ms: sum(int(x["y_violation"]) for x in lab[q] if x["answer_model"] in ms)
        w.writerow([q, r1[q], a3[q]["Clause present"], r1[q] == a3[q]["Clause present"], v(("claude-haiku-4-5", "gpt-4o-mini")), v(("Llama-3.2-3B-Instruct", "gemma-2-2b-it"))])
ORIG, EXT = ("claude-haiku-4-5", "gpt-4o-mini"), ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")
print("\nVIOLATIONS BY CLAUSE PRESENCE (answers | violations orig models | violations added models):")
for name, get in (("round-1 labels", lambda q: r1[q]), ("annotator-3 labels", lambda q: a3[q]["Clause present"])):
    print(" ", name)
    for c in ("YES", "PARTIAL", "NO"):
        s = [q for q in qs if get(q) == c]
        print(f"    {c:8} questions {len(s):2d} | answers {4 * len(s):3d} | orig {sum(int(x['y_violation']) for q in s for x in lab[q] if x['answer_model'] in ORIG):2d} | added {sum(int(x['y_violation']) for q in s for x in lab[q] if x['answer_model'] in EXT):2d}")
