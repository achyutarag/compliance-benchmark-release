"""Regenerates the paper's reported counts from the frozen labels and stored auditor outputs and compares them with the values written in the manuscript.
Prints one line per number (computed vs paper) and exits non-zero if any differs. No API calls.   python analysis/reproduce_paper_numbers.py"""
import contextlib, csv, io, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.dirname(HERE)
src = open(os.path.join(HERE, "audit_tables.py"), encoding="utf-8").read(); src = src[:src.index("for cond, keys in")]
g = {"__name__": "audit_header", "__file__": os.path.join(HERE, "audit_tables.py")}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "audit_header", "exec"), g)
Y, pred, oracle_keys, rd = g["Y"], g["pred"], g["oracle_keys"], g["rd"]
rows, bad = [], []
def check(name, computed, paper):
    ok = computed == paper; rows.append(ok)
    if not ok: bad.append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:78} computed {computed!s:24} paper {paper!s}")
L = {}
for f in ("labels_FROZEN_2026-09-30.csv", "labels_ext_FROZEN_2026-10-01.csv"):
    for r in rd(f): L[(r["qid"], r["answer_model"])] = r
V2 = {r["qid"]: r["clause_in_retrieved_v2"] for r in rd("clause_labels_v2_FROZEN_2026-10-11.csv")}
A = {(r["qid"], r["answer_model"]): r for r in rd("results/all416_per_answer.csv")}
STRONG = ("claude-haiku-4-5", "gpt-4o-mini"); SMALL = ("Llama-3.2-3B-Instruct", "gemma-2-2b-it")
viol = [k for k in Y if Y[k]]
# --- dataset
check("answers", len(Y), 416); check("violations", len(viol), 79)
for m, n in (("claude-haiku-4-5", 14), ("gpt-4o-mini", 14), ("Llama-3.2-3B-Instruct", 22), ("gemma-2-2b-it", 29)): check(f"violations, {m}", sum(Y[k] for k in Y if k[1] == m), n)
check("violations from the two small models", sum(Y[k] for k in Y if k[1] in SMALL), 51)
check("oracle subset answers / violations", (len(oracle_keys), sum(Y[k] for k in oracle_keys)), (376, 69))
fp = {}
for r in rd("judges/full_predictions.csv"): fp.setdefault((r["qid"], r["answer_model"]), r)
check("full-policy subset answers / violations", (len(fp), sum(Y[k] for k in fp)), (396, 72))
# --- clause table (Table clause) and RQ3 text
tab = defaultdict(lambda: [0, 0, 0])
for k in Y:
    c = V2[k[0]] if V2[k[0]] in ("YES", "PARTIAL", "NO") else "N/A"; tab[c][0] += 1
    if Y[k]: tab[c][1 if k[1] in STRONG else 2] += 1
for c, exp in (("YES", [224, 4, 21]), ("PARTIAL", [24, 2, 7]), ("NO", [128, 16, 19]), ("N/A", [40, 6, 4])): check(f"clause table {c}: answers, strong viol, small viol", tab[c], exp)
check("in-scope questions YES / PARTIAL / NO", tuple(sum(1 for q, c in V2.items() if c == t) for t in ("YES", "PARTIAL", "NO")), (56, 6, 32))
check("stronger models in-scope violations (total, clause absent)", (tab["YES"][1] + tab["PARTIAL"][1] + tab["NO"][1], tab["NO"][1]), (22, 16))
check("small models in-scope violations (total, clause fully present)", (tab["YES"][2] + tab["PARTIAL"][2] + tab["NO"][2], tab["YES"][2]), (47, 21))
summ = open(os.path.join(H, "results", "retrieval_misses_v2_summary.txt"), encoding="utf-8").read()
check("retrieval-miss summary mentions median rank 11", bool(re.search(r"median[^\n]*\b11\b", summ)), True)
fpr = lambda col, c: (lambda ks: (sum(int(A[k][col]) for k in ks), len(ks)))([k for k in A if not Y[k] and V2[k[0]] == c])
for nm, col, exp in (("GPT-5.5", "gpt55_retrieved", ((43, 199), (48, 93))), ("Sonnet 5.5", "sonnet55_retrieved", ((60, 199), (50, 93)))):
    check(f"{nm} flags on compliant answers, clause YES / NO", (fpr(col, "YES"), fpr(col, "NO")), exp)
# --- reasoning structures
S = {k: set(L[k]["structures"].split(";")) for k in Y}
def grp(ss): return [k for k in viol if S[k] & ss]
g1 = grp({"S7", "S8", "S10"}); g5 = grp({"S5"}); g6 = grp({"S6"})
Q = {}
for r in rd("judges/q_predictions.csv"):
    if r["condition"] == "retrieved": Q[(r["provider"], r["qid"], r["answer_model"])] = int(r["pred_violation"])
cnt = lambda ks, col: sum(int(A[k][col]) for k in ks)
check("S7/S8/S10 (multi-condition, procedural, conditional) violations", len(g1), 20)
check("  naive / MSCG / GPT-5.5 / Sonnet detections", (cnt(g1, "naive"), cnt(g1, "mscg"), cnt(g1, "gpt55_retrieved"), cnt(g1, "sonnet55_retrieved")), (0, 2, 11, 15))
check("S5 (category ban) violations", len(g5), 16)
check("  MSCG detections; judges (GPT, Sonnet)", (cnt(g5, "mscg"), cnt(g5, "gpt55_retrieved"), cnt(g5, "sonnet55_retrieved")), (7, 14, 15))
check("S6 (role-conditioned) violations", len(g6), 14)
check("  GPT-5.5 / Sonnet 5.5 / MSCG detections", (cnt(g6, "gpt55_retrieved"), cnt(g6, "sonnet55_retrieved"), cnt(g6, "mscg")), (2, 4, 6))
check("  with the question: GPT-5.5 / Sonnet 5.5", (sum(Q[("openai", *k)] for k in g6), sum(Q[("anthropic", *k)] for k in g6)), (7, 8))
# --- judge family, agreement
rec = lambda col, ms: (sum(int(A[k][col]) for k in viol if k[1] in ms), sum(1 for k in viol if k[1] in ms))
check("GPT-5.5 recall on Haiku / gpt-4o-mini answers", (rec("gpt55_retrieved", ("claude-haiku-4-5",)), rec("gpt55_retrieved", ("gpt-4o-mini",))), ((5, 14), (7, 14)))
check("Sonnet 5.5 recall on Haiku / gpt-4o-mini answers", (rec("sonnet55_retrieved", ("claude-haiku-4-5",)), rec("sonnet55_retrieved", ("gpt-4o-mini",))), ((7, 14), (4, 14)))
check("open-weight recall, GPT-5.5 / Sonnet 5.5", (rec("gpt55_retrieved", SMALL), rec("sonnet55_retrieved", SMALL)), ((37, 51), (43, 51)))
a = np.array([int(A[k]["gpt55_retrieved"]) for k in A]); b = np.array([int(A[k]["sonnet55_retrieved"]) for k in A])
po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); kap = (po - pe) / (1 - pe)
check("judge agreement (raw %, kappa), retrieved context", (round(100 * po, 1), round(kap, 2)), (80.5, 0.60))
# --- quoting and support
ans, ret = {}, {}
for m in STRONG + SMALL:
    for r in map(json.loads, open(os.path.join(H, "answers", f"answers_{m}.jsonl"), encoding="utf-8")): ans[(r["qid"], m)] = r["answer"]; ret[(r["qid"], m)] = " ".join(c["text"] for c in r["retrieved"])
g5w = lambda t: {tuple(w[i:i + 5]) for w in [re.findall(r"[a-z0-9$]+", t.lower())] for i in range(len(w) - 4)}
has = {k: int(bool(g5w(ans[k]) & g5w(ret[k]))) for k in Y}
check("quoting rate, violating vs compliant (%)", (round(100 * np.mean([has[k] for k in Y if Y[k]])), round(100 * np.mean([has[k] for k in Y if not Y[k]]))), (63, 69))
MC = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding", "scores_minicheck_whole.jsonl"), encoding="utf-8")}
check("MiniCheck median support, violating vs compliant", (round(float(np.median([MC[(k[0], k[1], "retrieved")] for k in Y if Y[k]])), 3), round(float(np.median([MC[(k[0], k[1], "retrieved")] for k in Y if not Y[k]])), 3)), (0.075, 0.051))
# --- failure analysis
fc = list(csv.DictReader(open(os.path.join(HERE, "failure_codes_llm_vs_human.csv"), encoding="utf-8"))) if os.path.exists(os.path.join(HERE, "failure_codes_llm_vs_human.csv")) else []
if fc:
    cc = Counter(r["llm_code"] for r in fc)
    check("38 failures by LLM code (prohibition, condition, boundary, role)", (cc["OVERRIDDEN_PROHIBITION"], cc["MISAPPLIED_CONDITION"], cc["EXACT_BOUNDARY"], cc["ROLE_SPECIFIC"]), (16, 15, 5, 2))
    mg = lambda c: "PC" if c in ("OVERRIDDEN_PROHIBITION", "MISAPPLIED_CONDITION") else c
    check("human agreement on the 38 (exact, merged prohibition+condition)", (sum(r["llm_code"] == r["annotatorA_code_blind"] for r in fc), sum(mg(r["llm_code"]) == mg(r["annotatorA_code_blind"]) for r in fc)), (34, 37))
HHs = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding", "scores_hhem_whole.jsonl"), encoding="utf-8")}
NLs = {tuple(json.loads(l)["id"].split("|")): json.loads(l) for l in open(os.path.join(H, "grounding", "scores_nli_large_whole.jsonl"), encoding="utf-8")}
sup3 = [r for r in fc if MC[(r["qid"], r["answer_model"], "oracle")] >= 0.5 and HHs[(r["qid"], r["answer_model"], "oracle")] >= 0.5 and NLs[(r["qid"], r["answer_model"], "oracle")]["E"] >= 0.5]
check("coded failures scored supported by MiniCheck, HHEM and DeBERTa-large together (oracle)", len(sup3), 4)
d09 = (MC[("D09", "gemma-2-2b-it", "oracle")], HHs[("D09", "gemma-2-2b-it", "oracle")], NLs[("D09", "gemma-2-2b-it", "oracle")]["E"], NLs[("D09", "gemma-2-2b-it", "oracle")]["C"])
check("Figure 1 item D09/gemma: MiniCheck, HHEM, DeBERTa entailment, contradiction", tuple(round(x, 2) for x in d09), (0.69, 0.87, 0.82, 0.0))
check("Figure 1 item D09/gemma: MSCG, naive NLI, GPT-5.5, Sonnet 5.5 (oracle)", (int(A[("D09", "gemma-2-2b-it")]["mscg_oracle"]), int(A[("D09", "gemma-2-2b-it")]["naive"]), int(A[("D09", "gemma-2-2b-it")]["gpt55_oracle"]), int(A[("D09", "gemma-2-2b-it")]["sonnet55_oracle"])), (0, 0, 1, 1))
# --- grounding-check AUROC restricted to the stronger answer models (Haiku, gpt-4o-mini)
from sklearn.metrics import roc_auc_score as _auc
_sc = [(nm, sn) for nm, pn, sn in g["NAMES"] if sn]
def _au(cond, ms, nm, sn):
    ks = [k for k in (sorted(Y) if cond == "retrieved" else oracle_keys) if k[1] in ms]
    s_ = [1 - HHs[(k[0], k[1], cond)] if nm == "HHEM" else g["score"](sn, cond, k) for k in ks]; return _auc([Y[k] for k in ks], s_)
for cond, exp in (("retrieved", (0.42, 0.61)), ("oracle", (0.43, 0.66))):
    v = [_au(cond, STRONG, nm, sn) for nm, sn in _sc] + [_au(cond, STRONG, "HHEM", None)]
    check(f"AUROC range of the eight scored checkers, stronger models only, {cond}", (round(min(v), 2), round(max(v), 2)), exp)
for nm in ("GPT-5.5", "Sonnet 5.5"):
    pn = {a: b for a, b, c in g["NAMES"]}[nm]
    def _j(cond, ms):
        ks = [k for k in (sorted(Y) if cond == "retrieved" else oracle_keys) if k[1] in ms]; y = np.array([Y[k] for k in ks]); p = np.array([pred(pn, cond, k) for k in ks])
        return round((p & y).sum() / y.sum() - (p & (1 - y)).sum() / (1 - y).sum(), 2)
    check(f"{nm} informedness, stronger / small models, retrieved", (_j("retrieved", STRONG), _j("retrieved", SMALL)), {"GPT-5.5": (0.22, 0.31), "Sonnet 5.5": (0.10, 0.37)}[nm])
# --- two-judge combinations
for cond, exp in (("retrieved", {"AND": (0.37, 58, 0.68), "OR": (0.28, 72, 0.68)}), ("oracle", {"AND": (0.34, 91, 0.76), "OR": (0.27, 93, 0.76)})):
    ks = [k for k in (sorted(Y) if cond == "retrieved" else oracle_keys)]; y = np.array([Y[k] for k in ks])
    gg = np.array([int(A[k][f"gpt55_{cond}"]) for k in ks]); ss = np.array([int(A[k][f"sonnet55_{cond}"]) for k in ks])
    for nm, pp in (("AND", gg & ss), ("OR", gg | ss)):
        check(f"two judges {nm}, {cond}: precision, recall %", (round((pp & y).sum() / pp.sum(), 2), round(100 * (pp & y).sum() / y.sum())), exp[nm][:2])
    check(f"two-judge vote count AUROC, {cond}", round(roc_auc_score(y, gg + ss), 2), exp["AND"][2])
# --- question-plus-answer hypothesis control (grounding/GROUNDING_PROTOCOL.md, Amendment 4)
_ld = lambda f: {tuple(json.loads(l)["id"].split("|")): json.loads(l) for l in open(os.path.join(H, "grounding", f), encoding="utf-8")}
QA = {"MiniCheck": _ld("scores_minicheck_qa.jsonl"), "HHEM": _ld("scores_hhem_qa.jsonl"), "DeBERTa": _ld("scores_nli_large_qa.jsonl")}
def _qa(nm, cond, rule):
    ks = sorted(Y) if cond == "retrieved" else oracle_keys; y = [Y[k] for k in ks]; D = QA[nm]
    if nm == "DeBERTa": sc_ = [D[(k[0], k[1], cond)]["C"] if rule == "C" else -D[(k[0], k[1], cond)]["E"] for k in ks]
    else: sc_ = [-D[(k[0], k[1], cond)]["support"] for k in ks]
    return round(_auc(y, sc_), 2)
check("question+answer AUROC, retrieved: MiniCheck, HHEM, DeBERTa contradiction, DeBERTa not entailed", (_qa("MiniCheck", "retrieved", 0), _qa("HHEM", "retrieved", 0), _qa("DeBERTa", "retrieved", "C"), _qa("DeBERTa", "retrieved", "E")), (0.48, 0.46, 0.55, 0.52))
check("question+answer AUROC, oracle: MiniCheck, HHEM, DeBERTa contradiction, DeBERTa not entailed", (_qa("MiniCheck", "oracle", 0), _qa("HHEM", "oracle", 0), _qa("DeBERTa", "oracle", "C"), _qa("DeBERTa", "oracle", "E")), (0.52, 0.63, 0.55, 0.64))
_fpr = lambda nm, cond: round(100 * np.mean([int(QA[nm][(k[0], k[1], cond)]["support"] < 0.5) for k in (sorted(Y) if cond == "retrieved" else oracle_keys) if not Y[k]]))
check("question+answer FPR %, MiniCheck retrieved / oracle; HHEM retrieved / oracle", (_fpr("MiniCheck", "retrieved"), _fpr("MiniCheck", "oracle"), _fpr("HHEM", "retrieved"), _fpr("HHEM", "oracle")), (98, 99, 47, 60))
# --- "two mechanisms" paragraph: strict support-based checkers flag 70-99% of compliant answers, AUROC 0.44-0.50 (retrieved context)
_ks = sorted(Y); _y = np.array([Y[k] for k in _ks]); _fp = {}; _au_ = {}
for _nm, _pn, _sn in g["NAMES"]:
    if _nm in ("MiniCheck", "DNA", "RoBERTa not entailed"):
        _p = np.array([pred(_pn, "retrieved", k) for k in _ks]); _fp[_nm] = round(100 * (_p & (1 - _y)).sum() / (1 - _y).sum()); _au_[_nm] = round(_auc(_y, [g["score"](_sn, "retrieved", k) for k in _ks]), 2)
_hp = np.array([int(HHs[(k[0], k[1], "retrieved")] < 0.5) for k in _ks]); _fp["HHEM"] = round(100 * (_hp & (1 - _y)).sum() / (1 - _y).sum()); _au_["HHEM"] = round(_auc(_y, [1 - HHs[(k[0], k[1], "retrieved")] for k in _ks]), 2)
check("support-based checkers' flag rate on compliant answers, retrieved: min-max %", (min(_fp.values()), max(_fp.values())), (70, 99))
check("support-based checkers' AUROC, retrieved: min-max", (min(_au_.values()), max(_au_.values())), (0.44, 0.50))
# --- oracle swap test: judge flag rate on compliant answers, by overlap of retrieved and oracle chunks
_orc = {r["qid"]: set(r["oracle_chunks"].split("|")) for r in rd("results/mscg_oracle_context_diagnostic.csv")}
_ret = {}
for _m in STRONG + SMALL:
    for _r in map(json.loads, open(os.path.join(H, "answers", f"answers_{_m}.jsonl"), encoding="utf-8")): _ret[(_r["qid"], _m)] = {c["chunk_id"] for c in _r["retrieved"]}
_kc = [k for k in sorted(Y) if k[0] in _orc and not Y[k]]
_same = [k for k in _kc if _ret[k] == _orc[k[0]]]; _none = [k for k in _kc if not (_ret[k] & _orc[k[0]])]
check("oracle swap test: compliant answers with retrieved = oracle chunks / sharing no chunk", (len(_same), len(_none)), (39, 93))
_rate = lambda grp, col: round(100 * np.mean([int(A[k][col]) for k in grp]))
for _j, _e in (("gpt55", ((21, 13), (49, 86))), ("sonnet55", ((23, 23), (54, 83)))):
    check(f"oracle swap test, {_j}: flag % retrieved->oracle, equal chunks / no shared chunk", ((_rate(_same, f"{_j}_retrieved"), _rate(_same, f"{_j}_oracle")), (_rate(_none, f"{_j}_retrieved"), _rate(_none, f"{_j}_oracle"))), _e)
et = Counter(L[k]["error_type"] for k in viol)
check("error types of the 79 violations (wrong verdict, failed refusal, invented rule)", (et["WRONG_VERDICT"], et["FAILED_REFUSAL"], et["INVENTED_RULE"]), (74, 3, 2))
# --- agreement statistics recorded in the manifests
m1 = json.load(open(os.path.join(H, "labels_FROZEN_MANIFEST.json"))); m2 = json.load(open(os.path.join(H, "labels_ext_FROZEN_MANIFEST.json"))); mk = json.load(open(os.path.join(H, "answer_key_FROZEN_MANIFEST.json"))); mc = json.load(open(os.path.join(H, "clause_labels_v2_FROZEN_MANIFEST.json")))
check("answer key kappa (94 in-scope)", mk["inter_annotator_agreement"]["core_excl_N_O"]["cohen_kappa"], 0.82)
check("round 2 compliance raw / kappa / PABAK", (m2["pre_reconciliation_agreement"]["compliance_raw"], m2["pre_reconciliation_agreement"]["compliance_kappa"], m2["pre_reconciliation_agreement"]["compliance_pabak"]), (0.812, 0.44, 0.62))
check("clause labels kappa three-way / yes-vs-not", (mc["agreement_before_adjudication"]["three_category_kappa"], mc["agreement_before_adjudication"]["yes_vs_not_yes_kappa"]), (0.57, 0.69))
# --- Table 1 (main) against the manuscript's own table cells (only if the .tex path is given: python analysis/reproduce_paper_numbers.py path/to/main.tex)
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    from sklearn.metrics import roc_auc_score
    tex = open(sys.argv[1], encoding="utf-8").read(); tb = tex[tex.index(r"\label{tab:main}"):]; tb = tb[:tb.index(r"\end{tabular}")]
    HH = {tuple(json.loads(l)["id"].split("|")): json.loads(l)["support"] for l in open(os.path.join(H, "grounding", "scores_hhem_whole.jsonl"), encoding="utf-8")}
    NAMES = g["NAMES"]; score = g["score"]
    spec = {"MiniCheck": "MiniCheck", "HHEM": "HHEM", "DeBERTa-v3-large, contradiction": "DeBERTa-L contradiction", "DeBERTa-v3-large, not entailed": "DeBERTa-L not entailed",
            "RoBERTa-large-MNLI, contradiction": "RoBERTa contradiction", "RoBERTa-large-MNLI, not entailed": "RoBERTa not entailed", r"\mscg": "MSCG", "DNA": "DNA", "GPT-5.5": "GPT-5.5", "Claude Sonnet 5.5": "Sonnet 5.5"}
    pn = {nm: (p, sn) for nm, p, sn in NAMES}
    def stats(nm, cond):
        ks = sorted(Y) if cond == "retrieved" else oracle_keys; y = np.array([Y[k] for k in ks])
        if nm == "HHEM":
            p = np.array([int(HH[(k[0], k[1], cond)] < 0.5) for k in ks]); au = roc_auc_score(y, [1 - HH[(k[0], k[1], cond)] for k in ks])
        else:
            pnm, sn = pn[nm]; p = np.array([pred(pnm, cond, k) for k in ks]); au = roc_auc_score(y, [score(sn, cond, k) for k in ks]) if sn else roc_auc_score(y, p)   # judges: AUROC of the binary verdict
        return int((p & y).sum()), int(y.sum()), round(100 * (p & (1 - y)).sum() / (1 - y).sum()), (None if au is None else round(au, 2))
    seen = 0
    for line in tb.split("\n"):
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) != 7: continue
        key = next((v for k, v in spec.items() if cells[0].replace("$", "").startswith(k.replace("$", ""))), None)
        if key is None: continue
        seen += 1
        for off, cond in ((1, "retrieved"), (4, "oracle")):
            m = re.match(r"(\d+)/(\d+)", cells[off]); fp_ = int(re.match(r"(\d+)", cells[off + 1]).group(1)); au = re.match(r"(0\.\d\d|--)", cells[off + 2]).group(1)
            check(f"Table 1 {key}, {cond}: recall, FPR%, AUROC", stats(key, cond), (int(m.group(1)), int(m.group(2)), fp_, None if au == "--" else float(au)))
    check("Table 1 rows found in the manuscript", seen, 10)
print(f"\n{sum(rows)}/{len(rows)} numbers reproduce" + ("" if not bad else f"; MISMATCHES: {bad}")); sys.exit(1 if bad else 0)
