"""
Remove non-policy text (site navigation, print headers/footers, page labels, link URLs)
from the extracted policy texts. Policy wording is never changed.

  policies/raw_extracted/<name>.txt   the uncleaned extraction (kept for comparison)
  policies/<name>.txt                 cleaned text used for the RAG corpus
  policies/CLEANING_LOG.md            every removed line, per file
  policies/MANIFEST.json              updated text checksums
"""
import glob, hashlib, json, os, re, shutil
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root; P = os.path.join(HERE, "policies")
RAW = os.path.join(P, "raw_extracted"); os.makedirs(RAW, exist_ok=True)
SEP = "=" * 70 + "\n"
PRINT_HEADER = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2} [AP]M .*$")      # browser print header
PRINT_FOOTER = re.compile(r"^https?://\S+ \d+/\d+$")                              # browser print footer
BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd", "caption"]


def html_blocks(path):
    """Extract text block-by-block so inline links stay on the same line."""
    soup = BeautifulSoup(open(path, "rb").read(), "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "button"]):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.body
    out = []
    for el in root.find_all(BLOCKS):
        if el.find(BLOCKS):          # skip containers; their children are emitted themselves
            continue
        s = " ".join(el.get_text(" ").split())
        s = re.sub(r"\s+([).,;:])", r"\1", s).replace("( ", "(")
        if s: out.append(s)
    return out


RULES = {  # per-file lines to drop (exact or regex); applied after generic rules
    "hartwick_6-42": [r"^Effective Date$", r"^Revised Date$", r"^\[email.*protected\]$", r"^Return to Policies$"],
    "upenn_VI-D": [r"^VI\.D\. Gift Policy\s+\d+$"],
    "utk_FI0717": [r"^Downloadable PDF$", r"^Related Policies:$", r"^Related Guidance Documents:$",
                   r"^< go back to main policy page$", r"^Policy Categories$",
                   r"^(General|Board of Trustees|Fiscal|Health & Safety / Emergency Management|Human Resources|Information Technology|Research)$",
                   r"^(Subject Matter|Office Name|Telephone Number|Email/Web Address|Policy Clarification and Interpretation)$",
                   r"^Policy Details:$", r"^GE0002 – Conflicts of Interest & Commitment$", r"^FI0717 – Acceptable and Unacceptable Gifts$",
                   r"^FI0717 – Employee Gift Acceptance Policy$"],
    "utk_FI0717_guidance": [r"^Downloadable PDF$", r"^Related Policies:$", r"^Version: // Effective:$",
                            r"^Policy Categories$",
                            r"^(General|Board of Trustees|Fiscal|Health & Safety / Emergency Management|Human Resources|Information Technology|Research)$",
                            r"^FI0717 – Employee Gift Acceptance Policy$"],
    "depaul_gifts_to_employees": [r"^Page \d+ of \d+$"],
    "uthealth_HOOP-226": [r"^Handbook of Operating Procedures$", r"^Search HOOP Policies GO$"],
    "uva_FIN-019": [],
    "ma_930CMR5": [r"^\d+\.\d+:\s+continued$"],
    "ma_advisory_19-1": [r"^OFFERED BY$", r"^(I|II|III|IV|V|VI)\. [A-Z].*$",
                         r"^State Organizations$", r"^© \d{4} Commonwealth of Massachusetts\.$",
                         # site contact widget (printed twice) and a stray floating 'Contact' button mid-sentence
                         r"^(CONTACT|Contact)$", r"^Phone$", r"^Main \(617\) 371-9500$", r"^Open M-F 9am-5pm$", r"^Online$",
                         r"^Staff Contact Information$", r"^Policy prohibiting weapons in Commission( offices)?$", r"^offices$",
                         r"^Mass\.gov® is a registered service mark.*$"],
}
# contiguous site-contact blocks in the MA advisory (header and footer), removed as a unit
MA_CONTACT = re.compile(r"(?ms)^(CONTACT|Contact)\n.*?policy-?\s*\n?prohibiting-weapons-in-the-commission-offices\)?\s*$")

log, man = ["# Cleaning log\n", "Every line removed from each extracted policy text. Policy wording was not edited.\n"], \
    json.load(open(os.path.join(P, "MANIFEST.json"), encoding="utf-8"))
for d in man["documents"]:
    name = d["name"]; fp = os.path.join(P, d["text"])
    raw_copy = os.path.join(RAW, name + ".txt")
    if not os.path.exists(raw_copy):
        shutil.copy2(fp, raw_copy)
    header, body = open(raw_copy, encoding="utf-8").read().split(SEP, 1)
    if d["archive"].endswith(".html"):
        # Re-join inline fragments split by link/emphasis tags. Only JOINS lines, never drops text:
        # a line is appended to the previous one if it starts with closing punctuation or a lowercase
        # word, or if the previous line ended mid-sentence and this one is a short fragment.
        lines = []
        for l in (re.sub(r"[ \t ]+", " ", x).strip() for x in body.splitlines()):
            if not l:
                lines.append(""); continue
            prev = next((i for i in range(len(lines) - 1, -1, -1) if lines[i]), None)
            joinable = prev is not None and lines[-1] != "" and (
                re.match(r"^[).,;:]", l) or re.match(r"^[a-z§]", l)
                or (not re.search(r"[.:;!?]$", lines[prev]) and len(l.split()) <= 3 and re.search(r"[(§\-–]$|the$|and$|of$|to$", lines[prev])))
            if joinable:
                sep = "" if re.match(r"^[).,;:]", l) else " "
                lines[prev] = lines[prev] + sep + l
            else:
                lines.append(l)
    else:
        text = body.replace(" ", " ")
        text = re.sub(r"\s*\((?:/|https?://)[^)]*\)", "", text, flags=re.S)         # link URLs in parentheses
        if name == "ma_advisory_19-1":
            text = MA_CONTACT.sub("", text)
            text = re.sub(r"(?m)^REFERENCED SOURCES:\s*\nState Ethics Commission Regulations: 930 CMR 1\.00 - 7\.00\s*\n?(100-700\))?\s*$", "", text)
        lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    kept, removed, seen_running = [], [], False
    for l in lines:
        if not l:
            if kept and kept[-1] != "": kept.append("")
            continue
        drop = bool(PRINT_HEADER.match(l) or PRINT_FOOTER.match(l)) or any(re.match(rx, l) for rx in RULES[name])
        if name == "ma_930CMR5" and l.startswith("930 CMR:") and "STATE ETHICS COMMISSION" in l:
            drop, seen_running = seen_running, True          # keep first running header as title
        (removed if drop else kept).append(l)
    body_new = "\n".join(kept).strip() + "\n"
    open(fp, "w", encoding="utf-8").write(header + SEP + body_new)
    d["text_sha256"] = hashlib.sha256((header + SEP + body_new).encode("utf-8")).hexdigest()
    d["text_chars"] = len(body_new); d["lines_removed_in_cleaning"] = len(removed)
    log.append(f"\n## {name} ({len(removed)} lines removed)\n")
    log += [f"- `{r[:160]}`" for r in removed] or ["- (none)"]
    print(f"{name:26} removed {len(removed):3d} lines -> {len(body_new):6d} chars")
man["cleaning"] = "Non-policy text removed by clean_policies.py; see CLEANING_LOG.md. Uncleaned text in raw_extracted/."
json.dump(man, open(os.path.join(P, "MANIFEST.json"), "w", encoding="utf-8"), indent=2)
open(os.path.join(P, "CLEANING_LOG.md"), "w", encoding="utf-8").write("\n".join(log) + "\n")
