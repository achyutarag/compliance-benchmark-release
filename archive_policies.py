"""
Archive the nine official policy documents and extract their text for the RAG corpus.

For each document:
  policies/archive/<name>_<date>.<html|pdf>   original bytes as served (archival copy)
  policies/<name>.txt                         extracted text (to be checked by a human before freezing)
  policies/MANIFEST.json                      URL, retrieval time, HTTP status, SHA-256 of archive and text

The script identifies itself honestly and does not try to get around sites that refuse
automated requests; those are reported for manual saving (Ctrl+P -> Save as PDF).
"""
import datetime, hashlib, io, json, os, re, sys
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "policies"); ARCH = os.path.join(OUT, "archive")
os.makedirs(ARCH, exist_ok=True)
TODAY = datetime.date.today().isoformat()
UA = {"User-Agent": "research-policy-archiver/1.0 (academic study of gift-policy compliance; one request per document)"}

DOCS = [
    ("hartwick_6-42", "https://www.hartwick.edu/policies/gifts-loans-favors-policy/"),
    ("upenn_VI-D", "https://catalog.upenn.edu/faculty-handbook/vi/vi-d/"),
    ("utk_FI0717", "https://policy.tennessee.edu/policy/fi0717-employee-gift-acceptance-policy/"),
    ("utk_FI0717_guidance", "https://policy.tennessee.edu/guidance-document/fi0717-acceptable-and-unacceptable-gifts/"),
    ("depaul_gifts_to_employees", "https://offices.depaul.edu/secretary/policies-procedures/policies/Documents/Gifts%20to%20Employees.pdf"),
    ("uva_FIN-019", "https://uvapolicy.virginia.edu/policy/FIN-019"),
    ("uthealth_HOOP-226", "https://www.uth.edu/hoop/policy.htm?id=7538ca07-b868-4a20-b9bf-33bbde5b5097"),
    ("ma_930CMR5", "https://www.sec.state.ma.us/reg_pub/pdf/900/930005.pdf"),
    ("ma_advisory_19-1", "https://www.mass.gov/advisory/19-1-gifts-and-gratuities"),
]


def pdf_text(data):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def html_text(data):
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "svg", "button"]):
        tag.decompose()
    node = soup.find("main") or soup.find("article") or soup.body or soup
    text = node.get_text("\n")
    lines = [re.sub(r"[ \t ]+", " ", l).strip() for l in text.splitlines()]
    out, blank = [], 0
    for l in lines:
        if not l:
            blank += 1
            if blank <= 1: out.append("")
        else:
            blank = 0; out.append(l)
    return "\n".join(out).strip()


sha = lambda b: hashlib.sha256(b).hexdigest()
manifest, manual = [], []
for name, url in DOCS:
    rec = {"name": name, "url": url, "retrieved": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        r = requests.get(url, headers=UA, timeout=30)
        rec["http_status"] = r.status_code
        if r.status_code != 200 or not r.content:
            raise RuntimeError(f"HTTP {r.status_code}")
        is_pdf = url.lower().endswith(".pdf") or r.headers.get("Content-Type", "").startswith("application/pdf")
        ext = "pdf" if is_pdf else "html"
        arch = os.path.join(ARCH, f"{name}_{TODAY}.{ext}")
        open(arch, "wb").write(r.content)
        text = pdf_text(r.content) if is_pdf else html_text(r.content)
        if len(text) < 300:
            raise RuntimeError(f"extracted only {len(text)} characters (page may need JavaScript or a login)")
        txt = os.path.join(OUT, f"{name}.txt")
        header = f"SOURCE: {url}\nRETRIEVED: {rec['retrieved']}\nARCHIVE: archive/{os.path.basename(arch)}\n" + "=" * 70 + "\n"
        open(txt, "w", encoding="utf-8").write(header + text + "\n")
        rec.update(status="ok", archive=os.path.relpath(arch, OUT), archive_sha256=sha(r.content),
                   text=os.path.relpath(txt, OUT), text_sha256=sha(open(txt, "rb").read()), text_chars=len(text))
        print(f"OK      {name:26} {ext:4} {len(r.content)/1024:7.1f} KB -> {len(text):6d} chars of text")
    except Exception as e:
        rec.update(status="manual_needed", error=str(e)); manual.append((name, url))
        print(f"MANUAL  {name:26} {e}")
    manifest.append(rec)

json.dump({"created": TODAY, "documents": manifest, "frozen": False,
           "note": "Set frozen=true only after a human has checked each .txt against its source."},
          open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8"), indent=2)
print(f"\n{len(DOCS) - len(manual)}/{len(DOCS)} archived automatically.")
for n, u in manual:
    print(f"  save manually: {n}  <-  {u}")
