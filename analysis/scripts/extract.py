#!/usr/bin/env python3
"""Extract body prose + structural counts from every PDF into one JSON per paper.

Usage: python3 -I extract.py <workdir> [--workers 8]
Reads <workdir>/pdfs/<corpus>/*.pdf, writes <workdir>/text/<corpus>/<id>.json (skips existing).

Prose paragraphs come from roast-paper's _common.py (pdftotext, running headers/footers
dropped, references and everything after them cut, captions dropped); paragraphs under 40
words (axis labels, table cells, equations) are dropped. Section structure comes from a
separate pass over the raw text: numbered headings in sequence (1, 2, 3, ...) plus standalone
standard names ("Related Work", "Limitations"), since ACL-style PDFs lose the numbers.
"""
import json, os, re, subprocess, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.expanduser("~/.claude/skills/roast-paper/scripts"))
from _common import read_paper, prose_paragraphs  # noqa: E402

MIN_WORDS = 40
REF_HEAD = re.compile(r"^\s*(\d+\.?\s*)?(R ?EFERENCES|References|B ?IBLIOGRAPHY|Bibliography)\s*$")
APPX_HEAD = re.compile(r"^\s*([A-Z]\.?\s+)?(A ?PPENDIX|Appendix|Appendices|APPENDICES|Supplementary Material)\b")
NUM_HEAD = re.compile(r"^(\d{1,2})\.?\s+([A-Z][A-Za-z0-9 &:,'/?()-]{2,70})$")
KNOWN = re.compile(
    r"^(?:\d{1,2}\.?\s+)?(introduction|related work[s]?|background(?: and motivation)?|motivation|"
    r"preliminar(?:y|ies)|method[s]?|methodology|approach|design|overview|implementation|"
    r"experiment[s]?|experimental (?:setup|results|evaluation)|evaluation|results|analysis|"
    r"discussion|limitation[s]?|conclusion[s]?|conclusions? and future work|future work|"
    r"ethics statement|ethical considerations|broader impacts?(?: statement)?|impact statement|"
    r"acknowledge?ments?|summary)$", re.I)
KEEP_TEXT = re.compile(r"introduction|related|limitation|discussion|conclusion|broader|ethic|impact")


def norm_heading(s):
    """'I NTRODUCTION' -> 'introduction' (pdftotext letter-spaces small caps)."""
    s = re.sub(r"\b([A-Z]) (?=[A-Z]{2,})", r"\1", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def segments(body):
    """Split raw body lines into [(heading, text)] using sequential numbered headings and
    standalone standard section names."""
    L = [l.strip() for l in body]
    segs, cur, buf, num = [], "(front)", [], 0
    i = 0
    while i < len(L):
        l, title = L[i], None
        m = NUM_HEAD.match(l)
        nl = norm_heading(l)
        if m and int(m.group(1)) == num + 1 and len(m.group(2).split()) <= 8 and not l.endswith("."):
            title, num = m.group(2), num + 1
        elif re.fullmatch(r"\d{1,2}\.?", l) and int(l.rstrip(".")) == num + 1:
            for j in (i + 1, i + 2):
                if j < len(L) and L[j]:
                    if re.match(r"[A-Z][A-Za-z0-9 &:,'/?()-]{2,70}$", L[j]) and len(L[j].split()) <= 8:
                        title, i, num = L[j], j, num + 1
                    break
        elif KNOWN.match(nl):
            title = l
            if nl.startswith("introduction"):
                num = max(num, 1)
        if title:
            segs.append((cur, "\n".join(buf)))
            cur, buf = norm_heading(re.sub(r"^\d{1,2}\.?\s+", "", title)), []
        else:
            buf.append(l)
        i += 1
    segs.append((cur, "\n".join(buf)))
    return [(h, t) for h, t in segs if h != "(front)" or t.strip()]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout


def one(args):
    corpus, path, dst = args
    try:
        rows = read_paper(path)
    except SystemExit:
        return None
    raw = run(["pdftotext", path, "-"])
    pm = re.search(r"Pages:\s+(\d+)", run(["pdfinfo", path]))
    lines = raw.replace("\f", "\n").splitlines()
    ref_i = next((i for i, l in enumerate(lines) if REF_HEAD.match(l)), len(lines))
    appx_i = next((i for i, l in enumerate(lines[:ref_i]) if APPX_HEAD.match(l) and i > 0.4 * ref_i), ref_i)
    body, refs = lines[:appx_i], "\n".join(lines[ref_i:])
    body_txt = "\n".join(body)
    paras = [txt for _, _, _, txt in prose_paragraphs(rows) if len(txt.split()) >= MIN_WORDS]
    segs = segments(body)
    n_bracket = len(set(re.findall(r"^\s*\[(\d{1,3})\]", refs, re.M)))
    n_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\.", refs))
    figs = {m.group(1) for l in body for m in [re.match(r"\s*(?:Figure|Fig\.)\s+(\d+)\s*[:.|]", l)] if m}
    tabs = {m.group(1) for l in body for m in [re.match(r"\s*(?:Table|TABLE)\s+(\d+|[IVX]+)\s*[:.|]", l)] if m}
    # Abstract: text between the Abstract heading and the Introduction heading.
    am = re.search(r"(?:^|\n)\s*(?:A ?BSTRACT|Abstract)\s*[.:—-]?\s*\n?(.*?)\n\s*(?:1\.?\s*)?\n?\s*(?:I ?NTRODUCTION|Introduction)\b",
                   raw, re.S)
    rec = {
        "corpus": corpus, "id": os.path.basename(path)[:-4],
        "pages": int(pm.group(1)) if pm else None,
        "abstract": re.sub(r"\s+", " ", am.group(1)).strip()[:4000] if am else "",
        "paras": paras,
        "segments": [(h, len(t.split()), t if KEEP_TEXT.search(h) else "") for h, t in segs],
        "n_figs": len(figs), "n_tabs": len(tabs),
        "n_refs": n_bracket if n_bracket >= 5 else n_years,
        "github": len(re.findall(r"github\.com/", body_txt)),
        "artifact": bool(re.search(r"[Aa]rtifact (?:[Ee]valuat|[Aa]vailab|[Aa]ppendix)|Artifacts? Available|"
                                   r"Reproducibility [Bb]adge", body_txt)),
        "has_appendix": appx_i < ref_i or bool(re.search(r"^\s*(?:[A-Z]\.?\s+)?(?:A ?PPENDIX|Appendix)\b", refs, re.M)),
    }
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(rec, f)
    return corpus


def main():
    wd = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    jobs = []
    for corpus in sorted(os.listdir(os.path.join(wd, "pdfs"))):
        d = os.path.join(wd, "pdfs", corpus)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".pdf"):
                dst = os.path.join(wd, "text", corpus, fn[:-4] + ".json")
                if not os.path.exists(dst):
                    jobs.append((corpus, os.path.join(d, fn), dst))
    with Pool(workers) as p:
        done = [r for r in p.imap_unordered(one, jobs, chunksize=4) if r]
    print(f"extracted {len(done)} / {len(jobs)}")


if __name__ == "__main__":
    main()
