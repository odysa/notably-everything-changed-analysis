"""Shared helpers: load a paper (.tex/.txt/.md/.pdf) into lines, optionally stripping LaTeX."""
import os
import re
import subprocess
import sys


def read_paper(path, follow_inputs=True, _seen=None):
    """Return list of (file, lineno, text) for the paper. PDFs go through pdftotext -layout."""
    _seen = _seen if _seen is not None else set()
    path = os.path.abspath(path)
    if path in _seen:
        return []
    _seen.add(path)
    if path.lower().endswith(".pdf"):
        try:
            # Default (non-layout) mode gives the right reading order for two-column papers.
            out = subprocess.run(["pdftotext", path, "-"], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as e:
            sys.exit(f"pdftotext failed on {path}: {e}")
        lines = out.replace("\f", "\n").splitlines()
        # Running page headers/footers (paper title, venue line) repeat on every page: drop them.
        # Page numbers make each footer line unique ("USENIX Association ... 423"), so count
        # repeats with digits blanked; also drop the first-page sponsorship/ISBN footer.
        key = lambda l: re.sub(r"\d+", "#", l.strip())
        counts = {}
        for l in lines:
            if len(l.strip()) > 15:
                counts[key(l)] = counts.get(key(l), 0) + 1
        lines = [l for l in lines if counts.get(key(l), 0) < 4 and not re.search(
            r"is sponsored by|ISBN \d|Open access to the Proceedings|^\s*USENIX Association\s", l)]
        rows, skip = [], set()
        for i, l in enumerate(lines):
            if i in skip:
                continue
            if re.fullmatch(r"\s*([A-Z]|\d{1,2}(\.\d{1,2})*)\s*", l):
                # Section number on its own line: glue it to a heading-like line that follows.
                for j in (i + 1, i + 2):
                    nxt = lines[j].strip() if j < len(lines) else ""
                    if nxt and len(nxt.split()) <= 10 and nxt[0].isupper() and not nxt.endswith("."):
                        rows.append((os.path.basename(path), i + 1, ""))
                        rows.append((os.path.basename(path), i + 1, f"{l.strip()} {nxt}"))
                        rows.append((os.path.basename(path), i + 1, ""))
                        skip.update(range(i + 1, j + 1))
                        break
                    if nxt:
                        break
                continue  # otherwise: review-copy line numbers and page numbers
            rows.append((os.path.basename(path), i + 1, l))
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()
    rows = []
    base = os.path.dirname(path)
    for i, line in enumerate(raw):
        m = re.search(r"\\(?:input|include)\{([^}]+)\}", line) if follow_inputs else None
        if m and not line.lstrip().startswith("%"):
            sub = m.group(1)
            cand = os.path.join(base, sub if sub.endswith(".tex") else sub + ".tex")
            if os.path.exists(cand):
                rows.extend(read_paper(cand, follow_inputs, _seen))
                continue
        rows.append((os.path.basename(path), i + 1, line))
    return rows


def is_tex(rows):
    return any(r[0].endswith(".tex") for r in rows)


def cut_references(rows):
    """Drop the bibliography / references section and anything after it."""
    for idx, (_, _, t) in enumerate(rows):
        s = t.strip()
        if re.match(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography|"
                    r"\\(sub)?section\*?\{(References|Bibliography)\}", s):
            return rows[:idx]
        # PDF small caps often come out letter-spaced: "R EFERENCES", "B IBLIOGRAPHY".
        if re.fullmatch(r"(\d+\s*)?(R ?EFERENCES|References|B ?IBLIOGRAPHY|Bibliography)", s):
            return rows[:idx]
    return rows


def strip_tex_line(t):
    """Remove comments and commands from one LaTeX line, keeping readable words."""
    t = re.sub(r"(?<!\\)%.*", "", t).replace("{,}", ",")
    t = re.sub(r"\\(cite\w*|ref|cref|Cref|autoref|label|eqref|url|href|includegraphics)\*?(\[[^\]]*\])*\{[^}]*\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " MATH ", t)
    t = re.sub(r"\\(textbf|textit|emph|texttt|textsc|underline|mbox|text)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"[{}~]", " ", t)
    t = t.replace("---", "—").replace("--", "–")
    return re.sub(r"\s+", " ", t).strip()


SKIP_ENVS = ("figure", "table", "tabular", "equation", "align", "algorithm", "lstlisting",
             "verbatim", "minted", "tikzpicture", "comment")


def prose_paragraphs(rows):
    """Yield (file, start_line, section, text) paragraphs of prose (LaTeX envs/tables skipped)."""
    tex = is_tex(rows)
    rows = cut_references(rows)
    depth = 0
    section = "(front)"
    buf, start = [], None
    in_doc = not tex or not any("\\begin{document}" in r[2] for r in rows)

    def flush():
        nonlocal buf, start
        if buf:
            yield_val = (start[0], start[1], section, " ".join(buf))
            buf, start = [], None
            return yield_val
        return None

    out = []
    for f, n, t in rows:
        if tex:
            if "\\begin{document}" in t:
                in_doc = True
                continue
            if not in_doc:
                continue
            if re.search(r"\\begin\{(%s)\*?\}" % "|".join(SKIP_ENVS), t):
                depth += 1
            if depth:
                if re.search(r"\\end\{(%s)\*?\}" % "|".join(SKIP_ENVS), t):
                    depth -= 1
                continue
            m = re.search(r"\\(sub)*section\*?\{([^}]*)\}", t)
            if m:
                p = flush()
                if p:
                    out.append(p)
                section = m.group(2)
                continue
            m = re.search(r"\\begin\{abstract\}", t)
            if m:
                section = "Abstract"
            s = strip_tex_line(t)
        else:
            s = t.strip()
            m = (re.fullmatch(r"([A-Z]|[1-9]\d?(\.\d+)*)\.?\s+([A-Z][A-Za-z ,:&'-]{2,70})", s)
                 or re.fullmatch(r"(ABSTRACT|Abstract|A BSTRACT)", s))
            # Affiliation superscripts ("1 School of ...") and axis ticks ("10 PowerDNS") are not
            # headings: a numbered heading needs a plausible section number and a title-like text.
            if m and m.group(1)[0].isdigit() and (int(m.group(1).split(".")[0]) > 15 or re.search(
                    r"\b(University|School|Institute|Laboratory|Lab|Inc|Group|College|Department|Corporation)\b", s)):
                m = None
            if m:
                p = flush()
                if p:
                    out.append(p)
                section = s
                continue
            ra = re.match(r"(Abstract|ABSTRACT)\s*[.:—–-]\s*(\S.*)", s)
            if ra:  # run-in abstract: "Abstract. As the core of DNS services, ..."
                p = flush()
                if p:
                    out.append(p)
                section, s = "Abstract", ra.group(2)
        if not s:
            p = flush()
            if p:
                out.append(p)
            continue
        if start is None:
            start = (f, n)
        buf.append(s)
    p = flush()
    if p:
        out.append(p)
    # Re-join hyphenated PDF line breaks; drop figure/table captions (not prose).
    # Title/author/affiliation block before an Abstract heading is not prose.
    first_abs = next((i for i, o in enumerate(out) if o[2].lower().replace(" ", "") == "abstract"), None)
    if first_abs is not None:
        out = out[first_abs:]
    cap = re.compile(r"^(Figure|Fig\.|Table|Algorithm|Listing)\s+[A-Z]?\d+[.:]")
    return [(f, n, sec, re.sub(r"(\w)- (\w)", r"\1\2", txt)) for f, n, sec, txt in out
            if not cap.match(txt)]


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", text)
    return [p for p in parts if len(p.split()) >= 3]
