#!/usr/bin/env python3
"""Review round 2: recover the 7 list entries lost to download/link problems, as a labelled
sensitivity check. The original cache and the 9,472 / 9,101 results are never modified.

Usage (from the repository root):
  python3 -I scripts/review_round2/recovery/recovery.py freeze   # fetch PDFs, write manifest.json
  # commit manifest.json BEFORE the next phase
  python3 -I scripts/review_round2/recovery/recovery.py run      # verify hashes, extract, compare

The 7 items are fixed by stage R0 of scripts/review_round2/reproduction: list entries with no
cached PDF (4 AAAI) or with a cached PDF that yields no text record (3 USENIX). Replacement
sources: the same USENIX URL (the cached copies are truncated), and for AAAI the article page's
citation_pdf_url (the listed galley resolves to a video page). PDFs and text stay in RAW.
"""
import gzip, hashlib, html, json, os, re, subprocess, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.expanduser("~/agent-papers-data/before-after-ai/v3-20261019")
RAW = os.path.expanduser("~/agent-papers-data/before-after-ai/recovery-20261020")
OUT = os.path.join(ROOT, "results", "review_round2", "recovery")
UA = "Mozilla/5.0 (research corpus study)"
SEED, BOOT = 0, 2000
ENDPOINTS = ("ing_func_1k", "ing_interp_core_1k", "addnot_1k")
sys.path.insert(0, os.path.join(ROOT, "scripts", "review_round2", "reproduction"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ident(c, p):
    return sha((c + p).encode())[:12]


def curl(url, dst=None):
    # --compressed: OJS article pages arrive gzip-encoded (as in build_lists.get).
    cmd = ["curl", "-sL", "--compressed", "-A", UA, "--max-time", "600", "-w", "%{http_code}"]
    r = subprocess.run(cmd + (["-o", dst] if dst else []) + [url], capture_output=True)
    return r.stdout if dst is None else r.stdout.decode()


def items():
    lists = [l.rstrip("\n").split("\t") for l in open(os.path.join(CACHE, "lists.tsv")) if l.strip()]
    out = []
    for c, p, u in lists:
        pdf = os.path.join(CACHE, "pdfs", c, p + ".pdf")
        if not os.path.exists(pdf):
            out.append({"corpus": c, "id": p, "listed_url": u, "problem": "no_cached_pdf"})
        elif not os.path.exists(os.path.join(CACHE, "text", c, p + ".json")):
            out.append({"corpus": c, "id": p, "listed_url": u, "problem": "cached_pdf_unreadable",
                        "cached_pdf_sha256": sha(open(pdf, "rb").read())})
    assert len(out) == 7, len(out)
    return out


def freeze():
    rows = []
    for it in items():
        u = it["listed_url"]
        m = re.match(r"https://ojs\.aaai\.org/index\.php/AAAI/article/download/(\d+)/\d+$", u)
        if m:
            page = curl(f"https://ojs.aaai.org/index.php/AAAI/article/view/{m.group(1)}")[:-3].decode("utf-8", "replace")
            cand = re.findall(r'<meta name="citation_pdf_url" content="([^"]+)"', page)
            it["source_rule"] = "aaai_article_page_citation_pdf_url"
            it["replacement_url"] = html.unescape(cand[0]).replace("/article/view/", "/article/download/") if cand else None
        else:
            it["source_rule"] = "same_listed_url"
            it["replacement_url"] = u
        dst = os.path.join(RAW, "pdfs", it["corpus"], it["id"] + ".pdf")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        it["http"] = curl(it["replacement_url"], dst) if it["replacement_url"] else None
        ok = os.path.exists(dst) and open(dst, "rb").read(4) == b"%PDF"
        it["retrieved"] = ok
        if ok:
            b = open(dst, "rb").read()
            it["pdf_sha256"], it["pdf_bytes"] = sha(b), len(b)
            it["pdf_complete"] = b"%%EOF" in b[-2048:]
        it["paper_hash"] = ident(it["corpus"], it["id"])
        rows.append(it)
        print(it["corpus"], it["paper_hash"], it["retrieved"], it.get("pdf_bytes"), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"phase": "frozen_before_extraction", "items": rows}, open(os.path.join(OUT, "manifest.json"), "w"),
              indent=1, sort_keys=True)


def cells_by_venue():
    import analyze  # legacy venue/era mapping; read-only
    main = {v: m for v, m in analyze.VENUES.items() if not v.endswith("*")}
    return main, analyze.SERIES


def contrasts(rows, main):
    by = defaultdict(list)
    for r in rows:
        by[r["corpus"]].append(r)
    res = {}
    for v, (_ctrl, pre, post) in main.items():
        pre_r = [r for c in pre for r in by[c]]
        post_r = [r for c in post for r in by[c]]
        res[v] = {"n_pre": len(pre_r), "n_post": len(post_r), **{
            k: float(np.mean([r[k] for r in post_r]) - np.mean([r[k] for r in pre_r])) for k in ENDPOINTS}}
    return res, by


def boot(by, cells_pre, cells_post, rng):
    """Cell-stratified paper bootstrap of post - pre for each endpoint."""
    out = {k: [] for k in ENDPOINTS}
    for _ in range(BOOT):
        draw = {c: [by[c][j] for j in rng.integers(len(by[c]), size=len(by[c]))] for c in cells_pre + cells_post}
        pre = [r for c in cells_pre for r in draw[c]]
        post = [r for c in cells_post for r in draw[c]]
        for k in ENDPOINTS:
            out[k].append(np.mean([r[k] for r in post]) - np.mean([r[k] for r in pre]))
    return {k: [float(x) for x in np.quantile(v, [.025, .975])] for k, v in out.items()}


def run():
    import repro  # binds the hashed extractor snapshot
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    snap = [json.loads(l) for l in gzip.open(os.path.join(ROOT, "results", "corpus_validation",
                                                          "features_numeric.jsonl.gz"), "rt")]
    main, series = cells_by_venue()
    # Corrected 9,101-paper main cohort: everything except the starred COLM/EuroSys cells.
    base = [dict(r) for r in snap if not r["corpus"].startswith(("colm", "eurosys"))]
    assert len(base) == 9101, len(base)
    status, added = [], []
    for it in man["items"]:
        st = {"corpus": it["corpus"], "paper_hash": it["paper_hash"], "problem": it["problem"]}
        pdf = os.path.join(RAW, "pdfs", it["corpus"], it["id"] + ".pdf")
        if not it.get("retrieved"):
            st["status"] = "not_retrieved"
        elif sha(open(pdf, "rb").read()) != it["pdf_sha256"]:
            st["status"] = "hash_mismatch_with_frozen_manifest"
        else:
            dst = os.path.join(RAW, "text", it["corpus"], it["id"] + ".json")
            if os.path.exists(dst):
                os.remove(dst)
            if not repro.EXTRACT.one((it["corpus"], pdf, dst)):
                st["status"] = "extraction_failed"
            else:
                f = repro.FEATURES.paper_features(json.load(open(dst)))
                if f is None:
                    st["status"] = "excluded_under_1500_words"
                else:
                    st["status"], st["words"] = "admitted", f["words"]
                    f["corpus"], f["id"] = it["corpus"], it["paper_hash"]
                    st.update({k: f[k] for k in ("ing_func_1k", "ing_interp_core_1k",
                                                 "op_additionally_1k", "op_notably_1k")})
                    added.append(f)
        st["role"] = next((("post" if it["corpus"] in post else "pre") + ":" + v
                           for v, (_c, pre, post) in main.items() if it["corpus"] in pre + post),
                          "series_only")
        status.append(st)
    for r in base + added:
        r["addnot_1k"] = r["op_additionally_1k"] + r["op_notably_1k"]

    b_res, b_by = contrasts(base, main)
    a_res, a_by = contrasts(base + added, main)
    affected = sorted({s["role"].split(":")[1] for s in status if s["status"] == "admitted" and ":" in s["role"]})
    venue_rows = {}
    for v in affected:
        _c, pre, post = main[v]
        venue_rows[v] = {
            "frozen": {**b_res[v], "ci95": boot(b_by, pre, post, np.random.default_rng(SEED))},
            "with_recovered": {**a_res[v], "ci95": boot(a_by, pre, post, np.random.default_rng(SEED))},
            "change": {k: a_res[v][k] - b_res[v][k] for k in ENDPOINTS}}
    pooled = {}
    for name, res in (("frozen", b_res), ("with_recovered", a_res)):
        pooled[name] = {k: float(np.mean([res[v][k] for v in main])) for k in ENDPOINTS}
    pooled["change"] = {k: pooled["with_recovered"][k] - pooled["frozen"][k] for k in ENDPOINTS}
    series_rows = {}
    for s in status:
        if s["status"] == "admitted" and s["role"] == "series_only":
            c = s["corpus"]
            series_rows[c] = {name: {"n": len(by[c]), **{k: float(np.mean([r[k] for r in by[c]])) for k in ENDPOINTS}}
                              for name, by in (("frozen", b_by), ("with_recovered", a_by))}
    rep = {"phase": "run", "manifest_sha256": sha(open(os.path.join(OUT, "manifest.json"), "rb").read()),
           "endpoints": ENDPOINTS, "seed": SEED, "bootstrap": BOOT,
           "items": status, "n_admitted": len(added),
           "pre_post_affected_venues": venue_rows,
           "ten_venue_equal_weight_mean_of_venue_contrasts": pooled,
           "series_cells_cell_means": series_rows}
    json.dump(rep, open(os.path.join(OUT, "sensitivity.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(rep, indent=1)[:6000])


if __name__ == "__main__":
    {"freeze": freeze, "run": run}[sys.argv[1]]()
