#!/usr/bin/env python3
"""Review round 2, stage R2: re-download a fixed subset of public PDFs and compare with the cache.

Usage: python3 -I redownload.py <cache> <scratch> <out.json>

Subset (fixed in PLAN.md before any download): for every venue, the smallest-hash paper with a
feature row in its earliest and its latest venue-year cell, plus every list entry that has no
cached PDF or whose cached PDF yields no text record. Requests are sequential (arXiv 3 s apart),
with the same curl options as download.py. Only hashes, sizes and feature deltas are written.
"""
import hashlib, json, os, re, subprocess, sys, time
from collections import defaultdict
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro import EXTRACT, FEATURES, ident, sha  # noqa: E402  (binds the hashed extractor snapshot)


def fetch(url, dst):
    for attempt in range(3):
        if os.path.exists(dst):
            os.remove(dst)
        r = subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0 (research corpus study)",
                            "--max-time", "600", "-o", dst, "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        ok = os.path.exists(dst) and open(dst, "rb").read(4) == b"%PDF"
        if ok:
            return "ok", r.stdout
        time.sleep(5 * (attempt + 1))
    return "fail", r.stdout


def main():
    cache, scratch, out = sys.argv[1:4]
    lists = [l.rstrip("\n").split("\t") for l in open(os.path.join(cache, "lists.tsv")) if l.strip()]
    url = {(c, p): u for c, p, u in lists}
    feats = {(r["corpus"], r["id"]): r for r in map(json.loads, open(os.path.join(cache, "features.jsonl")))}
    venue = lambda c: re.sub(r"\d{4}$", "", c)
    cells = defaultdict(lambda: defaultdict(list))
    for c, p in feats:
        cells[venue(c)][c].append(p)
    picks = []
    for v in sorted(cells):
        cs = sorted(cells[v])
        for c in dict.fromkeys([cs[0], cs[-1]]):
            picks.append(("sample", c, min(cells[v][c], key=lambda p: ident(c, p))))
    for c, p, _ in lists:
        pdf = os.path.join(cache, "pdfs", c, p + ".pdf")
        if not os.path.exists(pdf):
            picks.append(("no_cached_pdf", c, p))
        elif not os.path.exists(os.path.join(cache, "text", c, p + ".json")):
            picks.append(("no_text_record", c, p))

    rows, last_arxiv = [], 0.0
    for kind, c, p in picks:
        u = url[(c, p)]
        host = urlparse(u).netloc
        if host == "arxiv.org":
            time.sleep(max(0.0, 3 - (time.time() - last_arxiv)))
        dst = os.path.join(scratch, "pdfs", c, p + ".pdf")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        status, code = fetch(u, dst)
        if host == "arxiv.org":
            last_arxiv = time.time()
        row = {"kind": kind, "cell": c, "paper_hash": ident(c, p), "host": host,
               "status": status, "http": code}
        old = os.path.join(cache, "pdfs", c, p + ".pdf")
        if status == "ok":
            row["new_sha256"], row["new_bytes"] = sha(dst), os.path.getsize(dst)
            if os.path.exists(old):
                row["cached_sha256"], row["cached_bytes"] = sha(old), os.path.getsize(old)
                row["bytes_identical"] = row["new_sha256"] == row["cached_sha256"]
            tdst = os.path.join(scratch, "text", c, p + ".json")
            if os.path.exists(tdst):
                os.remove(tdst)
            try:
                got = EXTRACT.one((c, dst, tdst))
            except Exception as e:
                got, row["extract_error"] = None, type(e).__name__
            row["extracts"] = bool(got)
            if got and (c, p) in feats:
                pf = FEATURES.paper_features(json.load(open(tdst)))
                ref = feats[(c, p)]
                if pf is None:
                    row["features"] = "under_1500_words"
                else:
                    d = {k: pf[k] - ref[k] for k in ref if isinstance(ref[k], (int, float))
                         and not isinstance(ref[k], bool) and pf.get(k) != ref[k]}
                    row["features_identical"] = not d and all(pf.get(k) == ref[k] for k in ref
                                                             if k not in ("corpus", "id"))
                    row["feature_deltas"] = d
            elif got:
                pf = FEATURES.paper_features(json.load(open(tdst)))
                row["would_enter"] = pf is not None
                if pf:
                    row["words"] = pf["words"]
        rows.append(row)
        print(kind, c, row["paper_hash"], status, row.get("bytes_identical"), row.get("features_identical"),
              flush=True)
    json.dump({"stage": "R2", "requests": len(rows), "rows": rows}, open(out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
