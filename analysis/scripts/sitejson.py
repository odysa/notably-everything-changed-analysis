#!/usr/bin/env python3
"""Write the small JSON files the website inlines next to data.json.

Usage: python sitejson.py <workdir> <results> <sitedir>
  corpus.json  papers and body words per venue-year (from <workdir>/features.jsonl)
  stats2.json  copy of results/stats2_summary.json
  detect.json  {"eval": detect_eval.json, "scores": detect_scores.json}
"""
import json, os, re, sys
from collections import defaultdict

wd, res, site = sys.argv[1:4]
NAME = {"acl": "ACL", "icml": "ICML", "cvpr": "CVPR", "neurips": "NeurIPS", "vldb": "VLDB", "mlsys": "MLSys",
        "osdi": "OSDI", "nsdi": "NSDI", "iclr": "ICLR", "aaai": "AAAI", "colm": "COLM", "eurosys": "EuroSys"}
cells = defaultdict(lambda: [0, 0])
for line in open(os.path.join(wd, "features.jsonl")):
    r = json.loads(line)
    m = re.match(r"([a-z]+)(\d{4})$", r["corpus"])
    if not m or m.group(1) not in NAME:
        continue
    c = cells[(NAME[m.group(1)], int(m.group(2)))]
    c[0] += 1
    c[1] += int(r.get("words") or 0)
rows = [{"venue": v, "year": y, "papers": n, "words": w} for (v, y), (n, w) in sorted(cells.items())]
era = lambda y: "pre" if y <= 2020 else "mid" if y <= 2023 else "post"
by_era = defaultdict(int)
for r in rows:
    by_era[era(r["year"])] += r["papers"]
corpus = {"cells": rows, "total": sum(r["papers"] for r in rows), "words": sum(r["words"] for r in rows), "by_era": dict(by_era)}
json.dump(corpus, open(os.path.join(site, "corpus.json"), "w"), indent=1)
json.dump(json.load(open(os.path.join(res, "stats2_summary.json"))), open(os.path.join(site, "stats2.json"), "w"))
json.dump({"eval": json.load(open(os.path.join(res, "detect_eval.json"))), "scores": json.load(open(os.path.join(res, "detect_scores.json")))},
          open(os.path.join(site, "detect.json"), "w"))
print(f"corpus.json: {corpus['total']} papers, {corpus['words'] / 1e6:.1f}M words, {len(rows)} venue-years, {dict(by_era)}")
