#!/usr/bin/env python3
"""Stage 2 of the historical calibration (task #5): the §5.13 feature comparison on WildChat editing pairs.

The feature code (counts, rates, boot, lr, METRICS) is executed from scripts/calib.py itself, so the
definitions match the paper exactly. The corpus reference is corpus_lr in results/rewrite_calibration.json.
Usage: python analyze_pairs.py <pairs.jsonl> <tag>   -> results/historical_calibration/calibration_<tag>.json
"""
import ast
import json
import math
import os
import random
import re
import sys

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
from features import WORD, sentences, ing_riders, BOOST, HEDGE, WE, STYLE_LEX, FAMOUS  # noqa: E402
from newfeat import NOMINAL, PASSIVE, syllables  # noqa: E402
from pairs import RX as PAIR_RX  # noqa: E402

MARK = json.load(open(os.path.join(REPO, "results", "excess_markers.json")))["markers"]
NS = {"re": re, "math": math, "random": random, "np": np, "WORD": WORD, "sentences": sentences,
      "ing_riders": ing_riders, "BOOST": BOOST, "HEDGE": HEDGE, "WE": WE, "STYLE_LEX": STYLE_LEX, "FAMOUS": FAMOUS,
      "NOMINAL": NOMINAL, "PASSIVE": PASSIVE, "syllables": syllables, "PAIR_RX": PAIR_RX,
      "MARK_RX": re.compile(r"\b(" + "|".join(map(re.escape, MARK)) + r")\b", re.I)}
src = os.path.join(SCRIPTS, "calib.py")
tree = ast.parse(open(src).read())
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("counts", "rates", "boot", "lr"))
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "METRICS" for t in n.targets))]
exec(compile(ast.Module(body=keep, type_ignores=[]), src, "exec"), NS)
counts, rates, boot, lr, METRICS = NS["counts"], NS["rates"], NS["boot"], NS["lr"], NS["METRICS"]
REF = json.load(open(os.path.join(REPO, "results", "rewrite_calibration.json")))
CORPUS = REF["corpus_lr"]
KEYS = [k for k, _, _ in METRICS]
PRIMARY = ("func", "core", "addnot")


def boot_lr(oc, rc, B=500, seed=0):
    """Paired bootstrap over pairs of the log ratio edit vs original, per feature (95% CI)."""
    rnd = random.Random(seed)
    n = len(oc)
    res = []
    for _ in range(B):
        ix = [rnd.randrange(n) for _ in range(n)]
        a, b = rates([oc[i] for i in ix]), rates([rc[i] for i in ix])
        res.append({k: lr(a[k], b[k], k) for k in KEYS})
    return {k: (float(np.percentile([x[k] for x in res], 2.5)), float(np.percentile([x[k] for x in res], 97.5))) for k in KEYS}


def condition(name, rows):
    oc = [counts(r["passage"]) for r in rows]
    rc = [counts(r["reply"]) for r in rows]
    a, b = rates(oc), rates(rc)
    l = {k: lr(a[k], b[k], k) for k in KEYS}
    ci = boot_lr(oc, rc)
    x = np.array([CORPUS[k] for k in KEYS])
    y = np.array([l[k] for k in KEYS])
    sig = [k for k in KEYS if abs(CORPUS[k]) > 0.1]
    agree = sum(np.sign(CORPUS[k]) == np.sign(l[k]) for k in sig)
    share = [l[k] / CORPUS[k] for k in sig if np.sign(l[k]) == np.sign(CORPUS[k])]
    pos = all(ci[k][0] > 0 for k in PRIMARY)
    neg = any(ci[k][1] < 0 for k in PRIMARY)
    verdict = "reproduced" if pos else ("not reproduced" if neg else "inconclusive")
    return {"condition": name, "n": len(rows), "lr": l, "lr_ci95": ci, "rates_orig": a, "rates_edit": b,
            "spearman": float(st.spearmanr(x, y).correlation), "sign_agree": int(agree), "n_shifted": len(sig),
            "median_share_of_corpus_shift": float(np.median(share)) if share else 0.0,
            "orig_words": sum(c["words"] for c in oc), "edit_words": sum(c["words"] for c in rc),
            "primary_verdict": verdict, "interpretable": len(rows) >= 30}


def main():
    path, tag = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(path)]
    groups = {"all English editing pairs": rows, "academic": [r for r in rows if r["academic"]]}
    for fam in ("gpt-3.5", "gpt-4"):
        groups[f"{fam} · all"] = [r for r in rows if r["family"] == fam]
        groups[f"{fam} · academic"] = [r for r in rows if r["family"] == fam and r["academic"]]
    for b in ("grammar", "polish", "rewrite", "academic"):
        groups[f"bucket {b} · all"] = [r for r in rows if r["bucket"] == b]
    for q in sorted({r["quarter"] for r in rows}):
        groups[f"{q} · all"] = [r for r in rows if r["quarter"] == q]
    # every stratum is listed with its n; strata under 10 pairs are not estimated, under 30 not interpreted
    coverage = {k: len(v) for k, v in groups.items()}
    conds = [condition(k, v) for k, v in groups.items() if len(v) >= 10]
    out = {"pairs_file_rows": len(rows), "metrics": METRICS, "corpus_lr": CORPUS, "conditions": conds,
           "coverage_n": coverage, "not_estimated_lt10": sorted(k for k, n in coverage.items() if n < 10),
           "not_interpreted_lt30": sorted(k for k, n in coverage.items() if n < 30),
           # fixed by the lead (deviation 4): the academic subset is the primary analysis whatever its size
           "primary": "academic"}
    json.dump(out, open(os.path.join(REPO, "results", "historical_calibration", f"calibration_{tag}.json"), "w"),
              indent=1, default=float)
    print(f"{'condition':28s} {'n':>4s} " + " ".join(f"{k:>8s}" for k in KEYS[:9]) + "  spearman sign verdict")
    print(f"{'CORPUS 2019-20 -> 2024-26':28s} {'':4s} " + " ".join(f"{CORPUS[k]:+8.2f}" for k in KEYS[:9]))
    for c in conds:
        print(f"{c['condition'][:28]:28s} {c['n']:4d} " + " ".join(f"{c['lr'][k]:+8.2f}" for k in KEYS[:9]) +
              f"  {c['spearman']:+.2f} {c['sign_agree']}/{c['n_shifted']} {c['primary_verdict']}")
    for c in conds[:2]:
        print(c["condition"], {k: [round(v, 2) for v in c["lr_ci95"][k]] for k in PRIMARY})


if __name__ == "__main__":
    main()
