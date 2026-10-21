#!/usr/bin/env python3
"""Review round 2: run the frozen 1,200-paper same-field expansion (docs/review-round2/expansion/FEASIBILITY.md).

Every phase is resumable and refuses to run out of order. Usage, from the repository root:
  python3 -I scripts/review_round2/expansion_run/expansion_run.py select    # selection.json (commit it)
  python3 -I scripts/review_round2/expansion_run/expansion_run.py fetch     # PDFs into RAW; resumable
  python3 -I scripts/review_round2/expansion_run/expansion_run.py receipts  # receipts.json (commit it)
  python3 -I scripts/review_round2/expansion_run/expansion_run.py run       # extract, features, estimates
  python3 -I scripts/review_round2/expansion_run/expansion_run.py status    # checkpoint summary

PDFs and extracted text stay in RAW (outside git). Committed files hold source ids, URLs, hashes,
statuses and per-paper numeric endpoints only.
"""
import hashlib, json, os, subprocess, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CATALOG = os.path.join(ROOT, "results", "review_round2", "expansion", "catalog_entries.csv")
CATALOG_SHA256 = "8da15b4de55081ebe0c59d60484d970948053840d31d0f7517bafe34695b29a2"  # commit 2d7cfd0
RAW = os.path.expanduser("~/agent-papers-data/before-after-ai/expansion-20261020")
OUT = os.path.join(ROOT, "results", "review_round2", "expansion_run")
SELECTION, RECEIPTS = os.path.join(OUT, "selection.json"), os.path.join(OUT, "receipts.json")
FETCH_LOG = os.path.join(RAW, "fetch_log.jsonl")
UA = "Mozilla/5.0 (research corpus study)"
CELLS = ("cvpr2020", "cvpr2025", "icml2020", "icml2025", "neurips2020", "neurips2025")
VENUES = {"cvpr": ("cvpr2020", "cvpr2025"), "icml": ("icml2020", "icml2025"),
          "neurips": ("neurips2020", "neurips2025")}
PER_CELL, SALT = 200, "20261020:expansion:"
# mlresearch/v267 'main' at selection time; pinned so the ICML 2025 source cannot drift.
V267_COMMIT = "7914d3407046b308db90001490534e422dab7d87"
ENDPOINTS = ("ing_func_1k", "ing_interp_core_1k", "addnot_1k")
SEED, BOOT, PAUSE, TRIES = 20261020, 2000, 1.0, 3


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pdf_url(cell, sid, listed):
    """Same URL templates as the original paper_lists.tsv for these venues."""
    if cell.startswith("cvpr"):
        return listed
    if cell == "icml2020":
        return f"https://proceedings.mlr.press/v119/{sid}/{sid}.pdf"
    if cell == "icml2025":
        return f"https://raw.githubusercontent.com/mlresearch/v267/{V267_COMMIT}/assets/{sid}/{sid}.pdf"
    if cell == "neurips2020":
        return f"https://papers.nips.cc/paper_files/paper/2020/file/{sid}-Paper.pdf"
    if cell == "neurips2025":
        return f"https://papers.nips.cc/paper_files/paper/2025/file/{sid}-Paper-Conference.pdf"
    raise ValueError(cell)


def select():
    import csv
    raw = open(CATALOG, "rb").read()
    assert sha(raw) == CATALOG_SHA256, "catalog changed since the frozen design"
    rows = list(csv.DictReader(open(CATALOG, newline="")))
    sel = []
    for cell in CELLS:
        pool = [r for r in rows if r["cell"] == cell and r["expansion_catalog_eligible"] == "1"
                and r["in_original_manifest"] == "0"]
        assert len({r["source_id"] for r in pool}) == len(pool), cell
        for r in pool:
            r["rank_key"] = sha((SALT + cell + ":" + r["source_id"]).encode("utf-8"))
        pool.sort(key=lambda r: r["rank_key"])
        for i, r in enumerate(pool[:PER_CELL]):
            sel.append({"cell": cell, "source_id": r["source_id"], "rank": i, "rank_key": r["rank_key"],
                        "pool_size": len(pool), "track": r["track"],
                        "pdf_url": pdf_url(cell, r["source_id"], r["pdf_url"])})
    assert len(sel) == PER_CELL * len(CELLS)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"phase": "selection_frozen_before_download", "catalog_sha256": CATALOG_SHA256,
               "salt": SALT, "per_cell": PER_CELL, "v267_commit": V267_COMMIT, "items": sel},
              open(SELECTION, "w"), indent=1, sort_keys=True)
    print(Counter(s["cell"] for s in sel))


def committed(path):
    rel = os.path.relpath(path, ROOT)
    r = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", rel], capture_output=True, text=True)
    t = subprocess.run(["git", "-C", ROOT, "ls-files", "--error-unmatch", rel], capture_output=True)
    return t.returncode == 0 and not r.stdout.strip()


def pdf_path(it):
    return os.path.join(RAW, "pdfs", it["cell"], it["source_id"] + ".pdf")


def valid_pdf(p):
    if not os.path.exists(p):
        return False
    b = open(p, "rb").read()
    return b[:4] == b"%PDF" and b"%%EOF" in b[-2048:]


def fetch():
    assert committed(SELECTION), "commit selection.json before downloading"
    items = json.load(open(SELECTION))["items"]
    os.makedirs(RAW, exist_ok=True)
    # The 3-attempt cap counts attempts across restarts, from the log.
    tried = Counter((r["cell"], r["source_id"]) for r in map(json.loads, open(FETCH_LOG))) \
        if os.path.exists(FETCH_LOG) else Counter()
    for n, it in enumerate(items):
        dst = pdf_path(it)
        if valid_pdf(dst):
            continue  # resume: already complete
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(tried[(it["cell"], it["source_id"])] + 1, TRIES + 1):  # same frozen URL only
            r = subprocess.run(["curl", "-sL", "-A", UA, "--max-time", "600", "-w", "%{http_code}",
                                "-o", dst + ".part", it["pdf_url"]], capture_output=True, text=True)
            ok = valid_pdf(dst + ".part")
            with open(FETCH_LOG, "a") as f:
                f.write(json.dumps({"cell": it["cell"], "source_id": it["source_id"], "attempt": attempt,
                                    "http": r.stdout, "valid_pdf": ok, "utc": time.strftime(
                                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
            time.sleep(PAUSE)
            if ok:
                os.replace(dst + ".part", dst)
                break
            time.sleep(PAUSE * 5 * attempt)
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
        print(n + 1, it["cell"], it["source_id"][:40], "ok" if valid_pdf(dst) else "FAILED", flush=True)


def receipts():
    assert committed(SELECTION)
    items = json.load(open(SELECTION))["items"]
    rec = []
    for it in items:
        p = pdf_path(it)
        row = {"cell": it["cell"], "source_id": it["source_id"], "retrieved": valid_pdf(p)}
        if row["retrieved"]:
            b = open(p, "rb").read()
            row["pdf_sha256"], row["pdf_bytes"] = sha(b), len(b)
        rec.append(row)
    json.dump({"phase": "receipts_frozen_before_extraction",
               "selection_sha256": sha(open(SELECTION, "rb").read()), "items": rec},
              open(RECEIPTS, "w"), indent=1, sort_keys=True)
    print(Counter((r["cell"], r["retrieved"]) for r in rec))


def _extract(args):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "review_round2", "reproduction"))
    import repro  # binds the hashed extractor snapshot, as in the main study
    cell, sid, pdf, dst = args
    if os.path.exists(dst):
        os.remove(dst)
    try:
        ok = repro.EXTRACT.one((cell, pdf, dst))
    except Exception as e:
        return cell, sid, "extraction_error:" + type(e).__name__, None
    if not ok:
        return cell, sid, "extraction_failed", None
    f = repro.FEATURES.paper_features(json.load(open(dst)))
    if f is None:
        return cell, sid, "excluded_under_1500_words", None
    return cell, sid, "admitted", {k: f[k] for k in ("words", "ing_func_1k", "ing_interp_core_1k",
                                                     "op_additionally_1k", "op_notably_1k")}


def contrast(by):
    per = {v: {k: float(np.mean([r[k] for r in by[b]]) - np.mean([r[k] for r in by[a]])) for k in ENDPOINTS}
           for v, (a, b) in VENUES.items()}
    return per, {k: float(np.mean([per[v][k] for v in VENUES])) for k in ENDPOINTS}


def run():
    assert committed(SELECTION) and committed(RECEIPTS), "commit selection.json and receipts.json first"
    rec = json.load(open(RECEIPTS))
    assert rec["selection_sha256"] == sha(open(SELECTION, "rb").read())
    keys = [(r["cell"], r["source_id"]) for r in rec["items"]]
    want = [(s["cell"], s["source_id"]) for s in json.load(open(SELECTION))["items"]]
    assert len(keys) == len(set(keys)) == PER_CELL * len(CELLS), "receipts: wrong count or duplicates"
    assert set(keys) == set(want), "receipts do not match the frozen selection"
    jobs, status = [], []
    for r in rec["items"]:
        it = {"cell": r["cell"], "source_id": r["source_id"]}
        p = pdf_path(it)
        if not r["retrieved"]:
            status.append({**it, "status": "not_retrieved"})
        elif sha(open(p, "rb").read()) != r["pdf_sha256"]:
            status.append({**it, "status": "hash_mismatch_with_receipts"})
        else:
            jobs.append((r["cell"], r["source_id"], p,
                         os.path.join(RAW, "text", r["cell"], r["source_id"] + ".json")))
    for j in jobs:
        os.makedirs(os.path.dirname(j[3]), exist_ok=True)
    with Pool(6) as pool:
        for cell, sid, st, f in pool.imap(_extract, jobs):
            status.append({"cell": cell, "source_id": sid, "status": st, **(f or {})})
    for s in status:
        if s["status"] == "admitted":
            s["addnot_1k"] = s["op_additionally_1k"] + s["op_notably_1k"]
    counts = {c: dict(Counter(s["status"] for s in status if s["cell"] == c)) for c in CELLS}
    blocking = [s for s in status if s["status"] not in ("admitted", "excluded_under_1500_words")]
    rep = {"phase": "run", "receipts_sha256": sha(open(RECEIPTS, "rb").read()), "endpoints": ENDPOINTS,
           "seed": SEED, "bootstrap": BOOT, "status_counts": counts, "n_blocking": len(blocking)}
    if blocking:
        rep["confirmatory"] = "stopped: retrieval/extraction errors must be resolved from the same source first"
    else:
        by = defaultdict(list)
        for s in status:
            if s["status"] == "admitted":
                by[s["cell"]].append(s)
        per, pooled = contrast(by)
        rng = np.random.default_rng(SEED)
        draws = {"pooled": {k: [] for k in ENDPOINTS}, **{v: {k: [] for k in ENDPOINTS} for v in VENUES}}
        for _ in range(BOOT):
            b = {c: [by[c][j] for j in rng.integers(len(by[c]), size=len(by[c]))] for c in CELLS}
            p2, q2 = contrast(b)
            for k in ENDPOINTS:
                draws["pooled"][k].append(q2[k])
                for v in VENUES:
                    draws[v][k].append(p2[v][k])
        ci = {g: {k: [float(x) for x in np.quantile(d[k], [.025, .975])] for k in ENDPOINTS}
              for g, d in draws.items()}
        lows = [ci["pooled"][k][0] for k in ENDPOINTS]
        highs = [ci["pooled"][k][1] for k in ENDPOINTS]
        verdict = ("confirmed" if all(x > 0 for x in lows) else
                   "contrary" if any(x < 0 for x in highs) else "mixed_or_inconclusive")
        rep.update({"n_admitted": {c: len(by[c]) for c in CELLS},
                    "primary_equal_weight_mean_2025_minus_2020": {k: {"est": pooled[k], "ci95": ci["pooled"][k]}
                                                                  for k in ENDPOINTS},
                    "venue_2025_minus_2020": {v: {k: {"est": per[v][k], "ci95": ci[v][k]} for k in ENDPOINTS}
                                              for v in VENUES},
                    "cell_means": {c: {k: float(np.mean([r[k] for r in by[c]])) for k in ENDPOINTS} for c in CELLS},
                    "verdict": verdict})
    json.dump(sorted(status, key=lambda s: (s["cell"], s["source_id"])),
              open(os.path.join(OUT, "paper_status.json"), "w"), indent=1, sort_keys=True)
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(rep, indent=1))


def status():
    items = json.load(open(SELECTION))["items"] if os.path.exists(SELECTION) else []
    got = Counter(it["cell"] for it in items if valid_pdf(pdf_path(it)))
    print("selection committed:", os.path.exists(SELECTION) and committed(SELECTION))
    print("receipts committed:", os.path.exists(RECEIPTS) and committed(RECEIPTS))
    for c in CELLS:
        print(c, "selected", sum(it["cell"] == c for it in items), "pdf_ok", got[c])
    if os.path.exists(os.path.join(OUT, "results.json")):
        print("results:", json.load(open(os.path.join(OUT, "results.json"))).get("verdict", "stopped"))


if __name__ == "__main__":
    {"select": select, "fetch": fetch, "receipts": receipts, "run": run, "status": status}[sys.argv[1]]()
