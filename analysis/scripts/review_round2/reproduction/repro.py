#!/usr/bin/env python3
"""Review round 2: reproduce extracted text and features from the cached source PDFs.

Usage: python3 -I repro.py <cache> <scratch> <out.json> [--per-cell 3 | --all] [--workers 8]

<cache> is the private v3 cache (lists.tsv, pdfs/, text/, features.jsonl); <scratch> receives
re-extracted text (keep it outside git: it contains paper text). <out.json> gets only counts,
hashed ids and field names. Stages R0 (inventory) and R1 (re-extraction) of PLAN.md.
"""
import gzip, hashlib, importlib.util, json, os, platform, subprocess, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SNAPSHOT = os.path.join(ROOT, "scripts", "corpus_validation", "extractor_snapshot.py")
PROV = json.load(open(os.path.join(ROOT, "results", "corpus_validation", "extractor_provenance.json")))
FROZEN = json.load(open(os.path.join(ROOT, "results", "corpus_validation", "frozen_inputs.json")))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def ident(corpus, pid):
    return hashlib.sha256((corpus + pid).encode()).hexdigest()[:12]


def load_extract():
    """Import legacy extract.py with its helper bound to the hashed snapshot, not ~/.claude."""
    assert sha(SNAPSHOT) == PROV["snapshot_sha256"], "extractor snapshot hash mismatch"
    spec = importlib.util.spec_from_file_location("_common", SNAPSHOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["_common"] = mod
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import extract, features  # noqa: E402
    return extract, features


EXTRACT, FEATURES = load_extract()


def job(args):
    corpus, pid, pdf, dst = args
    try:
        r = EXTRACT.one((corpus, pdf, dst))
    except Exception as e:  # recorded, never silently dropped
        return corpus, pid, "error:" + type(e).__name__
    return corpus, pid, "ok" if r else "none"


def main():
    cache, scratch, out = sys.argv[1:4]
    per_cell = int(sys.argv[sys.argv.index("--per-cell") + 1]) if "--per-cell" in sys.argv else 3
    full = "--all" in sys.argv
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    rep = {"stage": "R0+R1", "mode": "all" if full else f"{per_cell}_per_cell"}

    # R0: cache integrity against the frozen manifest.
    rep["cache_hashes_match_frozen"] = {
        k: sha(os.path.join(cache, k)) == v for k, v in FROZEN["source_hashes"].items()}
    lists = [l.rstrip("\n").split("\t") for l in open(os.path.join(cache, "lists.tsv")) if l.strip()]
    feats = {(r["corpus"], r["id"]): r for r in map(json.loads, open(os.path.join(cache, "features.jsonl")))}
    snap = {(r["corpus"], r["id"]): r for r in map(json.loads, gzip.open(
        os.path.join(ROOT, "results", "corpus_validation", "features_numeric.jsonl.gz"), "rt"))}

    # R0: account for every list entry.
    fate, by_cell = Counter(), defaultdict(Counter)
    listed = set()
    for corpus, pid, _url in lists:
        key = (corpus, pid)
        if key in listed:
            fate["duplicate_list_entry"] += 1
            continue
        listed.add(key)
        pdf = os.path.join(cache, "pdfs", corpus, pid + ".pdf")
        txt = os.path.join(cache, "text", corpus, pid + ".json")
        if not os.path.exists(pdf):
            f = "no_pdf"
        elif not os.path.exists(txt):
            f = "no_text_record"
        elif key in feats:
            f = "feature_row"
        else:
            rec = json.load(open(txt))
            pf = FEATURES.paper_features(rec)
            if pf is None:
                f = "under_1500_words"
            elif corpus == "acl2020" and (pf["pages"] < 8 or pf["words"] < 3000):
                f = "acl2020_short_paper_filter"
            else:
                f = "unexplained_missing_feature_row"
        fate[f] += 1
        by_cell[corpus][f] += 1
    unlisted_pdfs = sum(1 for c in os.listdir(os.path.join(cache, "pdfs"))
                        for fn in os.listdir(os.path.join(cache, "pdfs", c))
                        if fn.endswith(".pdf") and (c, fn[:-4]) not in listed)
    rep["inventory"] = {
        "list_entries": len(lists), "unique_list_keys": len(listed), "fate": dict(fate),
        "unlisted_cached_pdfs": unlisted_pdfs,
        "feature_rows": len(feats), "feature_rows_not_listed": len(set(feats) - listed),
        "snapshot_rows": len(snap),
        "snapshot_key_match": set(snap) == {(c, ident(c, p)) for c, p in feats},
        "cells_with_drops": {c: dict(v) for c, v in sorted(by_cell.items())
                             if set(v) - {"feature_row"}},
    }

    # R1: content-independent sample (smallest hashed id per cell) of keys with feature rows.
    cells = defaultdict(list)
    for c, p in feats:
        cells[c].append(p)
    picks = []
    for c in sorted(cells):
        ids = sorted(cells[c], key=lambda p: ident(c, p))
        picks += [(c, p) for p in (ids if full else ids[:per_cell])]
    jobs = [(c, p, os.path.join(cache, "pdfs", c, p + ".pdf"),
             os.path.join(scratch, "text", c, p + ".json")) for c, p in picks]
    for j in jobs:
        if os.path.exists(j[3]):
            os.remove(j[3])  # never trust a previous scratch run
    with Pool(workers) as pool:
        status = pool.map(job, jobs, chunksize=4)

    text_same = Counter()
    field_diffs, feat_diffs, snap_diffs = Counter(), Counter(), Counter()
    exact_feat = exact_snap = 0
    max_abs = defaultdict(float)
    mismatched = []
    for (c, p, st), (_, _, _, dst) in zip(status, jobs):
        if st != "ok":
            text_same[st] += 1
            mismatched.append({"cell": c, "paper_hash": ident(c, p), "stage": "extract", "status": st})
            continue
        new = json.load(open(dst))
        old = json.load(open(os.path.join(cache, "text", c, p + ".json")))
        diff = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
        text_same["identical" if not diff else "different"] += 1
        field_diffs.update(diff)
        pf = FEATURES.paper_features(new)
        ref = feats[(c, p)]
        fd = [k for k in ref if k not in ("corpus", "id") and (pf or {}).get(k) != ref[k]]
        feat_diffs.update(fd)
        exact_feat += not fd
        for k in fd:
            a, b = (pf or {}).get(k), ref[k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                max_abs[k] = max(max_abs[k], abs(a - b))
        sref = snap[(c, ident(c, p))]
        sd = [k for k in sref if k not in ("corpus", "id", "paper_hash") and (pf or {}).get(k) != sref[k]]
        snap_diffs.update(sd)
        exact_snap += not sd
        if diff or fd or sd:
            mismatched.append({"cell": c, "paper_hash": ident(c, p), "text_fields": diff,
                               "feature_fields": fd, "snapshot_fields": sd})
    rep["reextraction"] = {
        "papers": len(jobs), "cells": len(cells), "text_record": dict(text_same),
        "text_fields_differing": dict(field_diffs),
        "features_bit_identical_to_cache": exact_feat,
        "features_bit_identical_to_public_snapshot": exact_snap,
        "feature_fields_differing": dict(feat_diffs), "snapshot_fields_differing": dict(snap_diffs),
        "max_abs_feature_difference": dict(max_abs), "mismatched_papers": mismatched,
    }
    rep["toolchain"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "pdftotext": subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr.splitlines()[0],
        "extractor_snapshot_sha256": sha(SNAPSHOT),
        "extract_py_sha256": sha(os.path.join(ROOT, "scripts", "extract.py")),
        "features_py_sha256": sha(os.path.join(ROOT, "scripts", "features.py")),
        "frozen_extract_py_sha256": FROZEN["legacy_code_hashes"]["extract.py"],
        "frozen_features_py_sha256": FROZEN["legacy_code_hashes"]["features.py"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: rep[k] for k in ("cache_hashes_match_frozen",)}),
          json.dumps(rep["inventory"]["fate"]),
          json.dumps({k: v for k, v in rep["reextraction"].items() if k != "mismatched_papers"}), sep="\n")


if __name__ == "__main__":
    main()
