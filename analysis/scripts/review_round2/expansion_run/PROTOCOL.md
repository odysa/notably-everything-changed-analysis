# Expansion run: execution protocol (owner: WindowsXP)

Executes the frozen design in `docs/review-round2/expansion/FEASIBILITY.md` without changes to its
population, selection rule, extraction, features, endpoints or decision rule. Written and committed
together with `results/review_round2/expansion_run/selection.json`, before any PDF is downloaded.

## Choices the design left open (fixed here, before any download)

1. **Sampling frame.** `results/review_round2/expansion/catalog_entries.csv`, SHA256
   `8da15b4d…29a2` (commit `2d7cfd0`); the script refuses any other bytes. Pool = rows with
   `expansion_catalog_eligible=1` and `in_original_manifest=0`.
2. **Selection.** Per cell, ascending SHA256 of UTF-8 `20261020:expansion:<cell>:<source_id>`;
   the first 200. `selection.json` lists each pick with its rank key and pool size.
3. **PDF URLs.** The same templates as the original `paper_lists.tsv` for these venues: CVPR uses
   the catalog PDF link; ICML 2020 `proceedings.mlr.press/v119/<id>/<id>.pdf`; NeurIPS
   `…/file/<hash>-Paper.pdf` (2020) and `-Paper-Conference.pdf` (2025, as in the original 2024
   list; the 2025 abstract pages carry no PDF meta link). ICML 2025 uses the same GitHub
   `mlresearch/v267` asset path as the original, but **pinned to commit `7914d340`** (the `main`
   head at selection time) instead of `main`, so the source cannot drift.
4. **Retrieval.** Sequential, 1 s between requests. A paper counts as retrieved only if the file
   starts with `%PDF` and ends with `%%EOF`. Up to 3 attempts on the **same** URL, with backoff.
   Every attempt is logged. No replacement papers. `fetch` is resumable: complete files are
   skipped.
5. **Receipts.** After fetching, SHA256 and size of every PDF go to `receipts.json`, which is
   committed **before extraction**. `run` refuses to start unless both files are committed, and
   it re-checks every PDF against its receipt.
6. **Extraction and features.** The original `extract.one` with the hashed helper snapshot
   (through `scripts/review_round2/reproduction/repro.py`), the original `paper_features`, and
   the 1,500-body-word gate. Short papers are reported, not replaced.
7. **Stop rule.** If any selected paper is not retrieved, mismatches its receipt, or fails
   extraction, `run` writes statuses but **no confirmatory estimates**. The only fix allowed is
   retrying the same source; every such step is dated in this file before outcomes are viewed.
8. **Estimates.** Endpoints `ing_func_1k`, `ing_interp_core_1k`, and
   `op_additionally_1k + op_notably_1k`. For each venue, the mean in 2025 minus the mean in 2020;
   the primary estimate per endpoint is the equal-weight mean of the three venue differences.
   Intervals: 2,000 bootstrap draws resampling papers within each of the six cells, seed
   20261020, 2.5/97.5 percentiles, the same draws used for venue and primary intervals.
9. **Decision.** "Confirmed" only if all three **primary** lower bounds are above 0; "contrary"
   if any primary upper bound is below 0; otherwise "mixed/inconclusive". Venue-level intervals
   are reported but do not enter the decision.
10. **Separation.** Nothing is merged into the original 9,472 / 9,101 cohorts. No classifier
    fitting, feature selection or retuning.

## Where things live

- Code: `scripts/review_round2/expansion_run/expansion_run.py` (phases `select`, `fetch`,
  `receipts`, `run`, `status`).
- Committed outputs: `results/review_round2/expansion_run/` (`selection.json`, `receipts.json`,
  `paper_status.json`, `results.json`, README).
- Private, outside git: `~/agent-papers-data/before-after-ai/expansion-20261020/` (PDFs, text,
  `fetch_log.jsonl`).

## Limits stated in advance

Publication year is not writing year. Six cells from three venues cannot separate an AI effect
from shared calendar shocks, topic change or author turnover. Proceedings can be revised after
publication. A confirmed result here supports the same-field era difference only.

## Change log

(Any change after this commit is dated here before its outcome is viewed.)
- 2026-10-20 17:33Z (during fetch, before any receipt or outcome; requested by 徐欢 in review):
  `fetch` now counts earlier attempts from `fetch_log.jsonl`, so the 3-attempt cap holds across
  restarts; `run` asserts the receipts contain exactly the 1,200 selected cell/source_id pairs with
  no duplicates. Sample, URLs and statistics are unchanged.
