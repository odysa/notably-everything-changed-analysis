# Review round 2: 1,200-paper same-field expansion (owner: WindowsXP)

Executes the frozen design in `docs/review-round2/expansion/FEASIBILITY.md` under
`scripts/review_round2/expansion_run/PROTOCOL.md`. Separate from the original 9,472 / 9,101
cohorts; nothing here is merged into them.

## Audit trail

| Step | Commit | What was fixed before the next step |
|---|---|---|
| Protocol + selection | `40c2423` | 200 papers per cell by hashed rank, PDF URLs (ICML 2025 pinned to `mlresearch/v267@7914d340`) |
| Selection check (徐欢) | `b9b2a01` | Pool sizes, top-200 picks, rank keys and original-sample exclusion recomputed independently |
| Guards | `128182a` | Attempt cap across restarts; receipts must equal the selection |
| Receipts | `219c294` | SHA256 and size of all 1,200 PDFs, before any extraction |
| Results | this commit | `paper_status.json`, `results.json` |

## Retrieval and admission

All **1,200/1,200** selected PDFs were retrieved on the first attempt (HTTP 200, complete
`%PDF…%%EOF`), with 1,200 distinct hashes (4.36 GB, kept privately). Every PDF matched its
receipt and extracted without error, so the stop rule did not trigger. The 1,500-word gate
excluded 8 papers, not replaced:

| Cell | Selected | Admitted | Under 1,500 words |
|---|---:|---:|---:|
| CVPR 2020 | 200 | 195 | 5 |
| CVPR 2025 | 200 | 199 | 1 |
| ICML 2020 | 200 | 200 | 0 |
| ICML 2025 | 200 | 199 | 1 |
| NeurIPS 2020 | 200 | 199 | 1 |
| NeurIPS 2025 | 200 | 200 | 0 |

## Primary result: **confirmed**

2025 minus 2020, per 1,000 words; 2,000 within-cell paper bootstraps, seed 20261020.

| Endpoint | Primary (equal-weight mean of venues) | CVPR | ICML | NeurIPS |
|---|---|---|---|---|
| Functional *-ing* matches | **+1.480** [1.369, 1.589] | +1.675 [1.492, 1.862] | +1.218 [1.047, 1.394] | +1.548 [1.354, 1.751] |
| Core interpretive *-ing* matches | **+0.178** [0.156, 0.202] | +0.187 [0.147, 0.231] | +0.161 [0.126, 0.200] | +0.186 [0.146, 0.229] |
| "Additionally," + "Notably," openers | **+0.439** [0.397, 0.482] | +0.575 [0.491, 0.661] | +0.384 [0.318, 0.452] | +0.358 [0.293, 0.428] |

All three primary lower bounds are above 0, so the frozen rule returns **confirmed**. Every
venue-level interval is also above 0 (reported, not part of the decision).

Cell means (2020 → 2025): functional 0.56 → 2.24 (CVPR), 0.43 → 1.65 (ICML), 0.41 → 1.96
(NeurIPS); core interpretive about 0.01–0.02 → 0.17–0.21; openers about 0.10–0.12 → 0.46–0.69.
For orientation only (not a test): these are the same order as the original cohort's cells
(e.g. original CVPR 2025 functional 2.55, ICML 2025 1.63); the new CVPR gaps are somewhat smaller.

## What this does and does not show

It shows that, in **new papers never seen by the study**, the three fixed pattern counts are
much higher in 2025 than in 2020 at CVPR, ICML and NeurIPS, with the original pipeline unchanged.
This addresses sampling luck and selection of the original 150-paper samples in these six cells.

It does not show: that the patterns carry the semantic function their names suggest (human
annotation is still pending; the core-interpretive count is a word-list/regex proxy); anything
about AI use or causation; anything about other venues, other fields or years between 2020 and
2025 (the PLOS cross-field test remains unconfirmed). Source IDs are disjoint from the original
sample, but authors and topics may overlap. Publication year is not writing year.

## Files

- `selection.json`: frozen picks, rank keys, pool sizes, URLs.
- `receipts.json`: PDF hashes and sizes.
- `paper_status.json`: per-paper status, body words and the endpoint values (no text).
- `results.json`: counts, estimates, intervals, verdict.
- Private: `~/agent-papers-data/before-after-ai/expansion-20261020/` (PDFs, text, fetch log).
