# Balanced expansion feasibility: six catalog cells

Checked official catalogs on 2026-10-20. This is a bounded audit of three venues and two publication years, **not a claim that all venues or all eligible papers are covered**. Only proceedings listings were fetched; no PDFs, extracted text, feature outcomes or new classifier predictions were retrieved.

| Cell | Current catalog entries | Original selected manifest | Current numeric cohort | New catalog candidates outside original selection |
|---|---:|---:|---:|---:|
| CVPR 2020 | 1466 | 150 | 149 | 1316 |
| CVPR 2025 | 2871 | 150 | 150 | 2721 |
| ICML 2020 (PMLR119) | 1084 | 150 | 150 | 934 |
| ICML 2025 (PMLR267) | 3330 | 150 | 150 | 3180 |
| NeurIPS 2020 | 1898 | 150 | 149 | 1748 |
| NeurIPS 2025 main conference track | 5286 | 0 | 0 | 5286 |

The original study sampled 150 source IDs in each of the first five cells, rather than exhausting their proceedings. All 750 selected IDs are present in today's corresponding catalogs. The numeric feature cache contains 748 of those papers; that difference alone does not identify why two papers were excluded. They remain excluded from the expansion pool to avoid reusing original selected sources. NeurIPS2025 is absent from the original manifest, but its proceedings **are published now**. Its main-conference volume lists 5823 entries across three tracks: 5286 main conference, 497 datasets/benchmarks, 40 position papers. The proposed expansion restricts 2025 to main conference track. The bare /2025 URL currently defaults to a separate 64-paper Creative AI volume; using it would silently choose the wrong population.

## Sources and audit trail

- [CVPR2020 June16](https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-16), [June17](https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-17), [June18](https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-18). The `?day=all` form returns a database date error for2020. The browser search tool returns403 for CVF, but direct HTTPS retrieval of these official pages succeeded; catalog counts come from those downloaded pages.
- [CVPR2025 all days](https://openaccess.thecvf.com/CVPR2025?day=all).
- [ICML2020 PMLR119](https://proceedings.mlr.press/v119/), [ICML2025 PMLR267](https://proceedings.mlr.press/v267/). The latter reports publication on2025-10-06.
- [NeurIPS2020](https://papers.nips.cc/paper_files/paper/2020), [NeurIPS2025 main volume](https://papers.nips.cc/paper_files/paper/2025/vol38-main-conference).

`results/review_round2/expansion/catalog_entries.csv` contains source IDs, source-page/PDF links when present in listings, track labels, original-selection membership and candidate flags; no titles or abstract prose. `summary.json` contains official URL/byte-hash receipts and hashes of original manifests/numeric inputs. Private HTML snapshots are kept under `$BEFORE_AFTER_AI_DATA_DIR/review_round2_expansion_metadata/` (default `~/.cache/before-after-ai/review_round2_expansion_metadata/`). `scripts/review_round2/expansion/catalogs.py` reproduces the metadata comparison from cached pages or current endpoints. PDFs are never fetched by this script.

## Eligibility and interpretation limits

These are unique catalog entries, not verified successfully downloadable/extractable long research papers. CVPR links are main-conference repository PDFs rather than workshop URLs. ICML entries are article-level links in the named proceedings volumes; short/front-matter status and PDF accessibility have not been separately checked. NeurIPS2020 has legacy `data-track=none` and is treated as its original proceedings population;2025 separates tracks explicitly. Proceedings can contain later corrections, withdrawals or revisions. Source IDs are scoped to publisher/volume: the same surname-year token in different PMLR volumes is not sufficient evidence of a duplicate paper. Author overlap and republished/content duplicates have not been screened. A larger same-field sample improves precision and coverage within these cells; it does not establish AI causality, remove common year shocks, or repair the inconclusive new-field result.

## Frozen-design proposal; execution not yet started

Before any selected PDFs/outcomes, commit an execution protocol and the selected source-map/hash. Use this exact saved catalog as the sampling frame. For each of the six cells, remove every original selected canonical publisher/volume/source identity, regardless of prior extraction success. Keep only the six populations defined above; no alternate venues, years or tracks. Sort remaining identities by ascending SHA256 of UTF-8 `20261020:expansion:` + cell + `:` + source_id. Take exactly200 per cell:1200 new selected papers, balanced across venue and era. All six current candidate pools exceed200. Freeze complete selected source URLs before downloads; resolving a missing PDF link may inspect only the selected paper's official metadata page.

Use unchanged original PDF extraction with committed helper injection, unchanged features and the1500-body-word gate. No replacements for missing downloads, extraction failures or short papers. Record all statuses and denominators; any retrieval/extraction error or incomplete selected count stops confirmatory estimates until honestly resolved using the same source, never replaced. Length exclusions remain reported without replacement and may unbalance analyzed counts. Do not merge expansion rows into the original study or relabel original totals.

Exactly three primary endpoints remain ing_func_1k, ing_interp_core_1k, and op_additionally_1k+op_notably_1k; directions positive. Define venue-specific2025-minus2020 mean differences and a primary equal-weight mean of the three venue differences. Use2000 within-venue/year paper bootstraps, seed20261020, reporting all endpoints, all six counts and venue-specific differences. Full three-endpoint confirmation requires all primary interval lower bounds >0; otherwise mixed/inconclusive or contrary. This is conditional on selected venue-years; six cells cannot identify a general calendar-shock variance. No classifier fitting, retuning, feature reselection or outcome-driven sample expansion. Record publication-versus-writing-time and source-revision limits explicitly.

This document freezes a concrete proposed design for review, **not a report that the1200-paper experiment has run**. Any execution change must be dated and recorded before looking at its outcomes.
