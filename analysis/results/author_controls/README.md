# Bounded within-author composition check

This is an exploratory descriptive ACL panel, **not** a representative cross-venue author control or an estimate of AI's effect. It supplies actual linked-author evidence with conservative identity handling. No author names, paper titles, or raw source IDs are released here.

## Sources and matching

Retrieved 2026-10-19, pinned official ACL Anthology repository revision `0d4dd55c88a79724914e339ba3511086c05653d0`. Source URLs, XML SHA-256 hashes and byte sizes are in `summary.json`. Frozen numerical input SHA-256 is also recorded. Official identity documentation hashes/URLs and failed DBLP probes are in `source_audit.json`.

Before examining feature outcomes, the script restricted eligibility to exact official paper-ID matches in ACL 2019/2020 (pre) and ACL 2024/2025/2026 (post), with an **explicit `author id` attribute** in official XML, and the same explicit ID observed in both eras. It never derives IDs from names, uses fuzzy title matching, or equates name strings. Legacy `P19` identifiers use volume plus three-digit paper numbering; modern identifiers use collection-volume.paper. Duplicate source paper IDs are rejected. This protocol was selected during exploratory work, not externally preregistered.

The [official names documentation](https://github.com/acl-org/acl-anthology/blob/0d4dd55c88a79724914e339ba3511086c05653d0/hugo/content/info/names.md) distinguishes name slugs from person IDs and explains manual and ORCID-based verification. We trust explicit source person assignments; we do **not** independently verify that each historical assignment is correct. Name-only occurrences, even if they appear to name the same person, are excluded. This loses many valid links and does not make source metadata infallible. [The new author-system description](https://github.com/acl-org/acl-anthology/blob/0d4dd55c88a79724914e339ba3511086c05653d0/hugo/content/posts/2026-01-26-new-author-system.md) documents ambiguity and ORCID verification.

DBLP official/uni-trier NSDI XML probes both returned HTTP 202 HTML anti-bot challenges. No challenge bypass or claim of missing DBLP records is made. Bounded progress therefore uses one venue; a general multi-venue control remains incomplete.

## Coverage and selection

| Corpus | Numerical papers | Exact metadata | Any explicit person ID | Panel papers |
|---|---:|---:|---:|---:|
| ACL 2019 | 80 | 80 | 31 | 3 |
| ACL 2020 | 202 | 202 | 77 | 14 |
| ACL 2024 | 80 | 80 | 18 | 4 |
| ACL 2025 | 200 | 200 | 195 | 3 |
| ACL 2026 | 198 | 198 | 198 | 7 |

All 760 numerical papers match exact source paper metadata: zero ambiguous paper matches, zero unmatched paper IDs. Author linkage is far less complete: 519 papers have any explicit ID, but only **11 recurring explicit person IDs, 31 unique papers (17 pre, 14 post)** survive the both-era rule. Other authors are unlinked, not assigned guessed identities. Explicit author-ID availability changes sharply by year; author occurrence coverage is recorded separately in JSON. There is no basis to treat this selected panel as representative.

`mapping_hashed.jsonl` contains every eligible numerical paper, its match status and hashed explicit/panel author IDs. Paper hashes follow the frozen input convention; author hashes are SHA-256(`acl-author:` + explicit source person ID), truncated to 12 hex characters. These are pseudonymous reproducibility keys, not anonymization guarantees.

## Estimation and observed results

Each paper-author exposure has weight `1 / number of recurring linked authors on that paper`, so a paper's weights sum to one. Weighted regression is `feature = author fixed effect + post coefficient`; venue is held constant by restricting to ACL. The coefficient is independently checked against direct weighted least squares. Unweighted original and panel means are in `summary.json`, including corpus-specific selection differences. No calendar-year fixed effects are added because a full set would absorb the post indicator. No topic adjustment is claimed.

| Feature | Original ACL pre/post means | Panel pre/post means | Author fixed-effect post minus pre |
|---|---|---|---:|
| Sentence length | 21.859 / 21.850 | 22.665 / 22.777 | +0.280 |
| Sentence length CV | 0.519 / 0.504 | 0.532 / 0.503 | -0.0245 |
| Em dashes / 1k words | 0.250 / 0.538 | 0.368 / 0.463 | +0.217 |
| Existing style lexicon / 1k | 1.113 / 3.817 | 1.010 / 2.340 | +0.912 |
| Existing “famous AI words” / 1k | 0.0249 / 0.1023 | 0.0134 / 0.0966 | +0.0837 |
| LLM terms / 1k | 0.0148 / 8.970 | 0.0390 / 9.502 | +9.317 |
| MATTR | 0.458 / 0.499 | 0.467 / 0.490 | +0.0233 |
| Extracted words | 4250 / 4013 | 4671 / 4261 | -345.9 |

Feature definitions are inherited unchanged from the frozen numerical input; names of lexicons are not proof of AI authorship. Some directions persist within recurring linked authors, but the selected style-lexicon shift is considerably smaller than the original ACL comparison. Increased LLM-topic terms show that within-author designs do not remove research-topic change. A paper's prose is coauthored; this is change in papers associated with an author, not attribution to that author's personal writing.

## Dependence and limits

There are 31 paper-author exposures for 31 unique papers in this particular panel, so no paper is duplicated here; general script weighting still protects against repeated paper rows. Shared recurring author IDs create 11 observed connected components of sizes 4,4,4,3,3,3,2,2,2,2,2. These are components under **linked recurring IDs only**: unlinked coauthors and other shared influences can connect them further. No p-values, independent-author-row standard errors or confidence intervals are reported. This tiny selected cohort cannot establish population robustness, eliminate author-composition bias, or identify a causal effect of AI. Data/extraction/version errors upstream remain possible.

## Reproduce

Requires Python 3, NumPy, network only on first metadata retrieval:

```sh
python scripts/author_controls/analyze.py --cache /private/path/author-controls-acl
```

The cache receives full public metadata including names/titles and must stay outside the repository. Released outputs contain only hashes, corpus labels, counts and numeric features/aggregates. Checks performed: exact metadata uniqueness; FE within-transform versus direct weighted least-squares agreement for all eight features; coverage/mapping consistency; no raw names/titles in released mapping. Current input hashes are asserted by inspection of the output manifest, not silently remapped.
