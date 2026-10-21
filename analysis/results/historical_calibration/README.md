# Historical ChatGPT editing calibration (task #5): result

Protocol: `PROTOCOL.md`. It was frozen before analysis and has four logged deviations: three to pair
selection, made before any feature was computed, and one reporting-rule correction made after (it changes no
estimate). Code: `scripts/historical_calibration/{select_pairs.py, analyze_pairs.py}`. The
feature code is executed from `scripts/calib.py` itself. Data: WildChat-1M revision in `manifest.json`.
Only aggregates are committed; pair text stays outside the repository.

## Coverage (80–200-word passages, as in §5.13)

Every stratum's n is in `calibration_max200.json` → `coverage_n`. Strata under 10 pairs are not estimated
(for example GPT-4 academic, n = 8), and strata under 30 are not interpreted.

837,989 conversations → 478,498 English → 677 editing pairs after all rules (exclusion counts in
`selection.json`). **Academic subset (primary): 64 pairs** (GPT-3.5: 56, GPT-4: 8). All English editing
pairs (exploratory): 677 (GPT-3.5: 564, GPT-4: 113). The 80–400-word sensitivity run gives 96 academic
and 880 total pairs.

## Primary result (academic subset, n = 64): **inconclusive**

Log ratio of edit vs original, with paired-bootstrap 95% CIs (B = 500). The corpus shift is shown for comparison.

| Feature | Corpus 2019–20 → 2024–26 | 2023–24 ChatGPT edits (academic) |
|---|---|---|
| Functional -ing clauses | +1.21 | +0.06 [−0.42, 0.74] |
| Interpretive -ing clauses | +1.12 | +1.69 [0.00, 2.53] |
| *Additionally/Notably* openers | +1.66 | +1.41 [0.64, 2.77] |
| Spearman with corpus shift (15 features) | | +0.59; sign agreement 12/14 |

Per the frozen rule this is *inconclusive*: the functional-clause CI includes zero and the interpretive
CI touches it. The 80–400-word sensitivity run (n = 96) gives the same verdict.

## Exploratory mechanism result (all English editing pairs, n = 677): reproduced

| Feature | Corpus | All pairs (n = 677) | GPT-4 only (n = 113) |
|---|---|---|---|
| Functional -ing clauses | +1.21 | +0.44 [0.23, 0.65] | +0.76 [0.08, 1.81] |
| Interpretive -ing clauses | +1.12 | +0.67 [0.17, 1.21] | +2.36 [1.61, 2.91] |
| *Additionally/Notably* | +1.66 | +1.20 [0.88, 1.52] | +2.11 [1.27, 3.27] |
| Spearman with corpus shift | | +0.65 (12/14 signs) | +0.70 (13/14) |

For contrast, the §5.13 calibration found that current Claude models *removed* these constructions
(Spearman −0.32 to +0.32). The two experiments differ in users, texts, requests and selection, so the
contrast cannot be attributed to model generation alone.

## Where the 2023–24 edits and the corpus disagree

- The famous ChatGPT words (*delve, tapestry*…) **rise** in the edits (+0.85) but **fell** in CS papers
  (−0.37). Either CS authors removed them, or the CS shift is not straight tool output.
- First-person *we/our/us* rises slightly in the edits (+0.15) but fell in the corpus (−0.15).

## What this does and does not show

- It shows that the editing tools of 2023–24 *could* produce most of the clause, opener and stance shift,
  which current Claude models do not. That addresses the paper's main open question about mechanism.
- It does **not** show that the paper authors used these tools, or how much of the corpus shift they caused.
- WildChat users are not a sample of CS authors. Passages cover many genres, and the academic subset is small
  (GPT-4 academic: 8 pairs, not interpretable).
