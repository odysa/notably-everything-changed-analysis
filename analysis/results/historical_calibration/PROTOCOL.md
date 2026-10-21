# Historical ChatGPT editing calibration: frozen protocol (task #5)

Written and committed **before** any feature was computed on WildChat. Changes are listed under
"Deviations" with a date and reason. Deviations 1–3 were made before any feature was computed.
Deviation 4 was made after, and only changes how results are reported.

## Question

When 2023–24 ChatGPT models were asked to edit prose, did their edits move the paper's style features in
the same direction as the real 2019–20 → 2024–26 corpus shift? This is a **mechanism check** of the tools
of that period. It is not evidence that any paper's authors used them, and it must not be written up as such.

## Data

- WildChat-1M (allenai/WildChat-1M, ODC-BY), real ChatGPT conversations from April 2023 to April 2024,
  each labelled with the serving model and a timestamp. The exact dataset revision is recorded in
  `REVISION` next to the downloaded files and copied into `results/historical_calibration/manifest.json`.
- Raw data and extracted text stay outside every repository. Only aggregate counts and rates are committed.

## Pair construction (first user turn → first assistant turn)

1. The conversation language is English and the first assistant turn exists.
2. **Passage:** the user turn is split on blank lines. The passage is the longest block. If that block has
   fewer than 80 words, the whole turn minus the instruction line(s) is tried instead.
3. **Instruction:** the user text outside the passage must match the edit-request pattern
   `polish|proofread|revise|rewrite|re-write|paraphrase|rephrase|improve|edit|refine|correct grammar|fix grammar|
   make … (academic|formal|professional|fluent)`, case-insensitive.
4. **Passage quality:** the passage must pass the paper's own `ok_para` rule from `scripts/rewrite.py`
   (no LLM mentions, at most 3% digits, no math/table/figure/URL markers, starts with a capital letter and
   ends with a period). The primary length window is **80–200 words**, as in §5.13. A sensitivity run uses 80–400.
5. **Reply:** a leading preamble line is removed if it is at most 30 words and ends with a colon. Trailing
   explanation blocks are removed from the first line that starts with "Changes", "Explanation", "Note",
   "I made", "I have", "In this revision" or "Revisions". The result must contain 0.5–2.0× the
   passage's word count. Otherwise the pair is excluded as "reply is not one edited passage".
6. **Deduplication:** pairs with identical normalized passages count once (earliest timestamp kept).
7. Every exclusion is counted by reason and reported in `selection.json`.

## Academic filter

- **Primary:** the instruction mentions `paper|manuscript|abstract|journal|conference|thesis|dissertation|
  research|scientific|academic`, or the passage contains a citation pattern (`[n]`, `et al.`, `(Name, 2019)`).
- **Sensitivity:** all English editing pairs.

## Request buckets (first match wins)

grammar-only (`grammar|proofread|spelling|typo` and none of `polish|improve|rewrite|paraphrase|rephrase`)
→ rewrite (`rewrite|re-write|paraphrase|rephrase`) → academic (`academic|paper|journal|conference|scientific|formal`)
→ polish (everything else).

## Strata

Model family: GPT-3.5 vs GPT-4, from WildChat's `model` field. Period: quarter of the conversation's
timestamp. Cells with fewer than 30 pairs are reported but not interpreted.

## Features and statistics: identical to §5.13

The functions `counts`, `rates`, `boot` and `lr`, and the `METRICS` list, are loaded at run time from
`scripts/calib.py` at baseline commit `40b0119` by parsing that file, not by copying it. The marker list is
`results/excess_markers.json`. The corpus reference is `corpus_lr` from `results/rewrite_calibration.json`.

For each stratum and bucket the run reports: pooled rates of the original passages and of the edits, a
paragraph-bootstrap 95% CI (B=500, seed 0), the log ratio per feature, Spearman correlation with the corpus
log ratios, sign agreement on features whose corpus shift exceeds 0.1, and the median share of the corpus
shift. These match the existing calibration table column for column.

## Primary readout (fixed now)

Three features carry the paper's open question: functional riders (`func`), interpretive riders (`core`),
and *Additionally/Notably* openers (`addnot`). The result is "reproduced" for a stratum if all three log
ratios are positive with bootstrap CIs excluding zero. It is "not reproduced" if any is negative with a CI
excluding zero. Anything else is "inconclusive".

## Deviations

1. **2026-10-19, before any feature was computed.** A dry run on one of the 14 shards showed that rule 3
   admitted conversations that are not edit requests: creative-writing prompts and code questions where an
   edit verb appears somewhere in the text, and replies that are new content. Two rules were tightened after
   reading kept *pairs* only; no feature, rate or outcome had been computed. (a) The instruction must be
   at most 80 words, and the edit verb must govern a text object (`this/these/the following/my/it/below/
   above/text/paragraph/passage/abstract/sentence/essay/introduction/section/writing/grammar` within 60
   characters) or be "make it more academic/formal/professional/fluent". (b) The reply must keep the
   passage's content: the word-sequence similarity (difflib ratio, lowercased) must be at least 0.3.
2. **2026-10-19, before any feature was computed.** The full selection (14 shards) kept 283 pairs, only 16
   of them academic, because a request written in the same block as the passage ("Polish this. We
   propose…") was never split from it. When the user turn is a single block, the first or the last
   sentence is now taken as the instruction if it contains an edit verb (after the existing
   "text up to a colon" rule). The counts before and after this change are both reported.
3. **2026-10-19, before any feature was computed.** After deviation 2 the primary (academic) set had 28
   pairs, below the 30-pair minimum the protocol sets for interpretation. Pairs are now taken from any
   user→assistant exchange, not only the first, with at most one pair per conversation (its first
   valid exchange). If the academic set is still under 30, the all-English editing set becomes the
   primary analysis and the academic set is reported as a labelled subset. That rule is fixed here,
   before seeing any outcome.
4. **2026-10-19, AFTER features had been computed (a reporting-rule correction, not a pre-analysis
   change).** At the lead's request the fallback rule in deviation 3
   (all-English pairs become primary if the academic set has fewer than 30 pairs) is withdrawn. Widening the
   population changes what is being estimated. **The academic set stays the primary analysis.** All-English
   pairs are a separate, exploratory mechanism result. The correction changes no estimate: the rule was
   never triggered, because the academic set has 64 pairs at 80–200 words.
