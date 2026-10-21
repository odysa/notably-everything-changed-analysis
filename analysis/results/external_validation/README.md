# Independent EMNLP corpus check

This bounded check was specified prospectively after the original study. Protocol commits `3cc4000` and `6d94e4a` precede metadata selection; sample/code commit `81107e6` precedes PDF downloads and external feature outcomes. It addresses transfer of the fixed exploratory feature set to one previously unexamined venue, not causal identification or new feature discovery.

`selection.json` fixes 40 anonymous sample hashes per year. Official pinned XML has no explicit long/short labels, so the eligible population is main research entries with authors and page metadata, excluding front matter. The 2019 volume is joint EMNLP-IJCNLP, whereas 2020, 2024 and 2025 are EMNLP main. Findings, workshops, demos, industry and tutorials are excluded. No download/processing failure is replaced. All selected source IDs were compared to the entire original paper list: zero overlap. This is neither author-disjoint nor field-disjoint from the original ACL sample.

The unchanged `extract.py` is imported after injecting the committed byte-identical private-helper snapshot as `_common`; original `features.py` computes the same STYLE33 values, including policy removal and the 1,500 body-word minimum. PDFs, XML, identifiers and extracted prose remain outside git. `retrieval.json` records anonymous PDF hashes and every processing status. `features_numeric.jsonl` contains only numeric features and anonymous hashes. `scores.csv` records untouched model applications. The training sample comprises the original ten-venue pre/post rows, excluding intermediate years and 2016 controls. Mean and population SD are estimated from pooled training pre rows only; there is no target-venue normalization or refitting.

Run with the scientific environment specified by `requirements-validation.txt` and Poppler on PATH:

```sh
python scripts/external_validation/emnlp.py fetch --cache /path/to/private/cache
python scripts/external_validation/emnlp.py select --cache /path/to/private/cache
# Freeze and review selection before retrieving PDFs or running:
python scripts/external_validation/emnlp.py run --cache /path/to/private/cache
```

Do not rerun selection to seek a more favorable sample. Source URLs and SHA256 hashes are recorded in `selection.json`; original extraction, feature and training-input hashes are in `summary.json`. A rerun uses cached raw inputs if available. The scripts write exclusively to this new result directory and the specified private cache, never the original study's outputs.

AUC is an era-discrimination measure, not an AI-writing probability. Feature gaps are post-minus-pre raw counts per 1,000 body words. Bootstrap intervals resample papers independently within each year and condition on these four selected venue-year populations; they do not cover global year shocks, author dependence, corpus choice, or topic/composition shifts. A single external venue and 160 selected papers cannot establish broad cross-field generalization. Original feature selection remains exploratory.

## Result

All 160 official PDFs downloaded and processed without errors. Two papers failed the fixed 1,500-body-word gate (one each in 2019 and 2025), leaving 158 unique eligible papers: 39/40/40/39 by year, 79 pre and 79 post. No replacement was made and all 33 features were finite.

The frozen classifier, trained on 3,949 original papers (1,297 pre and 2,652 post), achieves external pooled era AUC **0.891** (conditional paper-bootstrap 95% interval **0.836–0.934**). Pairwise AUCs are 0.863 (2019–2024), 0.937 (2019–2025), 0.842 (2020–2024), and 0.924 (2020–2025). The fixed three pooled raw mean gaps are positive: functional riders **+0.996** [0.758,1.240], core interpretive riders **+0.307** [0.214,0.406], and Additionally/Notably openers **+0.411** [0.313,0.513] per 1,000 body words. All individual year-pair gaps are positive too; full precision and intervals are in `summary.json`.

This supports transfer of the fixed era-associated style measurements to this sampled external venue. It does not resolve author, topic, venue-policy, or PDF-extraction confounding, and does not identify AI use. Source-ID disjointness is the only paper-overlap check: distinct IDs do not guarantee different underlying content or authors. The dated local protocol is a prospective specification for this bounded check, not an external preregistration.

`python scripts/external_validation/check.py` checks 160 status records, 158 unique eligible hashes, 79 papers per era, 40 frozen selections per year, finite complete STYLE33 data, unchanged provenance hashes, and count/schema consistency. It passes.

The runner refuses model evaluation if any selected paper has a download/extraction incompatibility or error, or if fewer than 160 processing statuses exist. It retains `retrieval.json` for diagnosis; threshold exclusions alone are allowed. The artifact check exercises both an injected error and an incomplete status list to verify refusal. The delivered run has zero errors, so this guard does not change its estimates and no model rerun was performed.
