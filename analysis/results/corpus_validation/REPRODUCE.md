# Reproduce the numerical validation

From the repository root, create a Python environment using the versions in
`requirements-validation.txt` (the reference run uses Python 3.14.5), then run:

```sh
python scripts/corpus_validation/reproduce_statistics.py --report /tmp/validation-receipt.json
```

This checks the frozen 9,472-row numerical input hash, runs normalization and
estimator contract checks, recomputes all five statistical cohorts with 500
bootstrap replicates in a temporary directory, and compares every released CSV
byte for byte. It does not overwrite any released result. The optional receipt
records output hashes and the runtime manifest. Floating-point/library changes
can cause a byte comparison to fail; investigate those differences rather than
silently accepting approximate agreement. The exact versions are pinned for
this reason.

A separately completed run can be checked with `--recomputed /path/to/output`.
The committed `statistics_reproduction.json` records the lead's independent
isolated rerun, separate from the worker's original generation.

The portable input contains pseudonymous paper hashes and extracted numeric
features, not paper text. It makes **numerical reanalysis** reproducible; it does
not independently validate the upstream extraction or reconstruct full PDFs.
Raw PDFs/text remain in a private durable cache. The extractor snapshot and
frozen input manifest record that upstream provenance. The exact-version source
sensitivity and linked-author analyses have separate reproduction instructions
and require public metadata/source downloads. Historical editing calibration
has its own selection protocol. None of these checks establish AI authorship
or a causal effect of AI.

The numerical reader uses only the `STYLE`/`INDEX9` and cohort-definition literals
from legacy scripts. Verification requires those actual definitions, the seed,
bootstrap count, frozen input hash, and all numerical CSVs to match. Whole-file
legacy-script hashes are preserved as provenance, with any difference recorded
explicitly in the receipt; a figure-title-only edit can change such a hash without
changing a numerical definition. No result difference is accepted on that basis.
The receipt also hashes the verification and analysis code actually used.


External model scores can be independently checked from numerical inputs:

```sh
python scripts/corpus_validation/verify_external_model.py --report /tmp/external-model-receipt.json
```

The site also has a DOM unit check (separate from visual browser review). With
Node and `jsdom@26.1.0` available to module resolution, run
`node scripts/corpus_validation/check_site.cjs`. It executes the inline script,
checks dynamic cards/charts and core handlers, and validates generated fragment
links. It makes no network requests and does not substitute for browser layout
inspection. `site_dom_review.json` records the checked HTML hash and versions.
