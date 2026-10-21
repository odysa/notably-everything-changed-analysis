# Before/after-AI analysis reproduction bundle

This archive contains tracked analysis code, aggregate results, pseudonymous
numeric features, protocols and checksums. It contains no source PDFs,
extracted paper prose or WildChat conversation/pair text. `MANIFEST.json`
lists every included file and its SHA-256 checksum.

Core numerical checks (Python 3.14 with `requirements-validation.txt`):

    python scripts/corpus_validation/reproduce_statistics.py --report /tmp/stat-receipt.json
    python scripts/corpus_validation/verify_external_model.py --report /tmp/external-receipt.json

The first command recomputes and byte-compares 36 statistical CSVs. The second
independently recomputes the 158 external model scores from numeric inputs.
Neither command downloads or extracts PDFs. The snapshot checks numerical
analysis, not the upstream extraction or the study's causal interpretation.

The separate 1,200-paper CS replication includes its frozen sampling catalog,
protocol, selected source IDs, PDF hash receipts, per-paper numeric measures
and aggregate results in `results/review_round2/expansion*`. Source PDFs and
extracted text are not included. These records support an audit of the fixed
selection and summary statistics, but not a PDF-to-text rerun from this archive.

Optional upstream reruns have separate requirements. This bundle includes the
public paper URL list and hashed author mapping, but not source PDFs, extracted
prose, or the author's full metadata cache. PDF acquisition and extraction
need version-matched public sources and Poppler; linked-author regeneration
also needs public ACL metadata as described in `results/author_controls/README.md`.
The historical WildChat reference run used Python 3.12.12, PyArrow 26.0.0 for
selection, and
NumPy 2.5.4 plus SciPy 1.18.1 for analysis. It also needs the separately
downloaded 14 WildChat-1M parquet files at the revision pinned in
`results/historical_calibration/manifest.json`; those files are not bundled.
The two scripts are under `scripts/historical_calibration/`. The core
requirements file does not cover this optional workflow. Private conversations
and derived pair text are deliberately absent from this archive.
