# Notably, Everything Changed

A study of changes in the prose of computer science papers around 2024. The main corpus covers 9,472 papers across 12 venue/source groups. A separate, frozen check sampled 1,200 additional papers in CVPR, ICML, and NeurIPS. The observed changes in published writing do **not** establish that any author used AI or that AI caused the changes.

- [Read the paper](paper/paper.pdf) or [view the public website](https://odysa.github.io/notably-everything-changed/).
- `paper/` contains the LaTeX source and figures for the public preprint.
- `analysis/` contains the reproducible numeric analyses, fixed protocols, feature definitions, source IDs and hash receipts. Start with [its README](analysis/README.md).
- The [website source](https://github.com/odysa/notably-everything-changed) is maintained separately.

This repository is a readable checkout of the public [paper source](https://odysa.github.io/notably-everything-changed/downloads/paper-source.tar.gz) and [analysis reproduction bundle](https://odysa.github.io/notably-everything-changed/downloads/analysis-reproduction.tar.gz). Its analysis snapshot was frozen before this repository was created. `analysis/MANIFEST.json` records the hashes of all files from that snapshot.

## Data boundary

The source papers' PDFs and extracted prose, the full author metadata cache, and WildChat conversation text are not redistributed here. The numeric data and source identifiers are sufficient for the documented downstream checks. Re-running PDF extraction requires obtaining the cited public sources separately; see `analysis/README.md` for the limits of reproduction.

## Citation

Please cite the paper when using the analysis. The public website presents the results and their limitations in plain language.

## License

Analysis code is under the [Apache License 2.0](LICENSE). Original paper text, figures, documentation and derived numeric measurements are under [CC BY 4.0](LICENSE-CONTENT.md), with third-party material excluded.
