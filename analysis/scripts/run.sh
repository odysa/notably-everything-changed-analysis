#!/usr/bin/env bash
# Re-run the whole study. Usage: ./run.sh <workdir>   (workdir holds PDFs/text; ~4 GB)
# Needs: python3, curl, poppler (pdftotext, pdfinfo); matplotlib+numpy for figures;
# ~/.claude/skills/roast-paper/scripts/_common.py (PDF loader + prose extractor).
set -euo pipefail
WD=${1:?workdir}; HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
mkdir -p "$WD"
[ -f "$WD/lists.tsv" ] || cp "$ROOT/results/paper_lists.tsv" "$WD/lists.tsv"   # or: python3 -I build_lists.py "$WD"
python3 -I "$HERE/download.py" "$WD"
python3 -I "$HERE/extract.py" "$WD" --workers 8
python3 -I "$HERE/analyze.py" "$WD" "$ROOT/results"
python3 -I "$HERE/pairs.py" "$WD" "$ROOT/results"
python3 -I "$HERE/excess.py" "$WD" "$ROOT/results"
python3 -I "$HERE/tables.py" "$ROOT/results"
python3 "$HERE/figures.py" "$ROOT/results" "$ROOT/figures"
python3 "$HERE/stats.py" "$WD" "$ROOT/results" A B C D E F G H P   # needs scipy, pandas, statsmodels, scikit-learn
python3 "$HERE/newfeat.py" "$WD" "$ROOT/results"          # second batch of features (§5.11)
python3 "$HERE/stats2.py" "$WD" "$ROOT/results"           # robustness checks (§5.12)
python3 "$HERE/detect_eval.py" "$WD" "$ROOT/results"       # detector accuracy (site section, detect_eval.md)
python3 "$HERE/webdata.py" "$WD" "$ROOT/results" "$ROOT/site/data.json"
python3 -I "$HERE/sitejson.py" "$WD" "$ROOT/results" "$ROOT/site" && python3 "$ROOT/site/build.py"
python3 "$ROOT/paper/md_tables.py"                         # refresh paper.md statistics tables
(cd "$ROOT/paper/arxiv" && python3 make_tables.py)        # refresh LaTeX tables; then compile with tectonic
# Optional, needs the Claude Code CLI: LLM-rewrite calibration (§5.13)
# python3 -I "$HERE/rewrite.py" sample "$WD"; for m in haiku sonnet; do for p in grammar polish improve rewrite draft; do python3 -I "$HERE/rewrite.py" run "$WD" $m $p; done; done
# python3 "$HERE/calib.py" "$WD" "$ROOT/results" "$ROOT/figures"
