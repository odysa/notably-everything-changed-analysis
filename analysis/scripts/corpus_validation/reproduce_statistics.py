#!/usr/bin/env python3
"""Recompute released statistics in isolation and compare every numerical CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional JSON verification receipt')
    parser.add_argument('--recomputed', type=Path, help='Verify a completed isolated run instead of recomputing')
    args = parser.parse_args()
    source = ROOT / 'results/corpus_validation/features_numeric.jsonl.gz'
    expected = ROOT / 'results/stat_validation'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != 'b85eb994dc0372093fa6da83719179d9a5a523b3361e09d921d93521af262ae0':
        raise SystemExit('Frozen numerical input hash mismatch')
    subprocess.run([sys.executable, str(ROOT / 'scripts/stat_validation/check.py')], cwd=ROOT, check=True)
    with (nullcontext(str(args.recomputed)) if args.recomputed else
          tempfile.TemporaryDirectory(prefix='before-after-ai-validation-')) as temp:
        output = Path(temp)
        if not args.recomputed:
            subprocess.run([sys.executable, str(ROOT / 'scripts/stat_validation/validate.py'),
                        '--features', str(source), '--out', str(output), '--bootstrap', '500'],
                       cwd=ROOT, check=True)
        actual_manifest = json.loads((output / 'manifest.json').read_text())
        released_manifest = json.loads((expected / 'manifest.json').read_text())
        for key in ('seed', 'bootstrap_B', 'source_sha256', 'definitions'):
            if actual_manifest[key] != released_manifest[key]:
                raise SystemExit(f'Analysis manifest mismatch: {key}')
        released = {p.name for p in expected.glob('*.csv')}
        generated = {p.name for p in output.glob('*.csv')}
        if released != generated:
            raise SystemExit(f'CSV inventory differs: {sorted(released ^ generated)}')
        mismatches = [name for name in sorted(released)
                      if (expected / name).read_bytes() != (output / name).read_bytes()]
        if mismatches:
            raise SystemExit(f'Numerical CSV bytes differ: {mismatches}; use pinned environment')
        receipt = {'status': 'passed', 'input_sha256': source_sha,
                   'bootstrap_replicates': 500, 'csv_files_equal': len(released),
                   'csv_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in sorted(released)},
                   'runtime_manifest': actual_manifest,
                   'original_definition_file_sha256': released_manifest['definition_sha256'],
                   'definition_file_hash_changes': {
                       name: {'original': digest, 'recomputed': actual_manifest['definition_sha256'][name]}
                       for name, digest in released_manifest['definition_sha256'].items()
                       if digest != actual_manifest['definition_sha256'][name]},
                   'verification_code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), ROOT / 'scripts/stat_validation/validate.py',
                                 ROOT / 'scripts/stat_validation/check.py']},
                   'scope': 'Numerical reanalysis only; no PDF extraction, author metadata downloads, or model calls.'}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'PASS: all {len(released)} released numerical CSVs reproduced exactly')


if __name__ == '__main__':
    main()
