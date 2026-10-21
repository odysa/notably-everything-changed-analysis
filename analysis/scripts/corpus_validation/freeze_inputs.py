"""Freeze anonymous numerical features and audit legacy cohort membership.

No PDFs, excerpts, titles, or raw paper IDs enter the repository. Original caches
and published results are read-only. Numeric data retain their original row order.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze as legacy

TEXT_FIELDS = {'intro_first', 'abstract_first', 'title'}
CATEGORIES = {'intro_open', 'title_src', 'venue', 'era', 'abstract_open'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(corpus, pid):
    return hashlib.sha256((corpus + pid).encode()).hexdigest()[:12]


def clean_value(value):
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [clean_value(x) for x in value]
    raise TypeError('non-numeric value in numeric field')


def anonymous(row):
    result = {}
    for k, v in row.items():
        if k in TEXT_FIELDS or k == 'id':
            continue
        if isinstance(v, str):
            assert k == 'corpus' or k in CATEGORIES, k
            result[k] = v
        else:
            result[k] = clean_value(v)
    h = identity(row['corpus'], row['id'])
    result['id'] = result['paper_hash'] = h
    return result


def write_gzip(path, rows):
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                              compresslevel=6, mtime=0) as stream:
        with io.TextIOWrapper(stream, encoding='utf-8', newline='\n') as text:
            for row in rows:
                text.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/corpus_validation')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base_path, extra_path = args.workdir / 'features.jsonl', args.workdir / 'newfeatures.jsonl'
    base = [json.loads(line) for line in base_path.open()]
    extra = [json.loads(line) for line in extra_path.open()]
    indexed = {(r['corpus'], r['id']): r for r in base}
    assert len(indexed) == len(base)
    hashes = {identity(*key): key for key in indexed}
    assert len(hashes) == len(base), 'anonymous key collision'
    extra_keys = {(r['corpus'], r['id']) for r in extra}
    assert len(extra_keys) == len(extra) and extra_keys <= indexed.keys()
    with (args.workdir / 'all_features.csv').open() as f:
        legacy_matrix = list(csv.DictReader(f))
    legacy_keys = {(r['corpus'], r['id']) for r in legacy_matrix}
    with (ROOT / 'results/paper_scores.csv').open() as f:
        published = list(csv.DictReader(f))
    published_keys = {(r['corpus'], r['paper_hash']) for r in published}
    assert len(published_keys) == len(published)
    assert digest(args.workdir / 'lists.tsv') == digest(ROOT / 'results/paper_lists.tsv')
    # Reconstruct the original index independently from the per-paper score file.
    normalizers = {}
    for venue, (_, pre, post) in legacy.VENUES.items():
        pre_rows = [r for r in base if r['corpus'] in pre]
        both = [r for r in base if r['corpus'] in pre + post]
        normalizers[venue] = {f: (statistics.fmean(r[f] for r in pre_rows),
                                 max(statistics.pstdev(r[f] for r in both), 1e-3))
                              for f in legacy.STYLE_INDEX}
    maximum_error, checked = 0.0, 0
    for row in published:
        key = hashes[row['paper_hash']]
        record = indexed[key]
        assert row['corpus'] == record['corpus']
        assert int(row['words']) == record['words']
        assert (row['llm_topic'].lower() in ('true', '1', '1.0')) == bool(record['llm_topic'])
        venue = legacy.ALL_ERAS[record['corpus']]
        score = statistics.fmean((record[f] - normalizers[venue][f][0]) /
                                 normalizers[venue][f][1] for f in legacy.STYLE_INDEX)
        errors = [abs(score-float(row['style_index']))]
        errors += [abs(record[f]-float(row[f])) for f in legacy.STYLE_INDEX]
        maximum_error = max(maximum_error, *errors)
        assert max(errors) <= 0.000500001, 'published score or feature mismatch'
        checked += 1
    membership = []
    for row in base:
        c, pid = row['corpus'], row['id']
        membership.append(dict(corpus=c, paper_hash=identity(c, pid), year=int(re.search(r'\d{4}', c)[0]),
                               source_scope='starred' if c.startswith(('colm', 'eurosys')) else 'main',
                               legacy_all_features=(c, pid) in legacy_keys,
                               legacy_paper_scores=(c, identity(c, pid)) in published_keys,
                               new_features_available=(c, pid) in extra_keys,
                               pdf_present=(args.workdir / 'pdfs' / c / (pid+'.pdf')).exists(),
                               text_present=(args.workdir / 'text' / c / (pid+'.json')).exists()))
    with (args.out / 'cohort_membership.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(membership[0])); writer.writeheader(); writer.writerows(membership)
    for filename, records in [('features_numeric.jsonl.gz', base), ('newfeatures_numeric.jsonl.gz', extra)]:
        write_gzip(args.out / filename, (anonymous(r) for r in records))
    summary = dict(
        source_hashes={name: digest(args.workdir / name) for name in
                       ('features.jsonl', 'newfeatures.jsonl', 'all_features.csv', 'lists.tsv')},
        published_score_sha256=digest(ROOT / 'results/paper_scores.csv'),
        corpus_counts_sha256=digest(ROOT / 'results/corpus_counts.json'),
        legacy_code_hashes={name: digest(ROOT / 'scripts' / name) for name in
                            ('analyze.py', 'stats.py', 'stats2.py', 'features.py', 'extract.py', 'newfeat.py')},
        rows=len(base), corpus_cells=len({r['corpus'] for r in base}),
        new_feature_rows=len(extra), legacy_matrix_rows=len(legacy_matrix), published_score_rows=len(published),
        published_rows_checked=checked, maximum_published_rounding_difference=maximum_error,
        corpus_counts=dict(sorted(collections.Counter(r['corpus'] for r in base).items())),
        absent_legacy_matrix=dict(collections.Counter(r['corpus'] for r in membership if not r['legacy_all_features'])),
        absent_published_scores=dict(collections.Counter(r['corpus'] for r in membership if not r['legacy_paper_scores'])),
        missing_pdf=sum(not r['pdf_present'] for r in membership), missing_text=sum(not r['text_present'] for r in membership),
        identifier='id and paper_hash are SHA256(corpus + original id)[:12], original row order retained',
        removed_text_fields=sorted(TEXT_FIELDS),
        frozen_hashes={name: digest(args.out / name) for name in
                       ('features_numeric.jsonl.gz', 'newfeatures_numeric.jsonl.gz', 'cohort_membership.csv')})
    expected = json.loads((ROOT / 'results/corpus_counts.json').read_text())
    assert summary['corpus_counts'] == expected
    (args.out / 'frozen_inputs.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('rows', 'corpus_cells', 'new_feature_rows', 'legacy_matrix_rows',
                                              'published_score_rows', 'published_rows_checked',
                                              'maximum_published_rounding_difference', 'absent_legacy_matrix',
                                              'absent_published_scores', 'missing_pdf', 'missing_text')}, indent=2))


if __name__ == '__main__':
    main()
