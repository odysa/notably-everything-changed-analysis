#!/usr/bin/env python3
"""Verify external scores using an independently written pooled-pre transform."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('cohorts', ROOT / 'scripts/stat_validation/validate.py')
    cohorts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cohorts)
    _, _, training, definitions = cohorts.load(ROOT, ROOT / 'scripts', ROOT / 'results/corpus_validation/features_numeric.jsonl.gz')
    training = training[training.era.isin(['pre', 'post'])]
    features = definitions['STYLE']
    external_path = ROOT / 'results/external_validation/features_numeric.jsonl'
    score_path = ROOT / 'results/external_validation/scores.csv'
    external = pd.read_json(external_path, lines=True)
    saved = pd.read_csv(score_path)
    assert external.paper_hash.tolist() == saved.paper_hash.tolist()
    assert external.year.tolist() == saved.year.tolist()
    baseline = training.loc[training.era == 'pre', features].to_numpy(float)
    mean, sd = baseline.mean(axis=0), baseline.std(axis=0)
    sd[sd == 0] = 1
    x_train = np.clip((training[features].to_numpy(float) - mean) / sd, -8, 8)
    x_test = np.clip((external[features].to_numpy(float) - mean) / sd, -8, 8)
    model = LogisticRegression(C=.3, max_iter=2000).fit(x_train, (training.era == 'post').astype(int))
    predicted = model.predict_proba(x_test)[:, 1]
    assert np.allclose(predicted, saved.score, atol=1e-12, rtol=0)
    auc = float(roc_auc_score(external.year >= 2024, predicted))
    expected = json.loads((ROOT / 'results/external_validation/summary.json').read_text())['results']['pooled']['auc']['estimate']
    assert abs(auc - expected) < 1e-12
    receipt = {'status': 'passed', 'training_n': len(training), 'external_n': len(external),
               'auc': auc, 'maximum_score_difference': float(np.max(np.abs(predicted - saved.score))),
               'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), external_path, score_path,
                                    ROOT / 'results/corpus_validation/features_numeric.jsonl.gz']},
               'scope': 'Numeric transform, model fit and prediction reproduction; not an independent PDF extraction or bootstrap reproduction.'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'PASS: {len(external)} external scores reproduced; AUC {auc:.6f}')


if __name__ == '__main__':
    main()
