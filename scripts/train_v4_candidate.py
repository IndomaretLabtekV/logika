"""Fit one fixed candidate with the same ten folds as the public 0.55845 ensemble."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, EXCLUDED, fit_candidate, metrics, predict_bundle
from spline_model import fit as fit_spline, predict as predict_spline, FEATURES

DESTINATION = OUT / 'v4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    candidate = args.candidate
    feature_file = ROOT / 'data/processed/application_features_v4.parquet'
    full = pd.read_parquet(feature_file)
    data = full[full.split.eq('train')].reset_index(drop=True)
    test = full[full.split.eq('test')].reset_index(drop=True)
    y = data.target.astype(int)
    if candidate == 'spline':
        spec = {'name': 'spline', 'family': 'spline', 'columns': [c for c in FEATURES if c in data],
                'params': {'C': .05, 'n_knots': 5, 'degree': 3}}
    else:
        prefix, iterations = candidate.rsplit('_', 1)
        specs = json.loads((DESTINATION / 'definitions.json').read_text())
        definition = next(s for s in specs if s['name'] == prefix)
        assert int(iterations) in definition['stages']
        spec = dict(definition, iterations=int(iterations))
    assert not set(spec['columns']) & EXCLUDED
    baseline_oof = pd.read_parquet(OUT / 'tenfold/tenfold_oof.parquet')
    baseline_test = pd.read_parquet(OUT / 'tenfold/tenfold_test_predictions.parquet')
    assert baseline_oof.application_id.equals(data.application_id)
    assert baseline_oof.target.astype(int).equals(y)
    assert baseline_test.application_id.equals(test.application_id)
    recipe = json.loads((OUT / 'tenfold/locked_recipe.json').read_text())['recipe']
    baseline = sum(w * baseline_oof[n] for n, w in recipe.items()).to_numpy()
    predicted = np.zeros(len(test))
    oof = np.full(len(data), np.nan)
    rows = []
    model_dir = DESTINATION / 'final_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_hash = hashlib.sha256(feature_file.read_bytes()).hexdigest()
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=9173).split(data, y))
    def train_one(fold, ti, vi):
        start = time.monotonic()
        model_file = model_dir / f'fold{fold}_{candidate}.joblib'
        signature = hashlib.sha256(json.dumps({'spec': spec, 'feature_sha256': feature_hash,
                                               'train_id_sha256': hashlib.sha256(ti.tobytes()).hexdigest(),
                                               'seed': 2026 + fold}, sort_keys=True).encode()).hexdigest()
        if model_file.exists():
            bundle = joblib.load(model_file)
            assert bundle.get('fit_signature') == signature, 'Cached model has a different configuration.'
        else:
            print(f'Training fold {fold}/10: {candidate}', flush=True)
            if candidate == 'spline':
                bundle = fit_spline(data, ti)
            else:
                bundle, _, _ = fit_candidate(spec, data, ti, None, spec['columns'], iterations=spec['iterations'], seed=2026+fold)
            bundle['fit_signature'] = signature
            joblib.dump(bundle, model_file, compress=3)
        predictor = predict_spline if candidate == 'spline' else predict_bundle
        pv = predictor(bundle, data.iloc[vi])
        pt = predictor(bundle, test)
        return fold, vi, pv, pt, time.monotonic() - start

    # Tree folds are independent. Each model uses eight threads.
    with ThreadPoolExecutor(max_workers=1 if candidate == 'spline' else 2) as executor:
        futures = [executor.submit(train_one, fold, ti, vi) for fold, (ti, vi) in enumerate(folds, 1)]
        for future in as_completed(futures):
            fold, vi, pv, pt, seconds = future.result()
            oof[vi] = pv
            predicted += pt / 10
            row = {'fold': fold, 'candidate': candidate, **metrics(y.iloc[vi], pv)}
            rows.append(row)
            print(json.dumps(dict(row, seconds=round(seconds, 1))), flush=True)
            pd.DataFrame({'application_id': data.application_id, 'target': y, candidate: oof}).to_parquet(DESTINATION / f'final_oof_{candidate}.parquet', index=False)
            pd.DataFrame({'application_id': test.application_id, candidate: predicted}).to_parquet(DESTINATION / f'final_test_{candidate}.parquet', index=False)
    assert np.isfinite(oof).all() and np.isfinite(predicted).all()
    report = {'candidate': candidate, 'definition': spec, 'feature_sha256': feature_hash,
              'fold_count': 10, 'fold_seed': 9173, 'model_seed': '2026 + fold',
              'metrics': metrics(y, oof), 'fold_metrics': rows,
              'baseline': metrics(y, baseline),
              'blends': {str(w): metrics(y, (1-w)*baseline + w*oof) for w in [.25, .5]}}
    (DESTINATION / f'final_metrics_{candidate}.json').write_text(json.dumps(report, indent=2))
    print('Final:', json.dumps({k: v for k, v in report.items() if k in ['candidate', 'metrics', 'baseline', 'blends']}), flush=True)


if __name__ == '__main__':
    main()
