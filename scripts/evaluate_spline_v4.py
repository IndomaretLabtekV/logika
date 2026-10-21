"""Evaluate one fixed spline candidate on the same three folds as tree candidates."""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics
from spline_model import fit, predict, FEATURES


def main():
    dest = OUT / 'v4'
    full = pd.read_parquet(ROOT / 'data/processed/application_features_v4.parquet')
    data = full[full.split.eq('train')].reset_index(drop=True)
    test = full[full.split.eq('test')].reset_index(drop=True)
    config = json.loads((OUT / 'tenfold/locked_recipe.json').read_text())
    old = pd.read_parquet(OUT / 'tenfold/selection_oof.parquet')
    old_test = pd.read_parquet(OUT / 'tenfold/selection_test_predictions.parquet')
    assert old.application_id.equals(data.application_id)
    assert old.target.astype(int).equals(data.target.astype(int))
    assert old_test.application_id.equals(test.application_id)
    baseline = sum(w * old[n] for n, w in config['recipe'].items()).to_numpy()
    baseline_test = sum(w * old_test[n] for n, w in config['recipe'].items()).to_numpy()
    oof = np.full(len(data), np.nan)
    predicted = np.zeros(len(test))
    rows = []
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(data, data.target))
    for fold, (ti, vi) in enumerate(folds, 1):
        start = time.monotonic()
        bundle = fit(data, ti)
        oof[vi] = predict(bundle, data.iloc[vi])
        predicted += predict(bundle, test) / 3
        joblib.dump(bundle, dest / f'fold{fold}_spline.joblib', compress=3)
        score = metrics(data.target.iloc[vi], oof[vi])
        scores = {str(w): metrics(data.target.iloc[vi], (1-w)*baseline[vi] + w*oof[vi])['ks'] for w in [.25, .5]}
        rows.append({'fold': fold, **score})
        print(json.dumps({'fold': fold, 'seconds': round(time.monotonic()-start, 1), **score, 'blend_ks': scores}), flush=True)
    pd.DataFrame({'application_id': data.application_id, 'target': data.target, 'spline': oof}).to_parquet(dest / 'spline_oof.parquet', index=False)
    pd.DataFrame({'application_id': test.application_id, 'spline': predicted}).to_parquet(dest / 'spline_test_predictions.parquet', index=False)
    report = {'parameters': {'C': .05, 'n_knots': 5, 'degree': 3, 'feature_columns': bundle['columns']},
              'metrics': metrics(data.target, oof), 'folds': rows,
              'baseline': metrics(data.target, baseline),
              'blends': {str(w): metrics(data.target, (1-w)*baseline + w*oof) for w in [.25, .5]}}
    (dest / 'spline_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'parameters'}), flush=True)


if __name__ == '__main__':
    main()
