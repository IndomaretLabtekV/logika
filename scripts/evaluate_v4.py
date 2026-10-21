"""Compare a fixed, small set of new candidates against the version-3 OOF recipe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, EXCLUDED, fit_candidate, metrics
from refine_ks import stage_probability

DESTINATION = OUT / 'v4'


def definitions(columns, added):
    richer = list(dict.fromkeys(columns + added))
    return [
        {'name': 'v3_cat4_balanced', 'family': 'cat', 'params': {'depth': 4, 'l2_leaf_reg': 25, 'class_weights': [1, 6]}, 'columns': columns, 'stages': [700, 1100]},
        {'name': 'v3_cat6_balanced', 'family': 'cat', 'params': {'depth': 6, 'l2_leaf_reg': 35, 'class_weights': [1, 6]}, 'columns': columns, 'stages': [500, 800]},
        {'name': 'v4_cat5_balanced', 'family': 'cat', 'params': {'depth': 5, 'l2_leaf_reg': 25, 'class_weights': [1, 6]}, 'columns': richer, 'stages': [500, 800]},
        {'name': 'v4_lgb31_balanced', 'family': 'lgb', 'params': {'num_leaves': 31, 'min_child_samples': 200, 'reg_lambda': 40, 'scale_pos_weight': 6}, 'columns': richer, 'stages': [250, 400]},
        {'name': 'v4_lgb15_balanced', 'family': 'lgb', 'params': {'num_leaves': 15, 'min_child_samples': 160, 'reg_lambda': 25, 'scale_pos_weight': 6}, 'columns': richer, 'stages': [400, 700]},
    ]


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(ROOT / 'data/processed/application_features_v4.parquet')
    data = full[full.split.eq('train')].reset_index(drop=True)
    test = full[full.split.eq('test')].reset_index(drop=True)
    reference = json.loads((OUT / 'tenfold/locked_recipe.json').read_text())
    columns = next(iter(reference['definitions'].values()))['columns']
    added = [c for c in data if c.startswith('v4_') and data[c].nunique(dropna=True) > 1]
    specs = definitions(columns, added)
    for spec in specs:
        assert not set(spec['columns']) & EXCLUDED
    (DESTINATION / 'definitions.json').write_text(json.dumps(specs, indent=2))
    old = pd.read_parquet(OUT / 'tenfold/selection_oof.parquet')
    old_test = pd.read_parquet(OUT / 'tenfold/selection_test_predictions.parquet')
    assert old.application_id.equals(data.application_id) and old.target.astype(int).equals(data.target.astype(int))
    assert old_test.application_id.equals(test.application_id)
    recipe = reference['recipe']
    oof = pd.DataFrame({'application_id': data.application_id, 'target': data.target,
                        'public_055845': sum(w * old[n] for n, w in recipe.items())})
    predictions = pd.DataFrame({'application_id': test.application_id,
                               'public_055845': sum(w * old_test[n] for n, w in recipe.items())})
    for spec in specs:
        for stage in spec['stages']:
            name = f'{spec["name"]}_{stage}'
            oof[name] = np.nan
            predictions[name] = 0.0
    y = data.target.astype(int)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(data, y))
    fold_rows = []

    def train_one(fold, spec, ti, vi):
        started = time.monotonic()
        print(f'Training fold {fold}/3: {spec["name"]}, {len(spec["columns"])} features', flush=True)
        bundle, _, _ = fit_candidate(spec, data, ti, None, spec['columns'], iterations=max(spec['stages']), seed=2026)
        values = {}
        for stage in spec['stages']:
            name = f'{spec["name"]}_{stage}'
            pv = stage_probability(bundle, data.iloc[vi], stage)
            pt = stage_probability(bundle, test, stage)
            values[name] = (pv, pt, metrics(y.iloc[vi], pv))
        joblib.dump(bundle, DESTINATION / f'fold{fold}_{spec["name"]}.joblib', compress=3)
        return fold, spec['name'], vi, values, time.monotonic() - started

    tasks = [(fold, spec, ti, vi) for fold, (ti, vi) in enumerate(folds, 1) for spec in specs]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(train_one, *task) for task in tasks]
        for future in as_completed(futures):
            fold, name, vi, values, elapsed = future.result()
            for stage_name, (pv, pt, score) in values.items():
                oof.loc[vi, stage_name] = pv
                predictions[stage_name] += pt / 3
                fold_rows.append({'fold': fold, 'name': stage_name, **score})
            oof.to_parquet(DESTINATION / 'selection_oof.parquet', index=False)
            predictions.to_parquet(DESTINATION / 'selection_test_predictions.parquet', index=False)
            pd.DataFrame(fold_rows).to_csv(DESTINATION / 'fold_metrics.csv', index=False)
            print(json.dumps({'completed': name, 'fold': fold, 'seconds': round(elapsed, 1), 'scores': {n: v[2]['ks'] for n, v in values.items()}}), flush=True)
    assert oof.notna().all().all()
    rows = []
    for name in oof.columns[2:]:
        per_fold = [metrics(y.iloc[vi], oof[name].iloc[vi])['ks'] for _, vi in folds]
        rows.append({'name': name, **metrics(y, oof[name]), 'mean_fold_ks': float(np.mean(per_fold)), 'fold_ks': per_fold})
    rows.sort(key=lambda r: r['mean_fold_ks'], reverse=True)
    (DESTINATION / 'candidate_summary.json').write_text(json.dumps(rows, indent=2))
    print('Completed three-fold comparison:', json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
