"""Nested LightGBM tuning; real fitting is restricted to Kaggle."""
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp


BASE = dict(objective='binary', metric='auc', learning_rate=0.02,
            num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            verbosity=-1, num_threads=min(os.cpu_count() or 4, 8),
            deterministic=True, force_col_wise=True)
CANDIDATES = {
    'control': {},
    'shallow': dict(num_leaves=7, max_depth=3, min_data_in_leaf=200,
                    feature_fraction=0.4, lambda_l2=10.0),
    'leaves15': dict(num_leaves=15),
    'leaves63': dict(num_leaves=63),
    'leaf50': dict(min_data_in_leaf=50),
    'leaf400': dict(min_data_in_leaf=400),
    'features80': dict(feature_fraction=0.8),
    'features30': dict(feature_fraction=0.3),
    'l2_30': dict(lambda_l2=30.0),
    'l1_5': dict(lambda_l1=5.0),
    'smooth20': dict(path_smooth=20.0),
    'extra': dict(extra_trees=True),
}


def ks(y, p):
    assert len(y) == len(p) and np.isfinite(p).all()
    assert set(np.unique(y)) == {0, 1}
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n)
             for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        train = np.setdiff1d(np.arange(len(y)), valid)
        assert not np.intersect1d(train, valid).size
        yield train, valid


def run(X, y, test, lgb, out, rounds=4000, patience=150):
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 11 * 3600
    cats = X.select_dtypes('category').columns.tolist()
    oof = {n: np.full(len(y), np.nan) for n in ('control', 'tuned')}
    test_pred = {n: np.zeros(len(test)) for n in oof}
    fold_id = np.full(len(y), -1)
    trials, selected = [], []
    for fold, (train, valid) in enumerate(splits(y, 5, 42)):
        # Outer labels are never supplied to early stopping or model selection.
        scores = {}
        for name, changes in CANDIDATES.items():
            inner_pred = np.full(len(train), np.nan)
            iterations = []
            for inner, (fit, stop) in enumerate(splits(y[train], 3, 170 + fold)):
                if time.monotonic() > deadline:
                    raise TimeoutError('Partial fold files saved; no complete submission produced.')
                params = dict(BASE, **changes, seed=42 + inner)
                dfit = lgb.Dataset(X.iloc[train[fit]], y[train[fit]], categorical_feature=cats)
                dstop = lgb.Dataset(X.iloc[train[stop]], y[train[stop]], categorical_feature=cats,
                                    reference=dfit)
                model = lgb.train(params, dfit, rounds, valid_sets=[dstop],
                                  callbacks=[lgb.early_stopping(patience, verbose=False)])
                inner_pred[stop] = model.predict(X.iloc[train[stop]], num_iteration=model.best_iteration)
                iterations.append(int(model.best_iteration))
            score = ks(y[train], inner_pred)
            count = max(1, int(np.median(iterations)))
            scores[name] = (score, count)
            trials.append(dict(fold=fold, candidate=name, inner_ks=score, rounds=count))
            pd.DataFrame(trials).to_csv(out / 'inner_trials.csv', index=False)
            print(f'fold {fold} {name}: inner KS {score:.5f}, rounds {count}', flush=True)
        winner = max(scores, key=lambda n: scores[n][0])
        selected.append(dict(fold=fold, candidate=winner, rounds=scores[winner][1]))
        for policy, name in [('control', 'control'), ('tuned', winner)]:
            dtrain = lgb.Dataset(X.iloc[train], y[train], categorical_feature=cats)
            model = lgb.train(dict(BASE, **CANDIDATES[name], seed=42), dtrain,
                              scores[name][1], valid_sets=[], callbacks=[])
            pv, pt = model.predict(X.iloc[valid]), model.predict(test)
            assert np.isfinite(pv).all() and np.isfinite(pt).all()
            oof[policy][valid] = pv
            test_pred[policy] += pt / 5
            np.savez(out / f'{policy}_fold{fold}.npz', rows=valid, prediction=pv, test=pt)
        fold_id[valid] = fold
        pd.DataFrame(selected).to_csv(out / 'selected.csv', index=False)
        print(f'Completed outer fold {fold}', flush=True)
    assert (fold_id >= 0).all()
    for p in oof.values():
        assert np.isfinite(p).all()
    return oof, test_pred, fold_id


def evaluate(y, oof, folds, out, repeats=500):
    delta = ks(y, oof['tuned']) - ks(y, oof['control'])
    rng = np.random.default_rng(2026)
    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    bootstrap = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        bootstrap.append(ks(y[idx], oof['tuned'][idx]) - ks(y[idx], oof['control'][idx]))
    by_fold = [ks(y[folds == k], oof['tuned'][folds == k]) -
               ks(y[folds == k], oof['control'][folds == k]) for k in range(5)]
    result = dict(control_ks=ks(y, oof['control']), tuned_ks=ks(y, oof['tuned']),
                  delta=delta, paired_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  fold_deltas=by_fold,
                  promotion_candidate=bool(delta > .002 and sum(d > 0 for d in by_fold) >= 4),
                  caveat='Bootstrap is conditional on fitted models; public score is unverified.')
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(5), by_fold)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set(title='Paired outer-fold KS gain', xlabel='Fold', ylabel='Tuned minus control')
    axes[1].hist(bootstrap, bins=30)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set(title='Conditional paired bootstrap', xlabel='Pooled KS gain')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=140)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return result


def main(build_all, lgb):
    assert Path('/kaggle/input').exists(), 'Real model training is permitted only on Kaggle.'
    data = next(Path('/kaggle/input').rglob('train_applications.csv')).parent
    out = Path('/kaggle/working/v10')
    out.mkdir(parents=True, exist_ok=True)
    names = ['train_applications', 'test_applications', 'customer_profiles',
             'previous_applications', 'repayment_history', 'account_behavior',
             'bureau_observations', 'ecommerce_transactions']
    tables = {n: pl.read_csv(data / f'{n}.csv', try_parse_dates=True,
                            infer_schema_length=100000) for n in names}
    assert not set(tables['train_applications']['customer_id']) & set(tables['test_applications']['customer_id'])
    features = build_all(tables).sort('application_id')
    assert features['application_id'].n_unique() == features.height
    train, test = (features.filter(pl.col('is_train') == v) for v in (1, 0))
    assert train['customer_id'].n_unique() == train.height
    y = train['target'].to_numpy().astype(int)
    columns = [c for c in features.columns if c not in
               {'application_id', 'customer_id', 'decision_datetime', 'target', 'is_train'}]
    X, Xt = train.select(columns).to_pandas(), test.select(columns).to_pandas()
    for c in columns:
        if features[c].dtype == pl.String:
            categories = sorted(X[c].dropna().unique())
            X[c] = pd.Categorical(X[c], categories=categories)
            Xt[c] = pd.Categorical(Xt[c], categories=categories)
        else:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            Xt[c] = Xt[c].replace([np.inf, -np.inf], np.nan)
    manifest = dict(lightgbm=lgb.__version__, polars=pl.__version__, numpy=np.__version__,
                    columns=columns, candidates=CANDIDATES, base=BASE,
                    feature_sha256=FEATURE_SHA256,
                    train_ids_sha256=hashlib.sha256('\n'.join(train['application_id']).encode()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    train.select('application_id', 'target').write_parquet(out / 'train_ids.parquet')
    test.select('application_id').write_parquet(out / 'test_ids.parquet')
    oof, predictions, folds = run(X, y, Xt, lgb, out)
    train.select('application_id', 'target').with_columns(
        fold=pl.Series(folds), **{k: pl.Series(v) for k, v in oof.items()}
    ).write_parquet(out / 'oof.parquet')
    evaluate(y, oof, folds, out)
    for name, pred in predictions.items():
        assert len(pred) == test.height and np.isfinite(pred).all()
        assert ((pred >= 0) & (pred <= 1)).all()
        test.select('application_id').with_columns(default_probability=pl.Series(pred)).write_csv(
            out / f'submission_{name}.csv')
    print('Completed nested experiment. Public KS still requires a scored Kaggle submission.')
