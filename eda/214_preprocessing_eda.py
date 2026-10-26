"""CPU-only EDA and fit-partition preprocessing. No model training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata

import valid2 as v
from common import ROOT
from strict_feature_pool import build_pool


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def numeric(frame, column):
    x = frame[column].cast(pl.Float64).to_numpy().copy()
    x[~np.isfinite(x)] = np.nan
    return x


def fit_preprocessor(frame, columns, fitting_ids):
    ids = list(fitting_ids)
    assert ids and len(ids) == len(set(ids)) and set(ids) <= set(frame['application_id'])
    fit = frame.filter(pl.col('application_id').is_in(ids))
    kept, dropped, seen, categories, medians = [], {}, {}, {}, {}
    for c in columns:
        if c in v.CAT_COLS:
            categories[c] = sorted(fit[c].drop_nulls().unique().to_list())
            kept.append(c)
            continue
        x = numeric(fit, c)
        finite = np.isfinite(x)
        unique = np.unique(x[finite])
        # A constant observed value plus missingness is NOT a constant feature.
        if not len(unique) or (len(unique) == 1 and finite.all()):
            dropped[c] = {'reason': 'all_missing' if not len(unique) else 'constant'}
            continue
        normalized = np.where(finite, x, 0.).astype('<f8')
        normalized[normalized == 0] = 0.
        key = hashlib.sha256(finite.tobytes() + normalized.tobytes()).hexdigest()
        if key in seen:
            dropped[c] = {'reason': 'exact_fit_duplicate', 'representative': seen[key]}
            continue
        seen[key] = c
        kept.append(c)
        medians[c] = float(np.median(x[finite]))
    return {'kept_columns': kept, 'dropped_columns': dropped, 'categories': categories,
            'fit_medians_for_optional_dense_export': medians,
            'fitting_ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'policy': 'Fit-only constants/exact duplicates; main columns preferred; numeric infinities become null; retain NaN and missingness, no automatic imputation or correlation pruning.'}


def transform(frame, recipe):
    expressions = []
    for c in recipe['kept_columns']:
        if c in recipe['categories']:
            expressions.append(pl.when(pl.col(c).is_in(recipe['categories'][c])).then(pl.col(c)).otherwise(None).alias(c))
        else:
            x = pl.col(c).cast(pl.Float64)
            expressions.append(pl.when(x.is_finite()).then(x).otherwise(None).alias(c))
    meta = [c for c in ('application_id','customer_id','decision_datetime','target','is_train') if c in frame.columns]
    return frame.select(*meta, *expressions)


def ks(y, x):
    return float(ks_2samp(x[y == 1], x[y == 0]).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT/'results/eda_23okt/preprocessing')
    ap.add_argument('--pool', type=Path, default=ROOT/'results/v60/strict_tfm/pool_inner_f0')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train = pl.read_csv(ROOT/'data/train_applications.csv').sort('application_id')
    ids, labels = train['application_id'].to_numpy(), train['target'].to_numpy().astype(int)
    tr, outer = v.outer(labels, 42)[0]
    fi, st = v.inner(labels[tr], 42, 0)
    fit_ids, confirmation_ids = ids[tr[fi]], ids[tr[st]]
    pool = build_pool(fit_ids.tolist(), a.pool)
    frame = pl.read_parquet(pool['parquet'])
    manifest = pool['manifest']
    main_columns = manifest['main_columns']
    columns = main_columns + sorted(set(manifest['union_columns']) - set(main_columns))
    recipe = fit_preprocessor(frame, columns, fit_ids)
    prepared = transform(frame, recipe)
    assert prepared.height == 100000 and prepared['application_id'].n_unique() == 100000
    assert prepared.filter(~pl.col('application_id').is_in(fit_ids.tolist()))['target'].null_count() == 52000
    prepared.write_parquet(a.out/'features.parquet', compression='zstd')
    (a.out/'preprocessing.json').write_text(json.dumps(recipe, indent=2, allow_nan=False))
    index = {key: i for i, key in enumerate(frame['application_id'])}
    fit_rows = np.array([index[k] for k in fit_ids])
    confirmation_rows = np.array([index[k] for k in confirmation_ids])
    test_rows = np.flatnonzero(frame['is_train'].to_numpy() == 0)
    yfit, yconfirmation = labels[tr[fi]], labels[tr[st]]
    numeric_columns = [c for c in columns if c not in v.CAT_COLS]
    sample = np.random.default_rng(214).choice(len(fit_rows), 8000, replace=False)
    rank_matrix = np.zeros((len(sample), len(numeric_columns)))
    rows = []
    for j, c in enumerate(numeric_columns):
        x = numeric(frame, c)
        xf = x[fit_rows]
        observed = np.isfinite(xf)
        median = float(np.median(xf[observed])) if observed.any() else 0.
        fitted = np.where(observed, xf, median)
        # Direction is fixed using fitting labels; confirmation never chooses it.
        pos_mean = float(rankdata(fitted)[yfit == 1].mean())
        neg_mean = float(rankdata(fitted)[yfit == 0].mean())
        direction = 1 if pos_mean >= neg_mean else -1
        rc = rankdata(fitted[sample])
        sd = rc.std()
        if sd:
            rank_matrix[:, j] = (rc - rc.mean()) / sd
        rows.append({'feature': c, 'family': 'main' if c in main_columns else 'prefix',
                     'fit_missing': float((~observed).mean()),
                     'confirmation_missing': float(np.isnan(x[confirmation_rows]).mean()),
                     'test_missing': float(np.isnan(x[test_rows]).mean()),
                     'fit_unique_observed': len(np.unique(xf[observed])),
                     'fit_ks_median_imputed': ks(yfit, fitted),
                     'fit_missing_indicator_ks': ks(yfit, (~observed).astype(float)),
                     'direction': direction, 'median': median})
    main_indices = [j for j,c in enumerate(numeric_columns) if c in main_columns]
    corr = np.abs(rank_matrix.T @ rank_matrix[:, main_indices] / len(sample))
    for j, row in enumerate(rows):
        matches = corr[j].copy()
        if j in main_indices:
            matches[main_indices.index(j)] = -1
        best = int(matches.argmax())
        row['max_abs_spearman_to_main_sample'] = float(matches[best])
        row['closest_main'] = numeric_columns[main_indices[best]]
    table = pl.DataFrame(rows).sort('fit_ks_median_imputed', descending=True)
    table.write_csv(a.out/'feature_diagnostics.csv')
    # Confirm only a fitting-selected list. No selection/revision from confirmation.
    shortlist = table.filter(pl.col('family') == 'prefix').head(20)
    confirmations = []
    for row in shortlist.iter_rows(named=True):
        x = numeric(frame, row['feature'])[confirmation_rows]
        score = np.where(np.isfinite(x), x, row['median']) * row['direction']
        fpr = np.sort(score[yconfirmation == 0])
        tpr = np.sort(score[yconfirmation == 1])
        cuts = np.unique(score)
        signed = np.searchsorted(fpr, cuts, side='right')/len(fpr) - np.searchsorted(tpr, cuts, side='right')/len(tpr)
        confirmations.append({**row, 'confirmation_ks': ks(yconfirmation, score),
                              'confirmation_directional_ks': float(signed.max()),
                              'confirmation_rows': len(score)})
    pl.DataFrame(confirmations).write_csv(a.out/'fit_selected_prefix_confirmation.csv')
    drops = recipe['dropped_columns']
    summary = {'mode': 'EDA and preprocessing only; no model trained or inferred',
               'pool_columns': len(columns), 'prepared_columns': len(recipe['kept_columns']),
               'dropped_counts': {reason: sum(vv['reason'] == reason for vv in drops.values()) for reason in ['constant','all_missing','exact_fit_duplicate']},
               'fitting_rows': len(fit_ids), 'confirmation_rows': len(confirmation_ids),
               'outer_rows_not_evaluated': len(outer), 'outer_seed': 42,
               'correlation_fit_sample_rows': len(sample),
               'prefix_columns': sum(r['family'] == 'prefix' for r in rows),
               'prefix_abs_correlation_below_095': sum(r['family'] == 'prefix' and r['max_abs_spearman_to_main_sample'] < .95 for r in rows),
               'pool_manifest_sha256': sha(pool['manifest_path']), 'source_sha256': sha(__file__),
               'preprocessing_sha256': sha(a.out/'preprocessing.json'), 'features_sha256': sha(a.out/'features.parquet'),
               'limits': ['Univariate KS is not pooled OOF KS of a credit-risk model.',
                          'Correlation uses8000 fitting rows; low correlation does not prove incremental predictive value.',
                          'Confirmation is inner only; no features/weights chosen from outer labels.',
                          'Exact duplicate or constant removal is fitted per partition and must be refitted inside every future fold.',
                          'No claim that preprocessing reaches KS0.53-0.54; no submission.']}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for family in ['main','prefix']:
        data = [r for r in rows if r['family'] == family]
        ax.scatter([r['max_abs_spearman_to_main_sample'] for r in data], [r['fit_ks_median_imputed'] for r in data], s=12, alpha=.5, label=family)
    ax.set(xlabel='Maximum absolute rank correlation to main feature (fit sample)', ylabel='Univariate fitting KS', title='Feature information audit, not model OOF performance')
    ax.legend(); fig.tight_layout(); fig.savefig(a.out/'information_audit.png', dpi=140); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    print(pl.DataFrame(confirmations).select('feature','fit_ks_median_imputed','confirmation_ks','max_abs_spearman_to_main_sample').head(10))


if __name__ == '__main__':
    main()
