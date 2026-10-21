"""CPU-only Kumo input audit; no model, GPU, cache rebuild, or test-distribution fit.

Run from repository root:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 taskset -c 0-11 nice -n 10 .venv/bin/python eda/153_kumo_input_audit.py
All supervised/descriptive feature statistics use seed42/fold0 INNER FIT only.
Held-out partitions are inspected only for structural identity/overlap and category
vocabulary coverage. Outputs are audit artifacts, not outer-selected candidates.
"""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from scipy.stats import rankdata, skew

from common import OUT, PQ, ROOT, TABLES
from valid2 import CAT_COLS, ID_COLS, fm_select, frames, inner, ks_nan, outer

PREFIX = "153_kumo_input_audit"


def save_csv(frame, suffix):
    frame.to_csv(OUT / f"{PREFIX}_{suffix}.csv", index=False)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    stat = path.stat()
    return {"path": str(path.relative_to(ROOT)), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": h.hexdigest()}


def family(name):
    for prefixes, label in [(('bc_', 'bp_', 'bh_', 'bz_'), 'bureau'),
                            (('rp_', 'rq_', 'sv_'), 'repayment'),
                            (('ac_', 'cf_'), 'account'), (('ec_', 'ec2_'), 'ecommerce'),
                            (('pp_',), 'price_premium'), (('pv_', 'ob_'), 'previous')]:
        if name.startswith(prefixes):
            return label
    if name in {'has_history', 'dti_verified', 'dti_total', 'income_verif_ratio',
                'debt_to_income', 'inst_to_avgbal', 'ecspend_to_income'}:
        return 'cross'
    return 'application_profile'


def signed_log1p(x):
    # Stateless Float64; inspect floating-point ties rather than assume exact ranks.
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def money_audit(numeric, logs):
    """Manual units audit against features.py, not a broad name-based transform."""
    groups = [
        ('features.py:19,28', 'IDR or IDR/month',
         'loan_amount proposed_monthly_installment declared_monthly_income income_per_head'),
        ('features.py:50-52,61-65,73,79', 'IDR balance or balance change',
         'bc_outstanding_max bc_revlimit_max bc_revbal_max bp_out_a bp_out_b bp_out_c bh_last_outstanding bh_outstanding_delta'),
        ('features.py:99,117-118', 'IDR obligation or installment',
         'rp_unpaid_amt rp_active_inst rp_remaining_oblig'),
        ('features.py:131-151', 'IDR balance, flow, or IDR/month slope',
         'ac_inflow_mean ac_outflow_mean ac_income_mean ac_income_std ac_income_min ac_avgbal_mean ac_minbal_min ac_minbal_mean ac_net_flow_mean ac_income_l3 ac_avgbal_l3 ac_last_close ac_avgbal_b ac_close_slope'),
        ('features.py:171-180,190', 'IDR spend or IDR/month',
         'ec_spend ec_aov ec_spend_30d ec_spend_90d ec_spend_per_month'),
        ('features.py:215-216,254-255', 'IDR requested, approved, or remaining',
         'pv_req_mean pv_appr_sum pv_appr_max ob_amt_90d ob_remaining'),
        ('features.py:291-293', 'IDR flow or minimum balance',
         'cf_out_l3 cf_free_min cf_minbal_l3'),
    ]
    annotations = {name: {'source_definition': source, 'units': units}
                   for source, units, names in groups for name in names.split()}
    assert set(annotations).issubset(set(numeric.feature))
    names = numeric.feature.str.contains(
        'income|amount|balance|outstanding|instal|limit|salary|principal|payment|cash', regex=True)
    review = numeric[['feature', 'selected100', 'max_abs', 'q01', 'q99']].copy()
    review['name_keyword_match'] = names
    review['manual_monetary_units'] = review.feature.isin(annotations)
    review['source_definition'] = review.feature.map(lambda c: annotations.get(c, {}).get('source_definition', ''))
    review['units'] = review.feature.map(lambda c: annotations.get(c, {}).get('units', 'not monetary: ratio/count/score/other'))
    review['inner_max_abs_gt100'] = review.max_abs > 100
    review['inner_tail_abs_gt100'] = review[['q01', 'q99']].abs().max(axis=1) > 100
    review['money_candidate'] = review.manual_monetary_units & review.inner_max_abs_gt100
    save_csv(review[review.name_keyword_match | review.manual_monetary_units], 'money_name_review')
    money = review[review.money_candidate].merge(logs, on='feature', validate='one_to_one')
    save_csv(money, 'money_candidates')
    checks = ['nan_preserved', 'sign_preserved', 'finite_preserved', 'nondecreasing_unique', 'strict_monotone_unique', 'rank_equal']
    result = {'selection_protocol': 'Manual monetary units from current features.py; inner-fit max_abs > 100; no held-out fit.',
              'columns': money.feature.tolist(),
              'selected100_columns': money.loc[money.selected100, 'feature'].tolist(),
              'n_candidates': len(money), 'n_selected100': int(money.selected100.sum()),
              'checks_all_true': money[checks].all().to_dict(),
              'rank_tie_findings': money.loc[~money.rank_equal,
                                            ['feature', 'unique_before', 'unique_after', 'ks_delta']].to_dict('records'),
              'max_abs_ks_delta': float(money.ks_delta.abs().max()),
              'excluded_keyword_columns': review.loc[review.name_keyword_match & ~review.manual_monetary_units, 'feature'].tolist(),
              'source_fingerprint_at_money_audit': fingerprint(ROOT / 'eda' / 'features.py'),
              'audit_script_fingerprint': fingerprint(Path(__file__).resolve()),
              'note': 'Audit of Float64 values, not assurance of model-internal precision; no exception repaired.'}
    (OUT / f'{PREFIX}_money.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('\nMONEY CANDIDATES (manual source units; inner-fit scale only):\n' +
          money[['feature', 'selected100', 'max_abs', 'rank_equal', 'unique_before', 'unique_after', 'ks_delta']].to_string(index=False))
    print('MONEY SUMMARY:', json.dumps(result, indent=2), flush=True)
    return result


def duplicate_audit(f, X, Xt, train, partitions):
    identities, overlaps = [], []
    predictor_frames = {"train": X, "test": Xt}
    predictor_frames.update({k: X.iloc[v] for k, v in partitions.items()})
    id_frames = {"train": train, "test": f.filter(pl.col("is_train") == 0)}
    id_frames.update({k: train[v] for k, v in partitions.items()})
    for split, ids in id_frames.items():
        d = predictor_frames[split]
        row_hash = pd.util.hash_pandas_object(d, index=False)
        identities.append({"split": split, "rows": len(d),
                           "duplicate_application_ids": ids.height - ids["application_id"].n_unique(),
                           "duplicate_customer_ids": ids.height - ids["customer_id"].n_unique(),
                           "duplicate_predictor_rows_exact": int(d.duplicated().sum()),
                           "duplicate_row_hashes": int(row_hash.duplicated().sum())})
    # Exact verification after hash candidate filtering; NaNs compare equal via drop_duplicates.
    pairs = list(itertools.combinations(["inner_fit", "inner_stop", "outer_valid", "test"], 2))
    for a, b in pairs:
        da, db = predictor_frames[a], predictor_frames[b]
        ha = pd.util.hash_pandas_object(da, index=False)
        hb = pd.util.hash_pandas_object(db, index=False)
        common = set(ha).intersection(hb)
        ca, cb = da.loc[ha.isin(common)], db.loc[hb.isin(common)]
        exact = len(ca.drop_duplicates()) + len(cb.drop_duplicates()) - len(pd.concat([ca, cb]).drop_duplicates())
        overlaps.append({"left": a, "right": b,
                         "application_id_overlap": len(set(id_frames[a]["application_id"]) & set(id_frames[b]["application_id"])),
                         "customer_id_overlap": len(set(id_frames[a]["customer_id"]) & set(id_frames[b]["customer_id"])),
                         "predictor_hash_overlap": len(common), "exact_predictor_overlap": exact})
    # Full official train/test must also be checked, not just the nested subsets.
    for key in ("application_id", "customer_id"):
        assert not (set(id_frames['train'][key]) & set(id_frames['test'][key]))
    save_csv(pd.DataFrame(identities), "duplicates")
    save_csv(pd.DataFrame(overlaps), "overlap")
    return identities, overlaps


def cutoff_audit(tables):
    apps = pl.concat([tables['train_applications'].with_columns(split=pl.lit('train')),
                      tables['test_applications'].with_columns(split=pl.lit('test'))], how='diagonal_relaxed')
    cut = apps.select('customer_id', 'split', cutoff='decision_datetime')
    checks = [('previous_applications', 'decision_datetime', False),
              ('repayment_history', 'due_date', False),
              ('repayment_history', 'last_payment_datetime', False),
              ('bureau_observations', 'available_datetime', False),
              ('bureau_observations', 'report_datetime', False),
              ('ecommerce_transactions', 'transaction_datetime', False),
              ('account_behavior', 'month', True)]
    rows = []
    for table, column, next_month in checks:
        d = tables[table].select('customer_id', column).join(cut, on='customer_id')
        timestamp = pl.col(column).dt.offset_by('1mo').cast(pl.Datetime) if next_month else pl.col(column)
        d = d.with_columns(observed=timestamp)
        for split in ('train', 'test'):
            s = d.filter(pl.col('split') == split)
            gap = s.group_by('customer_id').agg(
                gap_days=(pl.col('cutoff').first() - pl.col('observed').max()).dt.total_seconds() / 86400)
            rows.append({'table': table, 'timestamp': column + (' + 1 month' if next_month else ''),
                         'split': split, 'rows': s.height, 'customers': s['customer_id'].n_unique(),
                         'missing_timestamp_rows': s['observed'].null_count(),
                         'after_cutoff_rows': s.filter(pl.col('observed') > pl.col('cutoff')).height,
                         'median_last_gap_days': gap['gap_days'].median()})
    result = pd.DataFrame(rows)
    save_csv(result, 'cutoff')
    return result


def main():
    cache = PQ / 'features.parquet'
    paths = [cache] + [PQ / f'{t}.parquet' for t in TABLES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Requires existing caches; refuses rebuild: {missing}')
    f = pl.read_parquet(cache).sort('application_id')
    X, Xt, y, _, _ = frames(f)
    train = f.filter(pl.col('is_train') == 1)
    tr, va = outer(y, 42)[0]
    fi, st = inner(y[tr], 42, 0)
    fit, stop = tr[fi], tr[st]
    partitions = {'inner_fit': fit, 'inner_stop': stop, 'outer_valid': va}
    selected = fm_select(X, y, fit, 100)
    nums = [c for c in X if c not in CAT_COLS]
    sub = X.iloc[fit]
    yf = y[fit]
    thin = sub['has_history'].to_numpy() == 0
    print('PROTOCOL seed=42 fold=0; numeric/category/KS/missingness statistics: INNER FIT ONLY')
    print('No model fit; no GPU import; no fitting on test; no future-row injection performed.')
    print('rows:', {**{k: len(v) for k, v in partitions.items()}, 'test': len(Xt)},
          'predictors:', len(X.columns), 'numeric:', len(nums), 'selected:', len(selected), flush=True)

    category_rows, category_summary = [], []
    for c in CAT_COLS:
        vocabulary = sorted(sub[c].dropna().astype(str).unique())
        current_categories = list(X[c].cat.categories)
        category_summary.append({'feature': c, 'inner_cardinality': len(vocabulary),
                                 'frames_full_train_vocab_size': len(current_categories),
                                 'inner_missing': int(sub[c].isna().sum()),
                                 'inner_codes': sorted(sub[c].cat.codes.unique().tolist()),
                                 'full_train_only_vocab_not_in_inner': sorted(set(current_categories) - set(vocabulary))})
        for code, value in enumerate(current_categories):
            category_rows.append({'feature': c, 'label': value, 'current_float_code': float(code),
                                  'inner_count': int((sub[c] == value).sum()),
                                  'inner_vocab_code': vocabulary.index(value) if value in vocabulary else None})
        for name, frame in [('inner_stop', X.iloc[stop]), ('outer_valid', X.iloc[va]), ('test', Xt)]:
            # Coverage counts only: this does not alter/fill the inner-fit vocabulary.
            category_summary[-1][f'{name}_unseen_vs_inner'] = int((frame[c].notna() & ~frame[c].astype(str).isin(vocabulary)).sum())
            category_summary[-1][f'{name}_missing_or_unknown_full_train'] = int(frame[c].isna().sum())
    cat = pd.DataFrame(category_rows)
    save_csv(cat, 'categories')
    print('\nCATEGORY MAPPING (current numeric Kumo codes; alphabetical, not ordinal):\n' + cat.to_string(index=False))

    raw_rows, log_rows, missing_rows, examples = [], [], [], []
    for c in nums:
        x = sub[c].to_numpy(dtype=np.float64)
        finite = np.isfinite(x)
        v = x[finite]
        z = signed_log1p(x)
        transformed = z[finite]
        unique = np.unique(v)
        unique_z = signed_log1p(unique)
        raw_ks, log_ks = ks_nan(yf, x), ks_nan(yf, z)
        q = np.quantile(v, [0, .01, .25, .5, .75, .99, 1]) if len(v) else np.full(7, np.nan)
        raw_rows.append({'feature': c, 'family': family(c), 'selected100': c in selected,
                         'n_finite': int(finite.sum()), 'n_nan': int(np.isnan(x).sum()), 'n_inf': int(np.isinf(x).sum()),
                         'n_unique': len(unique), 'negative_fraction_finite': float((v < 0).mean()) if len(v) else np.nan,
                         'zero_fraction_finite': float((v == 0).mean()) if len(v) else np.nan,
                         'mean': float(v.mean()) if len(v) else np.nan, 'std': float(v.std()) if len(v) else np.nan,
                         'skew': float(skew(v, bias=False)) if len(unique) > 1 else np.nan,
                         **dict(zip(['min', 'q01', 'q25', 'q50', 'q75', 'q99', 'max'], q)),
                         'max_abs': float(np.max(np.abs(v))) if len(v) else np.nan, 'ks_inner': raw_ks})
        log_rows.append({'feature': c, 'nan_preserved': bool(np.array_equal(np.isnan(x), np.isnan(z))),
                         'sign_preserved': bool(np.array_equal(np.sign(v), np.sign(transformed))),
                         'finite_preserved': bool(np.isfinite(transformed).all()),
                         'nondecreasing_unique': bool((np.diff(unique_z) >= 0).all()),
                         'strict_monotone_unique': bool((np.diff(unique_z) > 0).all()),
                         'rank_equal': bool(np.array_equal(rankdata(v), rankdata(transformed))),
                         'unique_before': len(unique), 'unique_after': len(np.unique(transformed)),
                         'skew_after': float(skew(transformed, bias=False)) if len(unique) > 1 else np.nan,
                         'ks_raw': raw_ks, 'ks_signed_log1p': log_ks, 'ks_delta': log_ks - raw_ks})
        for label, mask in [('thin', thin), ('history', ~thin)]:
            missing_rows.append({'feature': c, 'family': family(c), 'selected100': c in selected,
                                 'segment': label, 'rows': int(mask.sum()),
                                 'missing_fraction': float(np.isnan(x[mask]).mean()),
                                 'all_missing': bool(np.isnan(x[mask]).all())})
        if c in ['declared_monthly_income', 'loan_amount', 'cf_free_min', 'cf_minbal_l3', 'pp_mean', 'rp_frac_dpd1']:
            for target in [0, 1]:
                for history in [0, 1]:
                    indices = np.flatnonzero((yf == target) & (sub['has_history'].to_numpy() == history))[:2]
                    for i in indices:
                        examples.append({'application_id': train['application_id'][int(fit[i])], 'target': target,
                                         'has_history': history, 'feature': c, 'raw': x[i], 'signed_log1p': z[i]})
    numeric, logs, miss = pd.DataFrame(raw_rows), pd.DataFrame(log_rows), pd.DataFrame(missing_rows)
    numeric['ks_rank_inner'] = numeric['ks_inner'].rank(method='first', ascending=False).astype(int)
    save_csv(numeric.sort_values('ks_inner', ascending=False), 'numeric')
    save_csv(logs, 'signed_log1p')
    save_csv(miss, 'missingness')
    save_csv(pd.DataFrame(examples), 'examples')
    families = numeric.groupby(['family', 'selected100']).agg(
        columns=('feature', 'size'), max_ks_inner=('ks_inner', 'max'), median_ks_inner=('ks_inner', 'median')).reset_index()
    save_csv(families, 'families')
    money_result = money_audit(numeric, logs)
    bool_cols = ['nan_preserved', 'sign_preserved', 'finite_preserved', 'nondecreasing_unique',
                 'strict_monotone_unique', 'rank_equal']
    # Strict rank/KS failures are audit findings, not a reason to hide the results.
    assert logs[['nan_preserved', 'sign_preserved', 'finite_preserved']].all().all()
    # Explicit stress values cover NaN, signed zero, extreme finite magnitudes, and both signs.
    stress = np.array([np.nan, -np.finfo(float).max, -1e200, -1, -1e-300, -0., 0., 1e-300, 1, 1e200, np.finfo(float).max])
    zs = signed_log1p(stress)
    assert np.array_equal(np.isnan(stress), np.isnan(zs)) and np.isfinite(zs[1:]).all()
    assert np.array_equal(np.sign(stress[1:]), np.sign(zs[1:]))
    assert (np.diff(signed_log1p(np.unique(stress[1:]))) > 0).all()
    print('\nFEATURE FAMILY COUNTS:\n' + families.to_string(index=False))
    print('\nTOP EXCLUDED (inner-fit KS, not model gain):\n' + numeric[~numeric.selected100].nlargest(16, 'ks_inner')[['feature', 'family', 'ks_inner']].to_string(index=False))
    print('\nLARGEST SCALES:\n' + numeric.nlargest(10, 'max_abs')[['feature', 'selected100', 'q50', 'q99', 'max_abs', 'skew']].to_string(index=False))
    print('\nSIGNED LOG CHECKS:', logs[bool_cols].all().to_dict(), 'max_abs_KS_delta=', float(logs.ks_delta.abs().max()))
    print('FLOAT64 RANK/TIE FINDINGS:\n' + logs.loc[~logs.rank_equal,
          ['feature', 'unique_before', 'unique_after', 'nondecreasing_unique', 'ks_delta']].to_string(index=False))
    print('Stateless transform, no fit; checks Float64 only, not Kumo internal Float16/preprocessing.')
    print('THIN/HISTORY:', {'thin_rows': int(thin.sum()), 'history_rows': int((~thin).sum()),
                             'thin_bad_rate': float(yf[thin].mean()), 'history_bad_rate': float(yf[~thin].mean()),
                             'all_numeric_thin_all_missing': int(miss[(miss.segment == 'thin') & miss.all_missing].shape[0]),
                             'selected_thin_all_missing': int(miss[(miss.segment == 'thin') & miss.all_missing & miss.selected100].shape[0])}, flush=True)

    duplicates, overlaps = duplicate_audit(f, X, Xt, train, partitions)
    print('\nDUPLICATES:\n' + pd.DataFrame(duplicates).to_string(index=False))
    print('\nSPLIT OVERLAPS:\n' + pd.DataFrame(overlaps).to_string(index=False), flush=True)
    tables = {t: pl.read_parquet(PQ / f'{t}.parquet') for t in TABLES}
    cutoff = cutoff_audit(tables)
    print('\nRAW TIMESTAMP AUDIT (logic139 plus bureau report date):\n' + cutoff.to_string(index=False), flush=True)

    # Raw source alignment is evidence of consistency, not proof of feature cache build provenance.
    apps = pl.concat([tables['train_applications'].with_columns(is_train=pl.lit(1)),
                      tables['test_applications'].with_columns(target=pl.lit(None, pl.Int64), is_train=pl.lit(0))], how='diagonal_relaxed').sort('application_id')
    source_keys_equal = f.select(ID_COLS).equals(apps.select(ID_COLS))
    direct_cols = ['loan_amount', 'annual_interest_rate', 'proposed_monthly_installment', 'loan_purpose',
                   'age', 'region', 'employment_type', 'employment_tenure_months',
                   'declared_monthly_income', 'dependents', 'housing_status']
    raw = apps.join(tables['customer_profiles'], on='customer_id').sort('application_id')
    source_direct_equal = {c: f[c].equals(raw[c]) for c in direct_cols}
    source_files = [ROOT / 'eda' / p for p in ['features.py', 'common.py', 'valid2.py', '139_cutoff_audit.py', '153_kumo_input_audit.py']]
    schema = pq.ParquetFile(cache).schema_arrow
    metadata = {k.decode(errors='replace'): v.decode(errors='replace') for k, v in (schema.metadata or {}).items()}
    provenance = {'files': [fingerprint(p) for p in paths + source_files],
                  'feature_schema': {field.name: str(field.type) for field in schema},
                  'feature_schema_metadata': metadata, 'raw_source_keys_equal': source_keys_equal,
                  'raw_direct_columns_equal': source_direct_equal,
                  'build_binding_proven': False, 'injection_tested_in_this_audit': False,
                  'notes': ['common.features returns cache without checking source hash/version/mtime.',
                            'Current file fingerprints identify this audit input, not the historical cache build.',
                            'All cached engineered columns were NOT independently recomputed.',
                            'features.v5_features price reference uses all customers, including test, with transaction month < decision month; historical unsupervised reference, not fold-local. No reference was refitted here.',
                            'Repayment filters due_date, not payment snapshot fields; max/actual DPD and amount_paid correctness relies on source being cutoff snapshots.',
                            'repayment_features joins previous loan term without its own cutoff filter; zero raw future previous rows currently prevents exposure.',
                            'Customer profiles have no availability timestamp; cutoff-time snapshot semantics cannot be established from timestamps alone.']}
    assert source_keys_equal and all(source_direct_equal.values())
    print('\nPROVENANCE: keys/direct raw columns match:', source_keys_equal, source_direct_equal,
          '\ncache schema metadata:', metadata, '\ncache build binding: NOT PROVEN; injection tested: NO', flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    view = numeric.sort_values('max_abs', ascending=False).head(18).iloc[::-1]
    axes[0, 0].barh(view.feature, np.log10(np.maximum(view.max_abs, 1e-12)),
                    color=np.where(view.selected100, '#176B87', '#B0B0B0'))
    axes[0, 0].set(xlabel='log10(max absolute value), inner fit', title='Large raw scales (blue = selected100)')
    order = numeric.sort_values(['family', 'ks_inner'], ascending=[True, False]).feature.tolist()
    matrix = miss.pivot(index='feature', columns='segment', values='missing_fraction').loc[order, ['thin', 'history']]
    axes[0, 1].imshow(matrix.to_numpy().T, aspect='auto', vmin=0, vmax=1, cmap='magma')
    axes[0, 1].set(yticks=[0, 1], yticklabels=['thin', 'history'], xlabel='All 228 numeric features ordered by family / KS',
                    title='Missing fractions, inner fit (dark=0, yellow=1)')
    family_counts = numeric.groupby(['family', 'selected100']).size().unstack(fill_value=0)
    family_counts.rename(columns={False: 'excluded', True: 'selected100'}).plot.barh(ax=axes[1, 0], stacked=True, color=['#B0B0B0', '#176B87'])
    axes[1, 0].set(xlabel='Feature count', ylabel='', title='Numeric family coverage')
    example_col = 'cf_free_min'
    raw_example = sub[example_col].to_numpy(float)
    for target, color in [(0, '#176B87'), (1, '#D95F02')]:
        z = signed_log1p(raw_example[(yf == target) & np.isfinite(raw_example)])
        axes[1, 1].hist(z, bins=np.linspace(-21, 21, 85), density=True, alpha=.5, color=color, label=f'target={target}')
    axes[1, 1].set(xlabel='sign(x) * log1p(abs(x)), Float64', ylabel='Density',
                    title='cf_free_min: both signs / classes; NaN excluded visually only')
    axes[1, 1].legend()
    fig.suptitle('Kumo input audit | seed42 fold0 INNER FIT only | no model / no outer score', fontsize=15)
    image = OUT / f'{PREFIX}_diagnostic.png'
    fig.savefig(image, dpi=150)
    plt.close(fig)
    summary = {'protocol': {'seed': 42, 'fold': 0, 'statistics_and_selection': 'inner_fit_only',
                            'partition_rows': {k: len(v) for k, v in partitions.items()}, 'test_rows': len(Xt),
                            'numeric_features': len(nums), 'selected_columns': selected, 'no_model': True, 'no_gpu': True,
                            'no_test_distribution_fit': True},
               'categories': category_summary, 'thin_rows': int(thin.sum()),
               'numeric_all_nan_thin': miss[(miss.segment == 'thin') & miss.all_missing].feature.tolist(),
               'selected_all_nan_thin': miss[(miss.segment == 'thin') & miss.all_missing & miss.selected100].feature.tolist(),
               'signed_log_checks_all_true': logs[bool_cols].all().to_dict(),
               'signed_log_rank_failure_features': logs.loc[~logs.rank_equal, 'feature'].tolist(),
               'signed_log_ks_changed': logs.loc[logs.ks_delta.abs() > 1e-12,
                                                ['feature', 'ks_raw', 'ks_signed_log1p', 'ks_delta']].to_dict('records'),
               'max_abs_signed_log_ks_delta': float(logs.ks_delta.abs().max()),
               'money_audit': money_result,
               'duplicates': duplicates, 'overlaps': overlaps,
               'cutoff_violations_total': int(cutoff.after_cutoff_rows.sum()),
               'provenance': provenance, 'diagnostic': str(image.relative_to(ROOT)),
               'recommendations_inner_search_only': ['categorical stype versus current arbitrary numerical codes',
                                                    'stateless signed_log1p monetary/large-scale columns versus raw; do not claim exact rank invariance; preserve NaNs and use same transform context/query',
                                                    'selection100 versus inner-only conditional/gain/union selection; audit shows exclusions, not conditional utility'],
               'not_claimed': ['No model improvement or public KS prediction', 'No future injection test',
                               'No Float16/Kumo internal preprocessing invariance', 'No proof cached engineering matches current source']}
    (OUT / f'{PREFIX}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('\nSAVED:', image.relative_to(ROOT), OUT.relative_to(ROOT) / f'{PREFIX}_summary.json')
    print('CANDIDATES FOR INNER SEARCH ONLY: categorical stype; stateless signed-log numeric; selection KS versus conditional/union. No outer-selected recommendation.')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--money-only']:
        # Cheap parent-requested extension: reuse the completed inner-fit audit,
        # never reload test data, rerun selection, or fit a distribution.
        money_audit(pd.read_csv(OUT / f'{PREFIX}_numeric.csv'),
                    pd.read_csv(OUT / f'{PREFIX}_signed_log1p.csv'))
    elif sys.argv[1:]:
        raise SystemExit('Usage: 153_kumo_input_audit.py [--money-only]')
    else:
        main()
