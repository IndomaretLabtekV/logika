"""Aggregate category-normalized prices and mature repayment events before decision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = pd.read_parquet(ROOT / 'data/processed/application_features_v3.parquet').set_index('customer_id')
    cutoff = base[['decision_datetime']].rename(columns={'decision_datetime': 'cutoff'})
    training_ids = set(base.index[base.split.eq('train')])
    audit = {'training_target_used': False, 'normalization_population': 'Training customer histories; no target labels.'}
    blocks = []

    transactions = pd.read_csv(ROOT / 'data/ecommerce_transactions.csv', parse_dates=['transaction_datetime'])
    transactions = transactions.merge(cutoff, left_on='customer_id', right_index=True, validate='many_to_one')
    audit['future_transaction_rows'] = int(transactions.transaction_datetime.ge(transactions.cutoff).sum())
    assert audit['future_transaction_rows'] == 0
    transactions = transactions[transactions.order_status.eq('completed') & transactions.quantity.gt(0)].copy()
    transactions['age_days'] = (transactions.cutoff - transactions.transaction_datetime).dt.total_seconds() / 86400
    transactions['log_price'] = np.log1p(transactions.item_total / transactions.quantity)
    reference = transactions[transactions.customer_id.isin(training_ids)].groupby('product_category').log_price.median()
    transactions['relative_log_price'] = transactions.log_price - transactions.product_category.map(reference)
    audit['category_log_price_reference'] = reference.to_dict()
    for window in [None, 180, 90]:
        part = transactions if window is None else transactions[transactions.age_days.le(window)]
        tag = 'all' if window is None else f'{window}d'
        grouped = part.groupby('customer_id').relative_log_price
        feature = grouped.agg(['mean', 'median', 'std', 'min', 'max'])
        feature['q10'] = grouped.quantile(.1)
        feature['q90'] = grouped.quantile(.9)
        feature = feature.add_prefix(f'v4_price_{tag}_')
        # Give each category equal weight instead of letting frequent grocery items dominate.
        categories = part.groupby(['customer_id', 'product_category']).relative_log_price.mean().unstack()
        feature[f'v4_price_{tag}_equal_category_mean'] = categories.mean(axis=1)
        feature[f'v4_price_{tag}_equal_category_median'] = categories.median(axis=1)
        feature[f'v4_price_{tag}_between_category_std'] = categories.std(axis=1)
        orders = part.groupby(['customer_id', 'order_id']).relative_log_price.mean()
        order_summary = orders.groupby(level='customer_id').agg(['mean', 'std'])
        order_summary.columns = [f'v4_price_{tag}_equal_order_{c}' for c in order_summary]
        blocks.append(feature.join(order_summary))
    print('Normalized category prices complete.', flush=True)
    del transactions, part, grouped, feature, categories, orders, order_summary

    repayments = pd.read_csv(ROOT / 'data/repayment_history.csv', parse_dates=['due_date', 'last_payment_datetime'])
    repayments = repayments.merge(cutoff, left_on='customer_id', right_index=True, validate='many_to_one')
    audit['future_due_rows'] = int(repayments.due_date.ge(repayments.cutoff).sum())
    audit['future_payment_rows'] = int(repayments.last_payment_datetime.ge(repayments.cutoff).sum())
    assert audit['future_due_rows'] == audit['future_payment_rows'] == 0
    repayments['age_days'] = (repayments.cutoff - repayments.due_date).dt.total_seconds() / 86400
    recent = repayments.sort_values('due_date').groupby('customer_id').tail(12)
    for threshold in [1, 7, 30, 90]:
        # A recently due installment cannot yet reach a larger DPD threshold.
        eligible = repayments[repayments.age_days.ge(threshold)].copy()
        eligible['late'] = eligible.max_days_past_due.ge(threshold).astype(int)
        prior = float(eligible[eligible.customer_id.isin(training_ids)].late.mean())
        audit[f'mature_late_{threshold}_prior'] = prior
        features = eligible.groupby('customer_id').late.agg(['sum', 'count', 'mean'])
        for strength in [6, 24]:
            features[f'posterior_{strength}'] = (features['sum'] + strength * prior) / (features['count'] + strength)
        features = features.add_prefix(f'v4_mature_late_{threshold}_')
        loan_rates = eligible.groupby(['customer_id', 'loan_id']).late.mean()
        loan_summary = loan_rates.groupby(level='customer_id').agg(['mean', 'std', 'min', 'max'])
        loan_summary.columns = [f'v4_loan_late_{threshold}_{c}' for c in loan_summary]
        features = features.join(loan_summary)
        last = recent[recent.age_days.ge(threshold)].copy()
        last['late'] = last.max_days_past_due.ge(threshold).astype(int)
        counts = last.groupby('customer_id').late.agg(['sum', 'count'])
        features[f'v4_last12_late_{threshold}_posterior'] = (counts['sum'] + 12 * prior) / (counts['count'] + 12)
        for half_life in [90, 365]:
            eligible['w'] = np.exp(-np.log(2) * eligible.age_days / half_life)
            eligible['wlate'] = eligible.w * eligible.late
            weighted = eligible.groupby('customer_id')[['w', 'wlate']].sum()
            features[f'v4_weighted_late_{threshold}_{half_life}d'] = (weighted.wlate + 6 * prior) / (weighted.w + 6)
        blocks.append(features)
    positive = repayments[repayments.max_days_past_due.gt(0)].groupby('customer_id').max_days_past_due.agg(['mean', 'median', 'std'])
    blocks.append(positive.add_prefix('v4_positive_dpd_'))
    print('Mature events and equal-loan delinquency rates complete.', flush=True)
    del repayments, recent, eligible, features, loan_rates, loan_summary, last, counts, weighted, positive

    bureau = pd.read_csv(ROOT / 'data/bureau_observations.csv', parse_dates=['report_datetime', 'available_datetime'])
    bureau = bureau.merge(cutoff, left_on='customer_id', right_index=True, validate='many_to_one')
    audit['future_bureau_report_rows'] = int(bureau.report_datetime.ge(bureau.cutoff).sum())
    audit['future_bureau_available_rows'] = int(bureau.available_datetime.ge(bureau.cutoff).sum())
    assert audit['future_bureau_report_rows'] == audit['future_bureau_available_rows'] == 0
    reference = bureau[bureau.customer_id.isin(training_ids)].groupby('bureau_provider').score.mean()
    bureau['centered_score'] = bureau.score - bureau.bureau_provider.map(reference)
    bureau['age_days'] = (bureau.cutoff - bureau.report_datetime).dt.total_seconds() / 86400
    audit['bureau_score_reference'] = reference.to_dict()
    for window in [None, 180]:
        part = bureau if window is None else bureau[bureau.age_days.le(window)]
        tag = 'all' if window is None else f'{window}d'
        provider = part.groupby(['customer_id', 'bureau_provider']).centered_score.agg(['mean', 'median', 'std'])
        for value in ['mean', 'median', 'std']:
            feature = provider[value].groupby(level='customer_id').agg(['mean', 'min', 'max', 'std'])
            blocks.append(feature.add_prefix(f'v4_bureau_{tag}_equal_provider_{value}_'))
    for half_life in [90, 365]:
        bureau['w'] = np.exp(-np.log(2) * bureau.age_days / half_life).where(bureau.score.notna(), 0)
        bureau['weighted_score'] = bureau.centered_score.fillna(0) * bureau.w
        weighted = bureau.groupby('customer_id')[['w', 'weighted_score']].sum()
        blocks.append((weighted.weighted_score / weighted.w.where(weighted.w.gt(0))).rename(f'v4_bureau_history_weighted_{half_life}d'))
    print('Centered bureau scores complete.', flush=True)

    result = pd.concat([base] + blocks, axis=1).reindex(base.index)
    assert result.columns.is_unique and result.index.is_unique
    added = [c for c in result if c.startswith('v4_')]
    assert all(pd.api.types.is_numeric_dtype(result[c]) for c in added)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert result.application_id.equals(base.application_id)
    assert result.target.equals(base.target)
    result.reset_index().to_parquet(ROOT / 'data/processed/application_features_v4.parquet', index=False)
    audit['added_columns'] = len(added)
    output = ROOT / 'artifacts/optimization/v4'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'feature_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'Saved {len(result):,} rows with {len(added)} new features.', flush=True)


if __name__ == '__main__':
    main()
