"""A regularized additive probability model to complement boosted trees."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = [
    'age', 'employment_tenure_months', 'declared_monthly_income', 'dependents',
    'loan_amount', 'proposed_monthly_installment', 'annual_interest_rate',
    'installment_declared_income_ratio', 'installment_observed_income_ratio',
    'bureau_score_mean', 'v2_bureau_history_score_mean', 'v2_bureau_history_score_min',
    'v2_bureau_history_score_max', 'bureau_utilization_max', 'bureau_inquiries_max',
    'bureau_delinquent_accounts_max', 'bureau_debt_income_ratio',
    'repayment_installment_count', 'repayment_loan_count', 'repayment_historical_dpd_mean',
    'v3_repay_all_late_1_posterior_24', 'v3_repay_all_late_7_posterior_24',
    'v3_repay_all_late_30_posterior_24', 'v3_repay_all_late_90_posterior_24',
    'v3_repay_180d_late_1_posterior_24', 'v3_repay_all_fully_paid_share',
    'v2_existing_total_burden_observed', 'v2_installment_disposable_income_ratio',
    'v2_existing_future_installments', 'v2_existing_future_scheduled_amount',
    'v2_account_3m_income_like_credits_mean', 'v2_account_3m_minimum_balance_min',
    'v2_account_3m_average_balance_mean', 'v2_account_3m_outflows_mean',
    'v3_cash_deficit_monthly_income_share', 'v3_income_after_installments_per_dependent',
    'ecom_last_purchase_recency_days', 'v2_ecom_30d_orders', 'v2_ecom_60d_orders',
    'v2_ecom_30d_spend', 'v2_ecom_60d_spend', 'v2_ecom_completed_per_observed_month',
    'v3_ecom_category_entropy', 'v3_ecom_monthly_spend_cv',
    'v4_price_all_mean', 'v4_price_all_equal_category_mean',
    'v4_price_all_std', 'v4_price_180d_mean', 'v4_price_90d_mean',
    'v4_bureau_all_equal_provider_mean_mean', 'v4_bureau_all_equal_provider_mean_std',
    'v4_loan_late_1_mean', 'v4_weighted_late_1_365d', 'v4_last12_late_1_posterior',
    'loan_purpose', 'region', 'employment_type', 'housing_status',
]


def prepare(frame, columns, categorical):
    x = frame[columns].copy()
    for c in categorical:
        x[c] = x[c].fillna('MISSING').astype(str)
    for c in x.columns.difference(categorical):
        x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    return x


def fit(frame, indices):
    columns = [c for c in FEATURES if c in frame]
    categorical = frame[columns].select_dtypes(include='object').columns.tolist()
    numeric = [c for c in columns if c not in categorical]
    training = frame.iloc[indices]
    continuous = [c for c in numeric if training[c].nunique(dropna=True) > 10]
    linear = [c for c in numeric if c not in continuous]
    encoder = ColumnTransformer([
        ('spline', Pipeline([('fill', SimpleImputer(strategy='median')),
                             ('spline', SplineTransformer(n_knots=5, degree=3, knots='quantile', extrapolation='constant')),
                             ('scale', StandardScaler())]), continuous),
        ('linear', Pipeline([('fill', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), linear),
        ('missing', MissingIndicator(features='all', sparse=False, error_on_new=False), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    x = prepare(training, columns, categorical)
    with threadpool_limits(limits=4):
        xt = encoder.fit_transform(x)
        model = LogisticRegression(C=.05, solver='newton-cholesky', max_iter=80, tol=1e-5)
        model.fit(xt, training.target.astype(int))
    return {'family': 'spline', 'columns': columns, 'categorical': categorical,
            'encoder': encoder, 'model': model}


def predict(bundle, frame):
    x = prepare(frame, bundle['columns'], bundle['categorical'])
    with threadpool_limits(limits=4):
        return bundle['model'].predict_proba(bundle['encoder'].transform(x))[:, 1]
