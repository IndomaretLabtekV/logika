"""Fixed interpretable candidates with corrected affordability features."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from threadpoolctl import threadpool_limits

from spline_model import FEATURES as V4_FEATURES, prepare
from optimize_ks import EXCLUDED, fit_candidate, predict_bundle

EXTRA_FEATURES = [
    "loan_income_ratio",
    "income_per_dependent",
    "v2_existing_total_burden_declared",
    "v4_price_all_equal_order_mean",
    "v4_price_180d_equal_order_mean",
    "v4_price_90d_equal_order_mean",
    "v4_bureau_history_weighted_90d",
    "v4_bureau_180d_equal_provider_mean_mean",
    "v2_bureau_90d_score_mean",
    "v2_bureau_180d_score_mean",
    "v2_bureau_history_delinquent_accounts_last_12m_mean",
    "v2_repay_365d_paid_ratio",
    "v2_latest_loan_age_days",
    "v2_account_income_like_credits_recent_prior_ratio",
    "has_previous",
    "has_account",
    "has_repayments",
    "v4_price_all_equal_category_median",
    "v4_weighted_late_30_365d",
    "v4_weighted_late_1_90d",
]
REVISED_FEATURES = list(
    dict.fromkeys(
        [
            (
                "installment_income_ratio"
                if c == "installment_declared_income_ratio"
                else c
            )
            for c in V4_FEATURES
        ]
        + EXTRA_FEATURES
    )
)
INTERACTION_FEATURES = [
    "installment_income_ratio",
    "v2_existing_total_burden_declared",
    "v3_repay_all_late_1_posterior_24",
    "v4_price_all_equal_order_mean",
    "v2_bureau_history_score_mean",
    "v2_ecom_30d_orders",
]


def definitions(tree_columns):
    specs = [
        {
            "name": "spline_corrected",
            "family": "spline_plus",
            "params": {
                "C": 0.02,
                "n_knots": 5,
                "positive_weight": 1,
                "interactions": False,
            },
        },
        {
            "name": "spline_interactions",
            "family": "spline_plus",
            "params": {
                "C": 0.02,
                "n_knots": 5,
                "positive_weight": 1,
                "interactions": True,
            },
        },
        {
            "name": "spline_balanced_interactions",
            "family": "spline_plus",
            "params": {
                "C": 0.02,
                "n_knots": 5,
                "positive_weight": 6,
                "interactions": True,
            },
        },
        {
            "name": "spline_knots8",
            "family": "spline_plus",
            "params": {
                "C": 0.02,
                "n_knots": 8,
                "positive_weight": 1,
                "interactions": False,
            },
        },
        {"name": "ebm_unweighted", "family": "ebm", "params": {"positive_weight": 1}},
        {"name": "ebm_balanced", "family": "ebm", "params": {"positive_weight": 6}},
        {
            "name": "xgb_depth2_balanced",
            "family": "xgb",
            "params": {
                "max_depth": 2,
                "min_child_weight": 80,
                "reg_lambda": 50,
                "scale_pos_weight": 6,
            },
            "stages": [600, 1200],
        },
        {
            "name": "xgb_depth3_balanced",
            "family": "xgb",
            "params": {
                "max_depth": 3,
                "min_child_weight": 80,
                "reg_lambda": 40,
                "scale_pos_weight": 6,
            },
            "stages": [500, 900],
        },
        {
            "name": "specialist_cat5",
            "family": "specialist",
            "base_family": "cat",
            "params": {"depth": 5, "l2_leaf_reg": 35, "class_weights": [1, 6]},
            "stages": [500, 900],
        },
        {
            "name": "specialist_lgb15",
            "family": "specialist",
            "base_family": "lgb",
            "params": {
                "num_leaves": 15,
                "min_child_samples": 120,
                "reg_lambda": 40,
                "scale_pos_weight": 6,
            },
            "stages": [350, 650],
        },
    ]
    for spec in specs:
        spec["columns"] = (
            tree_columns
            if spec["family"] in ["xgb", "specialist"]
            else REVISED_FEATURES
        )
        assert not set(spec["columns"]) & EXCLUDED
    return specs


def basket_definitions(tree_columns, new_columns, available):
    old_spline_columns = [c for c in V4_FEATURES if c in available]
    price_columns = [
        "v5_price_pooled_all_mean",
        "v5_price_pooled_180d_mean",
        "v5_price_pooled_all_equal_order_mean",
        "v5_price_pooled_all_equal_category_mean",
    ]
    return [
        {
            "name": "v5_basket_cat5",
            "family": "cat",
            "columns": tree_columns + new_columns,
            "params": {"depth": 5, "l2_leaf_reg": 25, "class_weights": [1, 6]},
            "stages": [350, 550],
        },
        {
            "name": "v5_basket_spline",
            "family": "spline_plus",
            "columns": old_spline_columns + price_columns,
            "params": {
                "C": 0.05,
                "n_knots": 5,
                "positive_weight": 1,
                "interactions": False,
            },
        },
    ]


def shallow_definitions(tree_columns, new_columns):
    return [
        {
            "name": "v5_shallow_cat3",
            "family": "cat",
            "columns": tree_columns + new_columns,
            "params": {"depth": 3, "l2_leaf_reg": 25, "class_weights": [1, 6]},
            "stages": [800, 1300],
        },
        {
            "name": "v5_shallow_cat4",
            "family": "cat",
            "columns": tree_columns + new_columns,
            "params": {"depth": 4, "l2_leaf_reg": 25, "class_weights": [1, 6]},
            "stages": [500, 800],
        },
    ]


def neural_definitions():
    columns = list(
        dict.fromkeys(
            REVISED_FEATURES
            + [
                "v5_price_pooled_all_mean",
                "v5_price_pooled_180d_mean",
                "v5_price_pooled_all_equal_order_mean",
                "v5_price_pooled_all_equal_category_mean",
                "repayment_amount_paid",
                "v2_repay_365d_paid",
                "repayment_paid_ratio",
                "v2_ecom_order_age_days_median",
                "v2_ecom_observed_days",
                "v2_bureau_current_score_mean",
                "v2_bureau_365d_score_mean",
                "v4_bureau_180d_equal_provider_median_mean",
            ]
        )
    )
    return [
        {
            "name": f"v5_tabm_weight{w}",
            "family": "tabm",
            "columns": columns,
            "params": {"positive_weight": w},
        }
        for w in [1, 6]
    ]


def core_neural_definitions():
    columns = [
        "v5_price_pooled_all_mean",
        "v5_price_pooled_all_equal_order_mean",
        "v4_price_all_mean",
        "v4_price_180d_mean",
        "v2_ecom_completed_per_observed_month",
        "v2_ecom_30d_orders",
        "v2_ecom_order_age_days_median",
        "ecom_last_purchase_recency_days",
        "v2_bureau_history_score_mean",
        "v2_bureau_180d_score_mean",
        "v2_bureau_history_delinquent_accounts_last_12m_mean",
        "bureau_utilization_max",
        "bureau_inquiries_max",
        "v3_repay_all_late_1_posterior_24",
        "v3_repay_all_late_30_posterior_24",
        "v3_repay_all_fully_paid_share",
        "repayment_installment_count",
        "repayment_paid_ratio",
        "v2_repay_365d_paid",
        "v2_latest_loan_age_days",
        "v2_existing_future_installments",
        "v2_existing_estimated_remaining_principal",
        "installment_income_ratio",
        "loan_income_ratio",
        "declared_monthly_income",
        "employment_tenure_months",
        "age",
        "dependents",
        "v2_account_3m_income_like_credits_mean",
        "v2_account_3m_outflows_mean",
        "v2_account_3m_minimum_balance_min",
        "has_repayments",
        "has_account",
        "loan_purpose",
        "employment_type",
        "housing_status",
        "region",
    ]
    return [
        {
            "name": f"v5_core_tabm_weight{w}",
            "family": "tabm",
            "columns": columns,
            "params": {
                "positive_weight": w,
                "k": 32,
                "d_block": 256,
                "dropout": 0.1,
                "n_blocks": 2,
            },
        }
        for w in [1, 6]
    ]


def smoothing_definitions(available):
    columns = [c for c in V4_FEATURES if c in available]
    return [
        {
            "name": name,
            "family": "spline_plus",
            "columns": columns,
            "params": {
                "C": c,
                "n_knots": knots,
                "positive_weight": 1,
                "interactions": False,
            },
        }
        for name, c, knots in [
            ("v5_spline_strong", 0.005, 5),
            ("v5_spline_medium", 0.02, 5),
            ("v5_spline_simple", 0.05, 3),
        ]
    ]


def fit_spline_plus(spec, data, indices):
    columns = spec["columns"]
    assert set(columns).issubset(data.columns), "A requested feature is absent."
    categorical = data[columns].select_dtypes(include="object").columns.tolist()
    numeric = [c for c in columns if c not in categorical]
    training = data.iloc[indices]
    continuous = [c for c in numeric if training[c].nunique(dropna=True) > 10]
    linear = [c for c in numeric if c not in continuous]
    params = spec["params"]
    transforms = [
        (
            "spline",
            Pipeline(
                [
                    ("fill", SimpleImputer(strategy="median")),
                    (
                        "basis",
                        SplineTransformer(
                            n_knots=params["n_knots"],
                            degree=3,
                            knots="quantile",
                            extrapolation="constant",
                        ),
                    ),
                    ("scale", StandardScaler()),
                ]
            ),
            continuous,
        ),
        (
            "linear",
            Pipeline(
                [
                    ("fill", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            linear,
        ),
        (
            "missing",
            MissingIndicator(features="all", sparse=False, error_on_new=False),
            numeric,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ),
    ]
    if params["interactions"]:
        transforms.append(
            (
                "interactions",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "products",
                            PolynomialFeatures(
                                degree=2, interaction_only=True, include_bias=False
                            ),
                        ),
                    ]
                ),
                INTERACTION_FEATURES,
            )
        )
    encoder = ColumnTransformer(transforms)
    weight = params["positive_weight"]
    y = training.target.astype(int)
    with threadpool_limits(limits=4):
        x = encoder.fit_transform(prepare(training, columns, categorical))
        model = LogisticRegression(
            C=params["C"], solver="newton-cholesky", max_iter=80, tol=1e-5
        )
        model.fit(x, y, sample_weight=np.where(y.eq(1), weight, 1))
    return {
        "family": "spline_plus",
        "model": model,
        "encoder": encoder,
        "columns": columns,
        "categorical": categorical,
        "weight": weight,
    }


def fit_ebm(spec, data, indices, seed):
    if Path("/tmp/logika-ebm-libs").exists():
        sys.path.insert(0, "/tmp/logika-ebm-libs")
    from interpret.glassbox import ExplainableBoostingClassifier

    columns = spec["columns"]
    assert set(columns).issubset(data.columns)
    x = data.iloc[indices][columns].copy()
    categorical = x.select_dtypes(include="object").columns.tolist()
    for c in categorical:
        x[c] = x[c].fillna("MISSING").astype(str)
    types = ["nominal" if c in categorical else "continuous" for c in columns]
    y = data.target.iloc[indices].astype(int)
    weight = spec["params"]["positive_weight"]
    # Early stopping and interaction selection use only these training indices.
    model = ExplainableBoostingClassifier(
        feature_types=types,
        max_bins=128,
        max_interaction_bins=16,
        interactions=10,
        outer_bags=4,
        inner_bags=0,
        validation_size=0.15,
        learning_rate=0.03,
        max_rounds=4000,
        early_stopping_rounds=100,
        min_samples_leaf=64,
        reg_lambda=4,
        n_jobs=8,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=np.where(y.eq(1), weight, 1))
    return {
        "family": "ebm",
        "model": model,
        "columns": columns,
        "categorical": categorical,
        "weight": weight,
    }


def fit_model(spec, data, indices, seed=2026):
    if spec["family"] == "tabm":
        from neural_v5 import fit_neural

        return fit_neural(spec, data, indices, seed)
    if spec["family"] == "spline_plus":
        return fit_spline_plus(spec, data, indices)
    if spec["family"] == "ebm":
        return fit_ebm(spec, data, indices, seed)
    iterations = spec["iterations"] if "iterations" in spec else max(spec["stages"])
    if spec["family"] == "specialist":
        models = {}
        for group in [0, 1]:
            group_indices = np.asarray(indices)[
                data.has_repayments.iloc[indices].eq(group)
            ]
            # Constant feature filtering is learned inside this training fold.
            columns = [
                c
                for c in spec["columns"]
                if data[c].iloc[group_indices].nunique(dropna=True) > 1
            ]
            inner_spec = dict(spec, family=spec["base_family"])
            models[group], _, _ = fit_candidate(
                inner_spec,
                data,
                group_indices,
                None,
                columns,
                iterations=iterations,
                seed=seed,
            )
        return {
            "family": "specialist",
            "models": models,
            "group_column": "has_repayments",
        }
    bundle, _, _ = fit_candidate(
        spec, data, indices, None, spec["columns"], iterations=iterations, seed=seed
    )
    return bundle


def predict_model(bundle, data, trees=None):
    family = bundle["family"]
    if family == "tabm":
        from neural_v5 import predict_neural

        return predict_neural(bundle, data)
    if family == "specialist":
        p = np.full(len(data), np.nan)
        for group, model in bundle["models"].items():
            mask = data[bundle["group_column"]].eq(group).to_numpy()
            if mask.any():
                p[mask] = predict_model(model, data.loc[mask], trees)
        assert np.isfinite(p).all()
        return p
    if family == "spline_plus":
        x = prepare(data, bundle["columns"], bundle["categorical"])
        with threadpool_limits(limits=4):
            p = bundle["model"].predict_proba(bundle["encoder"].transform(x))[:, 1]
    elif family == "ebm":
        x = data[bundle["columns"]].copy()
        for c in bundle["categorical"]:
            x[c] = x[c].fillna("MISSING").astype(str)
        p = bundle["model"].predict_proba(x)[:, 1]
    elif trees is not None and family == "cat":
        x = data[bundle["columns"]].copy()
        for c in bundle["categorical"]:
            x[c] = x[c].fillna("MISSING").astype(str)
        p = bundle["model"].predict_proba(x, ntree_end=trees)[:, 1]
    elif trees is not None and family == "lgb":
        x = bundle["encoder"].transform(data[bundle["columns"]])
        p = bundle["model"].predict_proba(x, num_iteration=trees)[:, 1]
    elif trees is not None:
        x = bundle["encoder"].transform(data[bundle["columns"]])
        p = bundle["model"].predict_proba(x, iteration_range=(0, trees))[:, 1]
    else:
        return predict_bundle(bundle, data)
    weight = bundle.get("weight", 1)
    return p / (weight * (1 - p) + p)
