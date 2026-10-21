"""Reconstruct observable rules and estimate latent signals without claiming original RNG recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/reverse_engineering"


def distribution_summary(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"rows": 0}
    sample = np.random.default_rng(91403).choice(
        x, size=min(50000, len(x)), replace=False
    )
    mu, sd = stats.norm.fit(sample)
    return {
        "rows": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "quantiles": dict(
            zip(
                ["q01", "q05", "q50", "q95", "q99"],
                np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
            )
        ),
        "skewness": float(stats.skew(sample)),
        "excess_kurtosis": float(stats.kurtosis(sample)),
        "normal_fit_ks_descriptive": (
            float(stats.kstest(sample, "norm", args=(mu, sd)).statistic)
            if sd > 0
            else None
        ),
        "normality_caveat": "Descriptive fitted-distribution distance, not a calibrated hypothesis-test p-value.",
    }


def aggregate_ratio_grid(values, steps):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {
        str(step): float(
            np.isclose(x / step, np.rint(x / step), rtol=0, atol=1e-8).mean()
        )
        for step in steps
    }


def reconstruct_basic_rules(profiles, applications):
    rate = applications.annual_interest_rate.to_numpy() / 12
    amount = applications.loan_amount.to_numpy()
    term = applications.term_months.to_numpy()
    # Stable annuity formula; integer rounding can differ at floating half ties.
    installment = amount * rate / (-np.expm1(-term * np.log1p(rate)))
    observed = applications.proposed_monthly_installment.to_numpy()
    residual = observed - installment
    lag = (
        applications.decision_datetime - applications.application_datetime
    ).dt.total_seconds()
    age_counts = profiles.age.value_counts().sort_index()
    workflow = []
    for low, high in [(0.2, 3), (6, 30), (36, 120)]:
        hours = lag / 3600
        selected = hours[hours.between(low, high)]
        workflow.append(
            {
                "low_hours": low,
                "high_hours": high,
                "observed_fraction": float(len(selected) / len(hours)),
                "rows": len(selected),
                "uniform_fit_ks": float(
                    stats.kstest((selected - low) / (high - low), "uniform").statistic
                ),
            }
        )
    result = {
        "age": {
            "support": [int(profiles.age.min()), int(profiles.age.max())],
            "frequencies": age_counts.to_dict(),
            "uniform_chi2_p": float(stats.chisquare(age_counts).pvalue),
            "interpretation": "Consistent with discrete uniform 23..60. It does not identify the RNG or seed.",
        },
        "profile_categories": {
            c: profiles[c].value_counts(normalize=True).sort_index().to_dict()
            for c in ["dependents", "region", "employment_type", "housing_status"]
        },
        "loan_amount_grid": aggregate_ratio_grid(amount, [1000, 10000, 100000]),
        "annual_interest_rates": applications.annual_interest_rate.value_counts()
        .sort_index()
        .to_dict(),
        "installment": {
            "formula": "P*(r/12)/(1-(1+r/12)^(-term))",
            "nearest_integer_exact_fraction": (
                float(np.rint(installment).eq(observed).mean())
                if isinstance(installment, pd.Series)
                else float((np.rint(installment) == observed).mean())
            ),
            "max_absolute_unrounded_residual_idr": float(np.abs(residual).max()),
            "ceil_exact_fraction": float((np.ceil(installment) == observed).mean()),
            "floor_exact_fraction": float((np.floor(installment) == observed).mean()),
            "signed_residual_idr": distribution_summary(residual),
            "within_one_idr_fraction": float((np.abs(residual) <= 1.000001).mean()),
        },
        "selected_decision_time": {
            "second_zero_fraction": float(
                applications.decision_datetime.dt.second.eq(0).mean()
            ),
            "hour_frequencies": applications.decision_datetime.dt.hour.value_counts()
            .sort_index()
            .to_dict(),
            "application_to_decision_seconds": distribution_summary(lag),
        },
        "workflow_lag_mixture": {
            "components": workflow,
            "candidate_weights": [0.45, 0.35, 0.20],
            "gap_3_6_hours_rows": int(
                (lag / 3600).between(3, 6, inclusive="neither").sum()
            ),
            "gap_30_36_hours_rows": int(
                (lag / 3600).between(30, 36, inclusive="neither").sum()
            ),
            "inference_status": "Intervals are observed support; rounded component probabilities are a generator hypothesis.",
        },
        "inference_status": "Observable quantization and identities are confirmed; generator source code is not available.",
    }
    return result


def fit_price_latent(base):
    columns = [
        "customer_id",
        "order_id",
        "product_category",
        "quantity",
        "item_total",
        "transaction_datetime",
    ]
    events = pd.read_csv(
        ROOT / "data/ecommerce_transactions.csv",
        usecols=columns,
        parse_dates=["transaction_datetime"],
        dtype={"customer_id": "category", "product_category": "category"},
    )
    cutoffs = base.decision_datetime
    assert events.transaction_datetime.lt(
        events.customer_id.map(cutoffs).astype("datetime64[ns]")
    ).all()
    assert events.quantity.gt(0).all()
    ids = pd.Series(np.arange(len(base)), index=base.index)
    customer = events.customer_id.map(ids).to_numpy(dtype=np.int64)
    category, categories = pd.factorize(events.product_category, sort=True)
    order, orders = pd.factorize(events.order_id, sort=False)
    first = ~events.order_id.duplicated().to_numpy()
    order_customer = customer[first]
    quantity = events.quantity.to_numpy()
    unit = events.item_total.to_numpy() / quantity
    log_price = np.log(unit)
    n_customer = np.bincount(customer, minlength=len(base))
    n_order = np.bincount(order)
    train_customer = base.split.eq("train").to_numpy()
    training = train_customer[customer]
    n_category = np.bincount(category[training], minlength=len(categories))
    intercept = (
        np.bincount(category[training], weights=log_price[training]) / n_category
    )
    # Two-way fixed effects remove product composition while fitting a customer preference.
    for _ in range(8):
        customer_effect = (
            np.bincount(
                customer, weights=log_price - intercept[category], minlength=len(base)
            )
            / n_customer
        )
        intercept = (
            np.bincount(
                category[training],
                weights=(log_price - customer_effect[customer])[training],
            )
            / n_category
        )
        center = float(customer_effect[train_customer].mean())
        customer_effect -= center
        intercept += center
    adjusted = log_price - intercept[category]
    order_mean = np.bincount(order, weights=adjusted) / n_order
    train_order = train_customer[order_customer]
    item_df = int((n_order[train_order] - 1).sum())
    item_noise_variance = float(
        np.square((adjusted - order_mean[order])[training]).sum() / item_df
    )
    orders_per_customer = np.bincount(order_customer, minlength=len(base))
    equal_order_mean = (
        np.bincount(order_customer, weights=order_mean, minlength=len(base))
        / orders_per_customer
    )
    order_df = int((orders_per_customer[train_customer] - 1).sum())
    centered_order_ss = float(
        np.square(order_mean - equal_order_mean[order_customer])[train_order].sum()
    )
    correction = item_noise_variance * float(
        ((1 - 1 / orders_per_customer[order_customer]) / n_order)[train_order].sum()
    )
    order_noise_variance = max((centered_order_ss - correction) / order_df, 0)
    observation_variance = order_noise_variance + item_noise_variance / n_order
    weight = 1 / np.maximum(observation_variance, 1e-12)
    precision = np.bincount(order_customer, weights=weight, minlength=len(base))
    raw_effect = (
        np.bincount(order_customer, weights=weight * order_mean, minlength=len(base))
        / precision
    )
    prior_mean = float(raw_effect[train_customer].mean())
    prior_variance = max(
        float(
            raw_effect[train_customer].var() - np.mean(1 / precision[train_customer])
        ),
        1e-8,
    )
    posterior_variance = 1 / (precision + 1 / prior_variance)
    posterior_mean = posterior_variance * (
        precision * raw_effect + prior_mean / prior_variance
    )
    features = pd.DataFrame(
        {
            "re_price_log_gls": raw_effect,
            "re_price_log_posterior": posterior_mean,
            "re_price_posterior_sd": np.sqrt(posterior_variance),
            "re_price_measurement_sd": np.sqrt(1 / precision),
            "re_price_latent_multiplier": np.exp(posterior_mean),
            "re_price_order_noise_sd": np.sqrt(order_noise_variance),
        },
        index=base.index,
    )
    same_previous_order = np.r_[False, order[1:] == order[:-1]]
    paired = np.diff(adjusted)[same_previous_order[1:]]
    residual = adjusted - customer_effect[customer]
    category_jitter = {}
    for j, name in enumerate(categories):
        category_jitter[str(name)] = distribution_summary(
            residual[training & (category == j)]
        )
    # A separate order split tests repeatability of the estimated preference, without targets.
    second_half = (
        np.random.default_rng(18117).integers(0, 2, size=len(n_order)).astype(bool)
    )
    half_estimates = []
    for half in [second_half, ~second_half]:
        part_weight = weight * half
        p = np.bincount(order_customer, weights=part_weight, minlength=len(base))
        half_estimates.append(
            np.bincount(
                order_customer, weights=part_weight * order_mean, minlength=len(base)
            )
            / np.where(p > 0, p, np.nan)
        )
    eligible = (
        train_customer & np.isfinite(half_estimates[0]) & np.isfinite(half_estimates[1])
    )
    audit = {
        "rows": len(events),
        "orders": len(n_order),
        "future_rows": 0,
        "targets_used": False,
        "unit_price_grid": aggregate_ratio_grid(unit, [100, 500, 1000]),
        "minimum_unit_price_idr": float(unit.min()),
        "unit_price_at_5000_fraction": float((unit == 5000).mean()),
        "unit_rounded_before_quantity_exact_fraction": float(
            events.item_total.mod(events.quantity * 1000).eq(0).mean()
        ),
        "category_log_intercepts": dict(zip(map(str, categories), intercept.tolist())),
        "category_geometric_baselines_idr": dict(
            zip(map(str, categories), np.exp(intercept).tolist())
        ),
        "estimated_item_log_noise_sd": float(np.sqrt(item_noise_variance)),
        "estimated_shared_order_log_noise_sd": float(np.sqrt(order_noise_variance)),
        "estimated_customer_effect_prior_sd": float(np.sqrt(prior_variance)),
        "within_order_pair_difference": distribution_summary(paired),
        "category_residuals": category_jitter,
        "order_split_customer_effect_correlation": float(
            np.corrcoef(half_estimates[0][eligible], half_estimates[1][eligible])[0, 1]
        ),
        "latent_customer_effect": distribution_summary(raw_effect[train_customer]),
        "quantization_interval_idr": "For unit prices above floor: pre-rounded unit value approximately in [unit-500,unit+500); item_total = rounded_unit*quantity.",
        "location_identifiability": "Category intercepts and customer effects have a shared arbitrary additive origin. Centering the mean training customer effect at zero fixes a convention, not the original generator scale.",
        "inference_status": "Variance components and latent values are estimates under an additive log-price random-effects model, not recovered original RNG draws.",
    }
    print("Ecommerce latent/noise analysis complete.", flush=True)
    return (
        features,
        audit,
        {
            "item_pair_difference": paired[:: max(1, len(paired) // 50000)],
            "latent_price": raw_effect[train_customer],
            "price_residual": residual[training][::100],
        },
    )


def fit_bureau_latent(base):
    bureau = pd.read_csv(
        ROOT / "data/bureau_observations.csv",
        parse_dates=["report_datetime", "available_datetime"],
    )
    bureau = bureau.merge(
        base[["decision_datetime"]],
        left_on="customer_id",
        right_index=True,
        validate="many_to_one",
    )
    assert (
        bureau.report_datetime.lt(bureau.decision_datetime).all()
        and bureau.available_datetime.lt(bureau.decision_datetime).all()
    )
    training_ids = base.index[base.split.eq("train")]
    paired = bureau.pivot_table(
        index=["customer_id", "application_id"],
        columns="bureau_provider",
        values="score",
        aggfunc="mean",
    )
    paired_train = paired[
        paired.index.get_level_values("customer_id").isin(training_ids)
    ]
    names = sorted(paired.columns)
    differences, variances, mean_differences = {}, {}, {}
    for a, b in [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]:
        values = (paired_train[a] - paired_train[b]).dropna().to_numpy()
        differences[a + "-" + b] = distribution_summary(values)
        variances[(a, b)] = float(values.var())
        mean_differences[(a, b)] = float(values.mean())
    # Solve differences with the first provider as reference; then center the offsets.
    offsets = np.array(
        [
            0,
            -mean_differences[(names[0], names[1])],
            -mean_differences[(names[0], names[2])],
        ]
    )
    offsets -= offsets.mean()
    v01, v02, v12 = [
        variances[p]
        for p in [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
    ]
    noise = np.maximum(
        np.array([(v01 + v02 - v12) / 2, (v01 + v12 - v02) / 2, (v02 + v12 - v01) / 2]),
        1e-8,
    )
    offset_map, variance_map = dict(zip(names, offsets)), dict(zip(names, noise))
    bureau["adjusted_score"] = bureau.score - bureau.bureau_provider.map(offset_map)
    bureau["precision"] = 1 / bureau.bureau_provider.map(variance_map)
    bureau["precision"] = bureau.precision.where(bureau.score.notna(), 0)
    bureau["weighted_score"] = bureau.adjusted_score.fillna(0) * bureau.precision
    summary = (
        bureau.groupby("customer_id")[["precision", "weighted_score"]]
        .sum()
        .reindex(base.index)
    )
    mean = summary.weighted_score / summary.precision.where(summary.precision.gt(0))
    train_mean = mean.reindex(training_ids).dropna()
    mean_precision = summary.precision.reindex(train_mean.index)
    prior_mean = float(train_mean.mean())
    prior_variance = max(
        float(train_mean.var(ddof=0) - (1 / mean_precision).mean()), 1e-8
    )
    posterior_variance = 1 / (summary.precision.fillna(0) + 1 / prior_variance)
    posterior_mean = posterior_variance * (
        summary.weighted_score.fillna(0) + prior_mean / prior_variance
    )
    features = pd.DataFrame(
        {
            "re_bureau_score_gls": mean,
            "re_bureau_score_posterior": posterior_mean,
            "re_bureau_posterior_sd": np.sqrt(posterior_variance),
        },
        index=base.index,
    )
    availability_days = (
        bureau.available_datetime - bureau.report_datetime
    ).dt.total_seconds() / 86400
    audit = {
        "rows": len(bureau),
        "targets_used": False,
        "future_report_rows": 0,
        "future_available_rows": 0,
        "score_integer_fraction": float(bureau.score.dropna().mod(1).eq(0).mean()),
        "report_second_zero_fraction": float(
            bureau.report_datetime.dt.second.eq(0).mean()
        ),
        "availability_lag_days": distribution_summary(availability_days),
        "estimated_provider_offsets": dict(zip(names, offsets.tolist())),
        "estimated_provider_noise_sd": dict(zip(names, np.sqrt(noise).tolist())),
        "estimated_latent_score_prior_mean": prior_mean,
        "estimated_latent_score_prior_sd": float(np.sqrt(prior_variance)),
        "pair_differences": differences,
        "inference_status": "Same-application paired scores estimate offset/noise under independent unbiased provider errors. Age effects, dependence and clipping can violate that model.",
    }
    print("Bureau matched-pair noise analysis complete.", flush=True)
    return (
        features,
        audit,
        (paired_train[names[0]] - paired_train[names[1]]).dropna().to_numpy()[::5],
    )


def fit_account_latent(base):
    account = pd.read_csv(ROOT / "data/account_behavior.csv", parse_dates=["month"])
    account = account.merge(
        base[["decision_datetime", "employment_type"]],
        left_on="customer_id",
        right_index=True,
        validate="many_to_one",
    )
    assert (
        (account.month + pd.offsets.MonthBegin(1)).le(account.decision_datetime).all()
    )
    account.sort_values(["customer_id", "month"], inplace=True)
    previous = account.groupby("customer_id").closing_balance.shift()
    carry = (account.opening_balance - previous).dropna()
    balance_residual = (
        account.closing_balance
        - account.opening_balance
        - account.inflows
        + account.outflows
    )
    account["log_income"] = np.log(
        account.income_like_credits.where(account.income_like_credits.gt(0))
    )
    account["log_outflow"] = np.log(account.outflows.where(account.outflows.gt(0)))
    summary = (
        account.groupby("customer_id")
        .agg(
            log_income=("log_income", "mean"),
            log_income_std=("log_income", "std"),
            log_outflow=("log_outflow", "mean"),
            n=("log_income", "count"),
        )
        .reindex(base.index)
    )
    account["income_log_difference"] = account.groupby("customer_id").log_income.diff()
    jitter = {
        str(name): distribution_summary(part.income_log_difference.dropna())
        for name, part in account.groupby("employment_type")
    }
    variance_map = {
        str(name): float(part.income_log_difference.dropna().var(ddof=0) / 2)
        for name, part in account.groupby("employment_type")
    }
    measurement_variance = base.employment_type.map(variance_map) / summary.n.where(
        summary.n.gt(0)
    )
    features = pd.DataFrame(
        {
            "re_income_geometric_mean": np.exp(summary.log_income),
            "re_income_log_mean": summary.log_income,
            "re_income_log_sd": summary.log_income_std,
            "re_income_log_measurement_sd": np.sqrt(measurement_variance),
            "re_declared_income_log_gap": np.log(base.declared_monthly_income)
            - summary.log_income,
            "re_outflow_income_geometric_ratio": np.exp(
                summary.log_outflow - summary.log_income
            ),
        },
        index=base.index,
    )
    average_residual = (
        account.average_balance
        - (account.opening_balance + account.closing_balance) / 2
    )
    normalized = average_residual / account.income_like_credits.where(
        account.income_like_credits.gt(0)
    )
    audit = {
        "rows": len(account),
        "targets_used": False,
        "incomplete_month_rows": 0,
        "opening_matches_previous_closing_fraction": float(carry.eq(0).mean()),
        "balance_identity_exact_fraction": float(balance_residual.eq(0).mean()),
        "income_adjacent_log_difference_by_employment": jitter,
        "estimated_iid_log_income_noise_sd_by_employment": {
            k: float(np.sqrt(v)) for k, v in variance_map.items()
        },
        "average_balance_midpoint_residual_in_income_units": distribution_summary(
            normalized
        ),
        "inference_status": "Income jitter estimate uses Var(log income_t-log income_(t-1))/2. Trend, autocorrelation and shocks can invalidate the iid assumption.",
    }
    print("Account identities and income jitter analysis complete.", flush=True)
    return features, audit, account.income_log_difference.dropna().to_numpy()[::20]


def beta_binomial_prior(k, n):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    positive = n > 0
    k, n = k[positive], n[positive]
    initial_mean = np.clip(k.sum() / n.sum(), 1e-5, 1 - 1e-5)

    def objective(log_params):
        alpha, beta = np.exp(log_params)
        return float(
            -(
                special.betaln(k + alpha, n - k + beta) - special.betaln(alpha, beta)
            ).sum()
        )

    result = optimize.minimize(
        objective,
        np.log([initial_mean * 12, (1 - initial_mean) * 12]),
        method="L-BFGS-B",
        bounds=[(-7, 10), (-7, 10)],
    )
    assert result.success, result.message
    alpha, beta = np.exp(result.x)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "strength": float(alpha + beta),
        "mean": float(alpha / (alpha + beta)),
        "customers": len(n),
        "optimizer_success": bool(result.success),
    }


def fit_repayment_latent(base):
    repayment = pd.read_csv(
        ROOT / "data/repayment_history.csv",
        parse_dates=["due_date", "last_payment_datetime"],
    )
    repayment = repayment.merge(
        base[["decision_datetime"]],
        left_on="customer_id",
        right_index=True,
        validate="many_to_one",
    )
    assert repayment.due_date.lt(repayment.decision_datetime).all()
    assert (
        repayment.last_payment_datetime.isna()
        | repayment.last_payment_datetime.lt(repayment.decision_datetime)
    ).all()
    repayment["age"] = (
        repayment.decision_datetime - repayment.due_date
    ).dt.total_seconds() / 86400
    features = pd.DataFrame(index=base.index)
    audit = {
        "rows": len(repayment),
        "targets_used": False,
        "future_due_rows": 0,
        "future_payment_rows": 0,
        "beta_binomial_priors": {},
    }
    train_ids = base.index[base.split.eq("train")]
    for threshold in [1, 7, 30]:
        eligible = repayment[repayment.age.ge(threshold)].copy()
        eligible["late"] = eligible.max_days_past_due.ge(threshold).astype(int)
        counts = (
            eligible.groupby("customer_id")
            .late.agg(["sum", "count"])
            .reindex(base.index)
            .fillna(0)
        )
        prior = beta_binomial_prior(
            counts["sum"].reindex(train_ids), counts["count"].reindex(train_ids)
        )
        a, b = prior["alpha"], prior["beta"]
        denominator = counts["count"] + a + b
        features[f"re_late_{threshold}_posterior"] = (counts["sum"] + a) / denominator
        features[f"re_late_{threshold}_posterior_sd"] = np.sqrt(
            (counts["sum"] + a)
            * (counts["count"] - counts["sum"] + b)
            / (denominator**2 * (denominator + 1))
        )
        audit["beta_binomial_priors"][str(threshold)] = prior
    paid_fraction = repayment.amount_paid / repayment.amount_due
    audit["payment_fraction"] = distribution_summary(paid_fraction)
    audit["full_payment_fraction"] = float(paid_fraction.eq(1).mean())
    audit["current_dpd_frequencies_top20"] = (
        repayment.days_past_due.value_counts().head(20).to_dict()
    )
    audit["inference_status"] = (
        "Historical propensity posterior, not original Bernoulli RNG recovery. Payment dependence/censoring may overstate precision."
    )
    print("Repayment empirical-Bayes propensity analysis complete.", flush=True)
    return features, audit


def write_figures(price_samples, bureau_difference, income_difference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, values, title in [
        (
            axes[0, 0],
            price_samples["item_pair_difference"],
            "Selisih log-harga dua item dalam order",
        ),
        (axes[0, 1], bureau_difference, "Selisih skor biro A dan B pada aplikasi sama"),
        (axes[1, 0], income_difference, "Perubahan log income-like credits antarbulan"),
        (
            axes[1, 1],
            price_samples["latent_price"],
            "Estimasi preferensi harga customer",
        ),
    ]:
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=80, density=True, color="#0284c7", alpha=0.65)
        mean, sd = stats.norm.fit(values)
        grid = np.linspace(np.quantile(values, 0.001), np.quantile(values, 0.999), 250)
        ax.plot(
            grid,
            stats.norm.pdf(grid, mean, sd),
            color="#ea580c",
            label="Normal dengan mean/SD terestimasi",
        )
        ax.set(title=title, xlabel="Nilai residual / laten", ylabel="Kepadatan")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Pola noise yang diamati: distribusi mendekati normal bukan bukti seed RNG"
    )
    fig.tight_layout()
    fig.savefig(
        ROOT / "artifacts/figures/synthetic_noise_patterns.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(ROOT / "data/processed/application_features_v5.parquet")
    base = full.set_index("customer_id")
    profiles = pd.read_csv(ROOT / "data/customer_profiles.csv")
    apps = pd.concat(
        [
            pd.read_csv(ROOT / "data/train_applications.csv"),
            pd.read_csv(ROOT / "data/test_applications.csv"),
        ],
        ignore_index=True,
    )
    for c in ["application_datetime", "decision_datetime"]:
        apps[c] = pd.to_datetime(apps[c])
    audit = {
        "basic_rules": reconstruct_basic_rules(profiles, apps),
        "target_used_in_latent_feature_estimation": False,
        "original_generator_code_available": False,
        "rng_reference": "https://numpy.org/doc/stable/reference/random/compatibility.html",
    }
    price, audit["ecommerce"], price_samples = fit_price_latent(base)
    bureau, audit["bureau"], bureau_difference = fit_bureau_latent(base)
    account, audit["account"], income_difference = fit_account_latent(base)
    repayment, audit["repayment"] = fit_repayment_latent(base)
    new = pd.concat([price, bureau, account, repayment], axis=1)
    assert new.index.equals(base.index) and new.columns.is_unique
    assert not any("target" in c for c in new.columns)
    result = base.join(new).replace([np.inf, -np.inf], np.nan)
    assert result.application_id.equals(base.application_id) and result.target.equals(
        base.target
    )
    result.reset_index().to_parquet(
        ROOT / "data/processed/application_features_reverse.parquet", index=False
    )
    audit["added_features"] = new.columns.tolist()
    (DESTINATION / "pattern_audit.json").write_text(json.dumps(audit, indent=2))
    write_figures(price_samples, bureau_difference, income_difference)
    print(
        "Saved pattern audit and",
        len(new.columns),
        "estimated latent/noise features.",
        flush=True,
    )


if __name__ == "__main__":
    main()
