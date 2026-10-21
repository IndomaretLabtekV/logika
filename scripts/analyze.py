"""Reproducible, decision-time-safe EDA for LOGIKA UI 2026."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
PROCESSED = ROOT / "data/processed"
OUT = ROOT / "artifacts"
TABLES = OUT / "tables"
TARGET = "target"
KEY = "customer_id"


def safe_ratio(a, b):
    return a / b.where(b > 0)


def read(name, dates=()):
    return pd.read_csv(RAW / f"{name}.csv", parse_dates=list(dates))


def build_features():
    for folder in [PROCESSED, TABLES, OUT / "figures"]:
        folder.mkdir(parents=True, exist_ok=True)
    train = read("train_applications", ["application_datetime", "decision_datetime"])
    test = read("test_applications", ["application_datetime", "decision_datetime"])
    train["split"] = "train"
    test["split"] = "test"
    base = pd.concat([train, test], ignore_index=True)
    assert base.application_id.is_unique and base.customer_id.is_unique
    cutoff = base[[KEY, "decision_datetime"]].rename(columns={"decision_datetime": "cutoff"})
    inventory, missing, audits, coverage = [], [], {}, []

    def inspect(df, name, primary):
        inventory.append({"table": name, "rows": len(df), "columns": len(df.columns),
                          "customers": df[KEY].nunique() if KEY in df else np.nan,
                          "size_mb": (RAW / f"{name}.csv").stat().st_size / 1e6})
        for col in df:
            missing.append({"table": name, "column": col, "dtype": str(df[col].dtype),
                            "missing": int(df[col].isna().sum()),
                            "missing_pct": 100 * df[col].isna().mean(),
                            "unique_values": df[col].nunique()})
        audits[name] = {"duplicate_rows": int(df.duplicated().sum()),
                        "duplicate_primary_keys": int(df.duplicated(primary).sum())}
        if KEY in df:
            ids = set(df[KEY])
            audits[name]["orphan_customers"] = len(ids - set(base[KEY]))
            for part in ["train", "test"]:
                subset = base[base.split.eq(part)]
                coverage.append({"table": name, "split": part,
                                 "covered_customers": int(subset[KEY].isin(ids).sum()),
                                 "coverage_pct": 100 * subset[KEY].isin(ids).mean()})
        print(f"Audited {name}: {len(df):,} rows", flush=True)

    def at_cutoff(df, name, cols):
        df = df.merge(cutoff, on=KEY, how="left", validate="many_to_one")
        assert df.cutoff.notna().all()
        for col in cols:
            audits[name][f"{col}_at_or_after_cutoff"] = int(df[col].ge(df.cutoff).sum())
        return df

    def attach(features):
        nonlocal base
        assert features.index.is_unique
        base = base.merge(features, left_on=KEY, right_index=True, how="left", validate="one_to_one")
        assert len(base) == len(train) + len(test)

    inspect(train.drop(columns="split"), "train_applications", "application_id")
    inspect(test.drop(columns="split"), "test_applications", "application_id")
    submission = read("sample_submission")
    inspect(submission, "sample_submission", "application_id")
    audits["submission"] = {
        "same_id_set_as_test": set(submission.application_id) == set(test.application_id),
        "same_row_order_as_test": submission.application_id.equals(test.application_id),
        "target_column": "default_probability",
    }
    profiles = read("customer_profiles")
    inspect(profiles, "customer_profiles", KEY)
    base = base.merge(profiles, on=KEY, how="left", validate="one_to_one")
    assert base.age.notna().all()
    base["installment_income_ratio"] = safe_ratio(base.proposed_monthly_installment, base.declared_monthly_income)
    base["loan_income_ratio"] = safe_ratio(base.loan_amount, base.declared_monthly_income)
    base["income_per_dependent"] = base.declared_monthly_income / (1 + base.dependents)
    base["decision_delay_hours"] = (base.decision_datetime - base.application_datetime).dt.total_seconds() / 3600
    rate = base.annual_interest_rate / 12
    theoretical = base.loan_amount * rate * (1 + rate) ** base.term_months / ((1 + rate) ** base.term_months - 1)
    audits["applications"] = {
        "train_test_customer_overlap": len(set(train[KEY]) & set(test[KEY])),
        "train_test_application_overlap": len(set(train.application_id) & set(test.application_id)),
        "decision_before_application": int((base.decision_datetime < base.application_datetime).sum()),
        "nonpositive_income": int(base.declared_monthly_income.le(0).sum()),
        "installment_formula_max_rounding_difference_idr": float(abs(base.proposed_monthly_installment - theoretical.round()).max()),
        "term_months_values": sorted(base.term_months.unique().tolist()),
    }
    del profiles

    name = "previous_applications"
    prev = read(name, ["application_datetime", "decision_datetime"])
    inspect(prev, name, "application_id")
    historical_ids = set(prev.application_id)
    approved_loans = set(prev.loan_id.dropna())
    loan_customers = prev.loc[prev.loan_id.notna(), ["loan_id", KEY]].set_index("loan_id")[KEY]
    audits[name]["overlap_with_selected_application_ids"] = len(historical_ids & set(base.application_id))
    prev = at_cutoff(prev, name, ["application_datetime", "decision_datetime"])
    prev = prev[prev.decision_datetime.lt(prev.cutoff)].copy()
    for status in ["approved", "rejected", "cancelled"]:
        prev[f"is_{status}"] = prev.decision.eq(status).astype(int)
    prev["recency_days"] = (prev.cutoff - prev.decision_datetime).dt.total_seconds() / 86400
    features = prev.groupby(KEY).agg(
        previous_count=("application_id", "size"), previous_approved_count=("is_approved", "sum"),
        previous_rejection_rate=("is_rejected", "mean"), previous_cancel_rate=("is_cancelled", "mean"),
        previous_requested_mean=("requested_amount", "mean"), previous_approved_amount_mean=("approved_amount", "mean"),
        previous_installment_mean=("scheduled_monthly_installment", "mean"),
        previous_last_recency_days=("recency_days", "min"))
    for days in [90, 180]:
        features[f"previous_count_{days}d"] = prev[prev.recency_days.le(days)].groupby(KEY).size().reindex(features.index, fill_value=0)
    attach(features)
    base["has_previous"] = base.previous_count.notna().astype(int)
    del prev, features
    gc.collect()

    name = "repayment_history"
    repay = read(name, ["due_date", "last_payment_datetime"])
    inspect(repay, name, "installment_id")
    audits[name]["loans_not_in_approved_history"] = len(set(repay.loan_id) - approved_loans)
    audits[name]["loan_customer_mismatches"] = int(repay.loan_id.map(loan_customers).ne(repay[KEY]).sum())
    audits[name]["amount_paid_above_due"] = int(repay.amount_paid.gt(repay.amount_due).sum())
    audits[name]["max_dpd_below_current_dpd"] = int(repay.max_days_past_due.lt(repay.days_past_due).sum())
    repay = at_cutoff(repay, name, ["due_date", "last_payment_datetime"])
    repay = repay[repay.due_date.lt(repay.cutoff)].copy()
    # Snapshot amounts/DPD are already measured before the selected decision.
    # Any future observed payment would invalidate this snapshot; do not silently use it.
    assert repay.last_payment_datetime.dropna().lt(repay.loc[repay.last_payment_datetime.notna(), "cutoff"]).all()
    repay["unpaid_amount"] = (repay.amount_due - repay.amount_paid).clip(lower=0)
    for threshold in [1, 30, 90]:
        repay[f"ever_dpd_{threshold}"] = repay.max_days_past_due.ge(threshold).astype(int)
    repay["recency_days"] = (repay.cutoff - repay.due_date).dt.total_seconds() / 86400
    features = repay.groupby(KEY).agg(
        repayment_installment_count=("installment_id", "size"), repayment_loan_count=("loan_id", "nunique"),
        repayment_amount_due=("amount_due", "sum"), repayment_amount_paid=("amount_paid", "sum"),
        repayment_unpaid_amount=("unpaid_amount", "sum"), repayment_current_dpd_max=("days_past_due", "max"),
        repayment_historical_dpd_max=("max_days_past_due", "max"), repayment_historical_dpd_mean=("max_days_past_due", "mean"),
        repayment_late_share=("ever_dpd_1", "mean"), repayment_dpd30_share=("ever_dpd_30", "mean"),
        repayment_dpd90_share=("ever_dpd_90", "mean"))
    features["repayment_paid_ratio"] = safe_ratio(features.repayment_amount_paid, features.repayment_amount_due)
    recent = repay[repay.recency_days.le(180)].groupby(KEY).agg(
        repayment_dpd_mean_180d=("max_days_past_due", "mean"), repayment_unpaid_180d=("unpaid_amount", "sum"))
    attach(features.join(recent))
    base["has_repayments"] = base.repayment_installment_count.notna().astype(int)
    del repay, features, recent
    gc.collect()

    name = "account_behavior"
    account = read(name, ["month"])
    inspect(account, name, [KEY, "month"])
    account = at_cutoff(account, name, ["month"])
    completed = account.month + pd.offsets.MonthBegin(1)
    audits[name]["incomplete_month_at_cutoff"] = int(completed.gt(account.cutoff).sum())
    audits[name]["balance_identity_mismatches"] = int((account.opening_balance + account.inflows - account.outflows).ne(account.closing_balance).sum())
    audits[name]["negative_amount_cells"] = int(account.select_dtypes(include="number").lt(0).sum().sum())
    account = account[completed.le(account.cutoff)].copy()
    account["net_inflow"] = account.inflows - account.outflows
    account["recency_days"] = (account.cutoff - account.month).dt.total_seconds() / 86400
    features = account.groupby(KEY).agg(
        account_month_count=("month", "size"), account_income_mean=("income_like_credits", "mean"),
        account_income_std=("income_like_credits", "std"), account_inflow_mean=("inflows", "mean"),
        account_outflow_mean=("outflows", "mean"), account_net_inflow_mean=("net_inflow", "mean"),
        account_average_balance_mean=("average_balance", "mean"), account_minimum_balance_min=("minimum_balance", "min"),
        account_low_balance_days_mean=("low_balance_days", "mean"), account_last_recency_days=("recency_days", "min"))
    features["account_income_cv"] = safe_ratio(features.account_income_std, features.account_income_mean)
    last = account.sort_values("month").groupby(KEY).tail(1).set_index(KEY)
    for col in ["closing_balance", "income_like_credits", "average_balance", "low_balance_days"]:
        features[f"account_last_{col}"] = last[col]
    # Three complete months immediately preceding the decision month.
    start = account.cutoff.dt.to_period("M").dt.to_timestamp() - pd.DateOffset(months=3)
    recent = account[account.month.ge(start)].groupby(KEY).agg(
        account_income_mean_3m=("income_like_credits", "mean"), account_outflow_mean_3m=("outflows", "mean"),
        account_balance_mean_3m=("average_balance", "mean"), account_low_balance_days_mean_3m=("low_balance_days", "mean"))
    attach(features.join(recent))
    base["has_account"] = base.account_month_count.notna().astype(int)
    base["observed_declared_income_ratio"] = safe_ratio(base.account_income_mean, base.declared_monthly_income)
    base["installment_observed_income_ratio"] = safe_ratio(base.proposed_monthly_installment, base.account_income_mean)
    base["balance_installment_ratio"] = safe_ratio(base.account_last_closing_balance, base.proposed_monthly_installment)
    del account, features, last, recent
    gc.collect()

    name = "bureau_observations"
    bureau = read(name, ["report_datetime", "available_datetime"])
    inspect(bureau, name, "bureau_observation_id")
    audits[name]["unknown_application_ids"] = len(set(bureau.application_id) - (historical_ids | set(base.application_id)))
    audits[name]["availability_before_report"] = int(bureau.available_datetime.lt(bureau.report_datetime).sum())
    audits[name]["score_out_of_range"] = int((bureau.score.notna() & ~bureau.score.between(300, 850)).sum())
    expected_util = safe_ratio(bureau.revolving_outstanding_balance, bureau.revolving_credit_limit)
    audits[name]["utilization_max_absolute_difference"] = float((bureau.utilization - expected_util).abs().max())
    bureau = at_cutoff(bureau, name, ["report_datetime", "available_datetime"])
    bureau = bureau[bureau.report_datetime.lt(bureau.cutoff) & bureau.available_datetime.lt(bureau.cutoff)].copy()
    latest = bureau.sort_values(["report_datetime", "available_datetime", "bureau_observation_id"]).groupby([KEY, "bureau_provider"]).tail(1).copy()
    latest["report_age_days"] = (latest.cutoff - latest.report_datetime).dt.total_seconds() / 86400
    latest["delivery_lag_days"] = (latest.available_datetime - latest.report_datetime).dt.total_seconds() / 86400
    features = latest.groupby(KEY).agg(
        bureau_provider_count=("bureau_provider", "size"), bureau_scored_count=("score", "count"),
        bureau_score_mean=("score", "mean"), bureau_score_min=("score", "min"), bureau_score_max=("score", "max"),
        bureau_debt_max=("outstanding_balance", "max"), bureau_debt_mean=("outstanding_balance", "mean"),
        bureau_active_accounts_max=("active_accounts", "max"), bureau_utilization_max=("utilization", "max"),
        bureau_inquiries_max=("inquiries_last_6m", "max"), bureau_delinquent_accounts_max=("delinquent_accounts_last_12m", "max"),
        bureau_report_age_mean=("report_age_days", "mean"), bureau_delivery_lag_mean=("delivery_lag_days", "mean"))
    features["bureau_score_spread"] = features.bureau_score_max - features.bureau_score_min
    # Provider balances can overlap; summing would imply unsupported unique debt.
    for col in ["score", "utilization", "outstanding_balance", "report_age_days"]:
        pivot = latest.pivot(index=KEY, columns="bureau_provider", values=col)
        pivot.columns = [f"bureau_{provider}_{col}" for provider in pivot.columns]
        features = features.join(pivot)
    attach(features)
    base["bureau_debt_income_ratio"] = safe_ratio(base.bureau_debt_max, base.declared_monthly_income)
    latest.drop(columns="cutoff").to_parquet(PROCESSED / "bureau_latest.parquet", index=False)
    del bureau, latest, features, pivot
    gc.collect()

    name = "ecommerce_transactions"
    ecom = read(name, ["transaction_datetime"])
    inspect(ecom, name, "order_item_id")
    # A transaction row is an ITEM, not an order. Verify before deduplicating.
    order_fields = [KEY, "transaction_datetime", "order_total", "payment_method", "order_status"]
    inconsistent = ecom.groupby("order_id")[order_fields].nunique(dropna=False).gt(1).any(axis=1)
    audits[name]["inconsistent_order_fields"] = int(inconsistent.sum())
    assert not inconsistent.any()
    order_sums = ecom.groupby("order_id").item_total.sum()
    orders = ecom.drop_duplicates("order_id")[["order_id"] + order_fields].copy()
    audits[name]["order_total_mismatches"] = int(orders.order_id.map(order_sums).ne(orders.order_total).sum())
    audits[name]["orders"] = len(orders)
    audits[name]["items"] = len(ecom)
    audits[name]["naive_completed_spend_idr"] = int(ecom.loc[ecom.order_status.eq("completed"), "order_total"].sum())
    audits[name]["correct_completed_spend_idr"] = int(orders.loc[orders.order_status.eq("completed"), "order_total"].sum())
    category_spend = ecom[ecom.order_status.eq("completed")].groupby([KEY, "product_category"]).item_total.sum().unstack(fill_value=0)
    # category_spend is safe only when all raw events precede the cutoff (audited below).
    del ecom, order_sums, inconsistent
    gc.collect()
    orders = at_cutoff(orders, name, ["transaction_datetime"])
    assert audits[name]["transaction_datetime_at_or_after_cutoff"] == 0
    orders = orders[orders.transaction_datetime.lt(orders.cutoff)].copy()
    orders["cancelled"] = orders.order_status.eq("cancelled").astype(int)
    features = orders.groupby(KEY).agg(ecom_order_count=("order_id", "size"), ecom_cancel_rate=("cancelled", "mean"))
    completed = orders[orders.order_status.eq("completed")].copy()
    completed["recency_days"] = (completed.cutoff - completed.transaction_datetime).dt.total_seconds() / 86400
    features = features.join(completed.groupby(KEY).agg(
        ecom_completed_count=("order_id", "size"), ecom_spend_sum=("order_total", "sum"),
        ecom_order_value_mean=("order_total", "mean"), ecom_order_value_std=("order_total", "std"),
        ecom_last_purchase_recency_days=("recency_days", "min")))
    for days in [90, 180]:
        recent = completed[completed.recency_days.le(days)].groupby(KEY).agg(
            **{f"ecom_completed_count_{days}d": ("order_id", "size"), f"ecom_spend_{days}d": ("order_total", "sum")})
        features = features.join(recent.reindex(features.index, fill_value=0))
    payment = pd.crosstab(completed[KEY], completed.payment_method, normalize="index").add_prefix("ecom_payment_share_")
    features = features.join(payment)
    category_share = category_spend.div(category_spend.sum(axis=1), axis=0).add_prefix("ecom_category_share_")
    attach(features.join(category_share))
    base["ecom_spend90_income_ratio"] = safe_ratio(base.ecom_spend_90d / 3, base.declared_monthly_income)
    del orders, completed, features, category_spend, category_share, payment, recent
    gc.collect()

    base.replace([np.inf, -np.inf], np.nan, inplace=True)
    base.to_parquet(PROCESSED / "application_features.parquet", index=False)
    pd.DataFrame(inventory).to_csv(TABLES / "table_inventory.csv", index=False)
    pd.DataFrame(missing).to_csv(TABLES / "column_quality.csv", index=False)
    pd.DataFrame(coverage).to_csv(TABLES / "history_coverage.csv", index=False)
    (OUT / "quality_audit.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False))
    print(f"Saved {len(base):,} application rows, {len(base.columns)} columns.", flush=True)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Render existing feature/benchmark results without retraining")
    args = parser.parse_args()
    base = pd.read_parquet(PROCESSED / "application_features.parquet") if args.reuse_features or args.report_only else build_features()
    from reporting import make_report
    cached = None
    if args.report_only:
        cached = (pd.read_csv(TABLES / "model_benchmarks.csv"),
                  pd.read_csv(TABLES / "feature_importance_gain.csv").head(15).to_dict("records"))
    make_report(base, run_models=not (args.skip_model or args.report_only), benchmark_results=cached)


if __name__ == "__main__":
    main()
