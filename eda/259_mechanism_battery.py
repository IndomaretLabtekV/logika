"""Bet 4: a battery of label-free mechanism guesses, screened with the 256 likelihood-ratio test on the trees+Markov OOF.
Anchor: mk_hit3 on the base family gave chi2 17.8 and later +0.003 public. All events are before decision_datetime.

repayment  rr_late_inst1     share of loans late at their first installment
           rr_early_minus_late late rate in installments 1-3 minus 4+
           rr_run_before     mean on-time run before the first late installment (loans that went late)
           rr_frac_loans_late share of loans ever late
           rr_short_paid     mean amount_paid / amount_due over short payments
           rr_late_big       late rate on the customer's larger-than-own-median installments minus smaller ones
           rr_cure_dpd       mean max_dpd at which a late spell ended (cured)
previous   pa_haircut_trend  slope of approved/requested over the customer's approved applications
           pa_req_growth     log requested amount, last minus first
           pa_gap_cv         coefficient of variation of gaps between applications
           pa_rej_after_late rejections in the 180 days after a late installment, per late installment
ecommerce  es_groc_shift     groceries share of items, last 90 days minus before
           es_cod_shift      COD share of orders, last 90 days minus before
           es_cancel_shift   cancel share, last 90 days minus before
           es_qty_premium    item quantity z-score vs category norm (all events, label-free)
           es_basket_div     categories per order, last 90 days minus before
           es_weekend        weekend share of orders
bureau     bu_out_growth     outstanding growth per 30 days between first and last pull
           bu_delinq_entry   share of consecutive pulls where delinquent accounts went from 0 to >0
           bu_score_slope    score slope per year over pulls
account    ac2_inflow_cv     coefficient of variation of non-income inflows
           ac2_min_avg       mean minimum_balance / average_balance
           ac2_outflow_spike max monthly outflow over median outflow
"""
import importlib

import numpy as np
import polars as pl

import common as c

lr = importlib.import_module("256_lr_screen")
scr = importlib.import_module("255_residual_screen")


def repayment(apps):
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(late=(pl.col("max_days_past_due") > 0).cast(pl.Int8),
                       idx=pl.col("due_date").rank("ordinal").over("loan_id"),
                       med=pl.col("amount_due").median().over("customer_id"))
         .with_columns(prev_late=pl.col("late").shift(1).over("loan_id")))
    first_late = r.filter(pl.col("late") == 1).group_by("loan_id").agg(pl.col("idx").min().alias("fl"))
    loans = r.group_by("customer_id", "loan_id").agg(pl.col("late").max().alias("any"), pl.col("late").first().alias("l1"))
    loans = loans.join(first_late, on="loan_id", how="left")
    a = loans.group_by("customer_id").agg(rr_late_inst1=pl.col("l1").mean(), rr_frac_loans_late=pl.col("any").mean(),
                                          rr_run_before=(pl.col("fl") - 1).mean())
    b = r.group_by("customer_id").agg(
        rr_early_minus_late=pl.col("late").filter(pl.col("idx") <= 3).mean() - pl.col("late").filter(pl.col("idx") > 3).mean(),
        rr_short_paid=(pl.col("amount_paid") / pl.col("amount_due")).filter(pl.col("amount_paid") < pl.col("amount_due")).mean(),
        rr_late_big=pl.col("late").filter(pl.col("amount_due") > pl.col("med")).mean()
        - pl.col("late").filter(pl.col("amount_due") <= pl.col("med")).mean(),
        rr_cure_dpd=pl.col("max_days_past_due").shift(1).filter((pl.col("late") == 0) & (pl.col("prev_late") == 1)).mean())
    return a.join(b, on="customer_id", how="full", coalesce=True)


def previous(apps):
    p = c.load("previous_applications").sort("customer_id", "application_datetime").with_columns(
        t=pl.col("application_datetime").dt.epoch("d").cast(pl.Float64))
    a = p.group_by("customer_id").agg(
        pa_req_growth=pl.col("requested_amount").log().last() - pl.col("requested_amount").log().first(),
        pa_gap_cv=pl.col("t").diff().std() / pl.col("t").diff().mean())
    ap = (p.filter(pl.col("approved_amount").is_not_null())
          .with_columns(hc=pl.col("approved_amount") / pl.col("requested_amount"))
          .group_by("customer_id").agg(pa_haircut_trend=pl.cov("t", "hc") / pl.col("t").var()))
    late = (c.load("repayment_history").filter(pl.col("max_days_past_due") > 0)
            .select("customer_id", lt=pl.col("due_date")))
    rej = p.filter(pl.col("decision") == "rejected").select("customer_id", rt=pl.col("application_datetime"))
    j = (late.join(rej, on="customer_id")
         .filter((pl.col("rt") > pl.col("lt")) & (pl.col("rt") <= pl.col("lt") + pl.duration(days=180)))
         .group_by("customer_id").len())
    nl = late.group_by("customer_id").agg(pl.len().alias("nl"))
    rl = nl.join(j, on="customer_id", how="left").select("customer_id", pa_rej_after_late=pl.col("len").fill_null(0) / pl.col("nl"))
    print(p["decision"].value_counts())
    return a.join(ap, on="customer_id", how="left").join(rl, on="customer_id", how="left")


def ecommerce(apps):
    cut = apps.select("customer_id", "decision_datetime")
    e = (c.load("ecommerce_transactions").join(cut, on="customer_id")
         .filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
         .with_columns(recent=(pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_days() <= 90,
                       qz=((pl.col("quantity").cast(pl.Float64).log1p() - pl.col("quantity").cast(pl.Float64).log1p().mean().over("product_category"))
                           / pl.col("quantity").cast(pl.Float64).log1p().std().over("product_category"))))
    it = e.group_by("customer_id").agg(
        es_groc_shift=(pl.col("product_category") == "groceries").filter("recent").mean()
        - (pl.col("product_category") == "groceries").filter(~pl.col("recent")).mean(),
        es_qty_premium=pl.col("qz").mean())
    o = e.group_by("customer_id", "order_id").agg(pl.col("recent").first(), pl.col("payment_method").first(),
                                                   pl.col("order_status").first(), pl.col("transaction_datetime").first(),
                                                   pl.col("product_category").n_unique().alias("ncat"))
    od = o.group_by("customer_id").agg(
        es_cod_shift=(pl.col("payment_method") == "COD").filter("recent").mean() - (pl.col("payment_method") == "COD").filter(~pl.col("recent")).mean(),
        es_cancel_shift=(pl.col("order_status") != "completed").filter("recent").mean()
        - (pl.col("order_status") != "completed").filter(~pl.col("recent")).mean(),
        es_basket_div=pl.col("ncat").filter("recent").mean() - pl.col("ncat").filter(~pl.col("recent")).mean(),
        es_weekend=(pl.col("transaction_datetime").dt.weekday() >= 6).mean())
    print(e["payment_method"].unique().to_list(), e["order_status"].unique().to_list())
    return it.join(od, on="customer_id", how="full", coalesce=True)


def bureau(apps):
    cut = apps.select("customer_id", "decision_datetime")
    b = (c.load("bureau_observations").join(cut, on="customer_id")
         .filter(pl.col("available_datetime") < pl.col("decision_datetime"))
         .sort("customer_id", "report_datetime")
         .with_columns(t=pl.col("report_datetime").dt.epoch("d").cast(pl.Float64),
                       pd=pl.col("delinquent_accounts_last_12m").shift(1).over("customer_id")))
    return b.group_by("customer_id").agg(
        bu_out_growth=(pl.col("outstanding_balance").last() - pl.col("outstanding_balance").first())
        / ((pl.col("t").last() - pl.col("t").first()) / 30 + 1),
        bu_delinq_entry=((pl.col("pd") == 0) & (pl.col("delinquent_accounts_last_12m") > 0)).sum() / (pl.col("pd") == 0).sum(),
        bu_score_slope=(pl.cov("t", "score") / pl.col("t").var()) * 365)


def account(apps):
    a = c.load("account_behavior").with_columns(nonin=pl.col("inflows") - pl.col("income_like_credits"))
    return a.group_by("customer_id").agg(
        ac2_inflow_cv=pl.col("nonin").std() / (pl.col("nonin").mean().abs() + 1),
        ac2_min_avg=(pl.col("minimum_balance") / (pl.col("average_balance").abs() + 1)).mean(),
        ac2_outflow_spike=pl.col("outflows").max() / (pl.col("outflows").median() + 1))


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime")
    out = apps.select("application_id", "customer_id")
    for part in (repayment(apps), previous(apps), ecommerce(apps), bureau(apps), account(apps)):
        out = out.join(part, on="customer_id", how="left")
    out = out.drop("customer_id")
    out = out.with_columns([pl.col(k).cast(pl.Float64).fill_nan(None) for k in out.columns if k != "application_id"])
    out.write_parquet(c.PQ / "f259_mechanism_battery.parquet")
    print(out.null_count())


if __name__ == "__main__":
    main()
