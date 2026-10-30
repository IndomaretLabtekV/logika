"""Bet 8: feature families from the teammate's 638-column Kumo input spec that features.parquet does not have.
Only a written spec was available, so these are re-implementations, not copies. All label-free; every table is already
cut at decision_datetime by the organisers (checked: 0 rows after the decision in every table).

cx_  credit exposure: previous approved loans still running at the decision, their installments and principal,
     new + running installment against declared and observed income, observed vs declared income, outflow / income
ll_  loan-by-loan repayment: each earlier loan summarised first (mean/max DPD, share 30+/90+, latest DPD, recent vs
     average severity, arrears fraction), then mean and max per customer; loans that ever hit 90+; loans ending late
al_  account liquidity over all months and the last 3: median income credits, median outflow, minimum balance,
     share of months whose balance fell, share of low-balance days
bw_  bureau: recency-weighted score (half-lives 90 and 365 days), score volatility, 5/50/95th percentiles of score,
     utilisation, outstanding balance and inquiries across reports, score IQR
pl_  prices: price level (categories weighted equally) and its 90-day version, number of categories, price level vs
     income and vs loan size, payment-method-adjusted level, cancelled-item level, basket shape
cs_  category spend over 90 and 365 days as a monthly share of declared income, and orders containing each category
Usage: cd eda && ../.venv/bin/python 271_spec_block.py
"""
import importlib

import numpy as np
import polars as pl

import common as c
from features import PP_CATS

px = importlib.import_module("268_price_latent")


def days(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds() / 86400


def exposure(cut):
    p = (c.load("previous_applications").drop("application_id")
         .join(cut.select("customer_id", "cutoff"), on="customer_id")
         .filter((pl.col("decision") == "approved") & (pl.col("decision_datetime") <= pl.col("cutoff")))
         .with_columns(running=pl.col("decision_datetime").dt.offset_by(pl.format("{}mo", pl.col("term_months"))) > pl.col("cutoff")))
    return p.group_by("customer_id").agg(
        cx_run_n=pl.col("running").sum(), cx_run_inst=pl.col("scheduled_monthly_installment").filter("running").sum(),
        cx_run_principal=pl.col("approved_amount").filter("running").sum(), cx_appr_total=pl.col("approved_amount").sum(),
        cx_term_mean=pl.col("term_months").mean())


def loans(cut):
    r = (c.load("repayment_history").join(cut.select("customer_id", "cutoff"), on="customer_id")
         .sort("customer_id", "loan_id", "due_date")
         .with_columns(k=pl.col("due_date").rank("ordinal").over("loan_id"), n=pl.len().over("loan_id")))
    dpd = pl.col("max_days_past_due")
    per = r.group_by("customer_id", "loan_id").agg(
        dmean=dpd.mean(), dmax=dpd.max(), s30=(dpd >= 30).mean(), s90=(dpd >= 90).mean(),
        last=dpd.last(), cur=(pl.col("amount_paid") < pl.col("amount_due")).last().cast(pl.Float64),
        recent=dpd.tail(3).mean() - dpd.mean(),
        arrears=(pl.col("amount_due") - pl.col("amount_paid")).sum() / pl.col("amount_due").sum())
    keys = ["dmean", "dmax", "s30", "s90", "last", "cur", "recent", "arrears"]
    return per.group_by("customer_id").agg(
        *[pl.col(k).mean().alias(f"ll_{k}_mean") for k in keys], *[pl.col(k).max().alias(f"ll_{k}_max") for k in keys],
        ll_n90=(pl.col("dmax") >= 90).sum(), ll_n_last_late=(pl.col("last") > 0).sum(), ll_n=pl.len())


def liquidity(cut):
    a = (c.load("account_behavior").join(cut.select("customer_id"), on="customer_id")
         .with_columns(l3=pl.col("month").rank("dense", descending=True).over("customer_id") <= 3,
                       fell=(pl.col("closing_balance") < pl.col("opening_balance")).cast(pl.Float64),
                       low=pl.col("low_balance_days") / pl.col("month").dt.month_end().dt.day()))
    out = []
    for tag, f in (("all", pl.lit(True)), ("l3", pl.col("l3"))):
        out += [pl.col("income_like_credits").filter(f).median().alias(f"al_inc_med_{tag}"),
                pl.col("outflows").filter(f).median().alias(f"al_out_med_{tag}"),
                pl.col("minimum_balance").filter(f).min().alias(f"al_minbal_{tag}"),
                pl.col("fell").filter(f).mean().alias(f"al_fell_{tag}"), pl.col("low").filter(f).mean().alias(f"al_low_{tag}")]
    return a.group_by("customer_id").agg(*out)


def bureau(cut):
    b = (c.load("bureau_observations").join(cut.select("customer_id", "cutoff"), on="customer_id")
         .with_columns(age=days("cutoff", "report_datetime")))
    out = [pl.col("score").std().alias("bw_score_sd"),
           (pl.col("score").quantile(0.75) - pl.col("score").quantile(0.25)).alias("bw_score_iqr")]
    for h in (90, 365):
        w = (-pl.col("age") * np.log(2) / h).exp() * pl.col("score").is_not_null()
        out.append(((w * pl.col("score").fill_null(0)).sum() / w.sum()).alias(f"bw_score_hl{h}"))
    for col, tag in (("score", "score"), ("utilization", "util"), ("outstanding_balance", "debt"), ("inquiries_last_6m", "inq")):
        out += [pl.col(col).cast(pl.Float64).quantile(q).alias(f"bw_{tag}_p{int(q * 100)}") for q in (0.05, 0.5, 0.95)]
    return b.group_by("customer_id").agg(*out)


def prices(apps):
    e = px.items(apps).with_columns(recent=days("cutoff", "transaction_datetime") <= 90,
                                    done=pl.col("order_status") == "completed")
    # payment-method adjustment: remove the population mean z of each method (label-free)
    e = e.with_columns(zpm=pl.col("z") - pl.col("z").mean().over("payment_method"))
    cat = e.filter("done").group_by("customer_id", "product_category").agg(
        z=pl.col("z").mean(), zr=pl.col("z").filter("recent").mean(), zpm=pl.col("zpm").mean())
    lvl = cat.group_by("customer_id").agg(pl_level=pl.col("z").mean(), pl_level_90=pl.col("zr").mean(),
                                          pl_level_pm=pl.col("zpm").mean(), pl_n_cats=pl.len())
    canc = e.filter(~pl.col("done")).group_by("customer_id").agg(pl_level_cancel=pl.col("z").mean())
    o = e.group_by("customer_id", "order_id").agg(items=pl.len(), units=pl.col("quantity").sum(),
                                                  cats=pl.col("product_category").n_unique(), value=pl.col("order_total").first())
    bask = o.group_by("customer_id").agg(pl_bask_items=pl.col("items").mean(), pl_bask_units=pl.col("units").mean(),
                                         pl_bask_cats=pl.col("cats").mean(), pl_bask_value_med=pl.col("value").median(),
                                         pl_bask_value_max=pl.col("value").max())
    return lvl.join(canc, on="customer_id", how="left").join(bask, on="customer_id", how="left")


def category_share(cut):
    e = (c.load("ecommerce_transactions").join(cut.select("customer_id", "cutoff"), on="customer_id")
         .filter(pl.col("order_status") == "completed").with_columns(age=days("cutoff", "transaction_datetime")))
    out = []
    for cat in PP_CATS:
        m = pl.col("product_category") == cat
        out += [pl.col("item_total").filter(m & (pl.col("age") <= 90)).sum().alias(f"cs_{cat}_90"),
                pl.col("item_total").filter(m & (pl.col("age") <= 365)).sum().alias(f"cs_{cat}_365"),
                pl.col("order_id").filter(m).n_unique().alias(f"cs_{cat}_orders")]
    return e.group_by("customer_id").agg(*out)


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime", "proposed_monthly_installment",
                               "loan_amount", "declared_monthly_income")
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"))
    d = apps
    for part in (exposure(cut), loans(cut), liquidity(cut), bureau(cut), prices(apps), category_share(cut)):
        d = d.join(part, on="customer_id", how="left", maintain_order="left")
    inc = pl.col("declared_monthly_income").cast(pl.Float64)
    z = lambda e: (e - e.mean()) / e.std()  # noqa: E731
    d = d.with_columns(
        cx_inst_total_to_inc=(pl.col("proposed_monthly_installment") + pl.col("cx_run_inst").fill_null(0)) / inc,
        cx_inst_total_to_obs=(pl.col("proposed_monthly_installment") + pl.col("cx_run_inst").fill_null(0)) / (pl.col("al_inc_med_l3") + 1e5),
        cx_obs_vs_decl=pl.col("al_inc_med_l3") / inc, cx_out_to_inc=pl.col("al_out_med_l3") / (pl.col("al_inc_med_l3") + 1e5),
        pl_level_vs_inc=pl.col("pl_level") - z(inc.log()), pl_level_vs_loan=pl.col("pl_level") - z(pl.col("loan_amount").cast(pl.Float64).log()),
        **{f"cs_{cat}_{w}": pl.col(f"cs_{cat}_{w}") / (inc * (3 if w == 90 else 12)) for cat in PP_CATS for w in (90, 365)})
    d = d.drop("customer_id", "decision_datetime", "proposed_monthly_installment", "loan_amount", "declared_monthly_income")
    d = d.with_columns([pl.col(k).cast(pl.Float64).fill_nan(None) for k in d.columns if k != "application_id"])
    d = d.with_columns([pl.when(pl.col(k).is_infinite()).then(None).otherwise(pl.col(k)).alias(k) for k in d.columns if k != "application_id"])
    d.write_parquet(c.PQ / "f271_spec_block.parquet")
    print(d.shape)
    print(d.select([pl.col(k).is_null().mean().alias(k) for k in d.columns[1:]]).transpose(include_header=True)
          .sort("column_0", descending=True).head(8))


if __name__ == "__main__":
    main()
