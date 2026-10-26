"""Exact analogue of the competition target on each customer's PREVIOUS loans: did the loan reach end-of-day
DPD >= 90 within 12 calendar months of origination (previous decision_datetime)? Loans whose 12-month window
ends after the current decision are censored (only an observed hit counts). Label-free w.r.t. the current
target; pre-decision data only. Features per application, written in 237's per-fold format (identical per fold).
Usage: cd eda && ../.venv/bin/python 247_analog_target.py
"""
import os

import numpy as np
import polars as pl

import valid2 as v
from common import ROOT

D = ROOT / "data"
apps = pl.concat([pl.read_csv(D / f"{s}_applications.csv", try_parse_dates=True).select("application_id", "customer_id", "decision_datetime") for s in ("train", "test")])
prev = pl.read_csv(D / "previous_applications.csv", try_parse_dates=True).filter(pl.col("loan_id").is_not_null()) \
    .select("loan_id", "customer_id", pl.col("decision_datetime").alias("orig"), "term_months")
rep = pl.read_csv(D / "repayment_history.csv", try_parse_dates=True)
cut = apps.select("customer_id", pl.col("decision_datetime").alias("cut"))
r = rep.join(prev.select("loan_id", "orig"), on="loan_id").join(cut, on="customer_id").filter(pl.col("due_date") < pl.col("cut"))
# DPD>=90 reached within the 12-month window: an installment due inside the window whose max DPD hit 90 while the
# 90th day (due + 90d) is also inside the window and before the current cutoff
r = r.with_columns(win_end=pl.col("orig") + pl.duration(days=365),
                   hit=(pl.col("max_days_past_due") >= 90) & (pl.col("due_date") + pl.duration(days=90) <= pl.col("orig") + pl.duration(days=365)))
loan = r.group_by("loan_id", "customer_id").agg(pl.col("hit").any().alias("hit"), pl.col("win_end").first(), pl.col("cut").first(),
                                                pl.col("orig").first(), pl.col("max_days_past_due").max().alias("mdpd"))
loan = loan.with_columns(full=pl.col("win_end") <= pl.col("cut"))
f = loan.group_by("customer_id").agg(
    an_loans=pl.len(), an_full=pl.col("full").sum(),
    an_hit_any=pl.col("hit").sum(),
    an_hit_full=(pl.col("hit") & pl.col("full")).sum(),
    an_rate_full=((pl.col("hit") & pl.col("full")).sum() / pl.col("full").sum().clip(1)),
    an_rate_bayes=((pl.col("hit").sum() + 0.136) / (pl.col("full").sum() + pl.col("hit").sum() * 0 + 1)),
    an_days_since_hit=((pl.col("cut") - pl.col("orig")).dt.total_seconds() / 86400).filter(pl.col("hit")).min(),
    an_open_censored=(~pl.col("full")).sum(),
    an_censored_maxdpd=pl.col("mdpd").filter(~pl.col("full")).max())
out = apps.select("application_id", "customer_id").join(f, on="customer_id", how="left").drop("customer_id")
for k in range(5):
    path = ROOT / f"results/v66/series_oof/analog_f{k}.parquet"
    tmp = path.with_name(path.name + ".tmp"); out.write_parquet(tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)
t = pl.read_csv(D / "train_applications.csv").select("application_id", "target").join(out, on="application_id")
y = t["target"].to_numpy()
for c in [c for c in out.columns if c.startswith("an_")]:
    x = t[c].to_numpy().astype(float)
    print(f"{c:22s} coverage {np.isfinite(x).mean():.2f} univariate KS {v.ks_nan(y, x):.4f}")
hit = t.filter(pl.col("an_hit_any") > 0)
print("customers with an analogue default:", hit.height, "their target rate", round(hit["target"].mean(), 4), "vs overall", round(y.mean(), 4))
