"""Repayment history: delinquency depth, recency, active obligations."""
import polars as pl
from common import analyze, features

f = features()
cols = [c for c in f.columns if c.startswith("rp_")] + ["dti_total"]
analyze(f, cols, "05_repay", top_plot=16)
tr = f.filter(pl.col("is_train") == 1).with_columns(
    grp=pl.when(pl.col("rp_n_loans").is_null()).then(pl.lit("no repay history"))
    .when(pl.col("rp_maxdpd_max") == 0).then(pl.lit("always on time"))
    .when(pl.col("rp_maxdpd_max") < 30).then(pl.lit("1-29"))
    .when(pl.col("rp_maxdpd_max") < 90).then(pl.lit("30-89")).otherwise(pl.lit("90+")))
print(tr.group_by("grp").agg(n=pl.len(), bad=pl.col("target").mean()).sort("bad"))
print("currently delinquent (unpaid dpd>0):", tr.group_by(pl.col("rp_cur_dpd_max") > 0).agg(pl.len(), pl.col("target").mean()))
print("has_history but no repayment rows (rejected/cancelled only):",
      tr.filter((pl.col("has_history") == 1) & pl.col("rp_n_loans").is_null()).select(pl.len(), pl.col("target").mean()))
