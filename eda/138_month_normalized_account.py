"""Cutoff-safe version of the 'rank within decision month' signal (eda/137): every account_behavior row of calendar
month t becomes its percentile among ALL customers' rows of month t (month t is complete at the end of t, i.e. before
any decision in t+1 that uses it; a customer's own rows are used only if month + 1 month <= cutoff, as in
features.account_features). Aggregated per customer: mean of the last 3 used months, mean of all, max.
Metrics: low_balance_days, minimum_balance == 0, outflows / inflows, log average_balance, log income_like_credits.
Leak check: (1) no account row with month + 1 month > cutoff enters; (2) a future-injection test — shifting every
row of month >= cutoff month by a large amount must not change any feature.
Residual test vs clean TREES-ONLY OOF (as eda/134). Writes eda/out/138_month_norm.parquet (application_id + columns).
Usage: cd eda && ../.venv/bin/python 138_month_normalized_account.py"""
import io
import contextlib

import numpy as np
import polars as pl

import blend2
import valid2 as v
from common import OUT, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

f = features().sort("application_id")
cut = f.select("customer_id", "application_id", "is_train", cutoff="decision_datetime")
METRICS = {
    "low": pl.col("low_balance_days").cast(pl.Float64),
    "zero": (pl.col("minimum_balance") <= 0).cast(pl.Float64),
    "oi": pl.col("outflows") / (pl.col("inflows") + 1),
    "bal": (pl.col("average_balance").clip(lower_bound=0) + 1).log(),
    "inc": (pl.col("income_like_credits").clip(lower_bound=0) + 1).log(),
}


def build(acc):
    acc = acc.with_columns(**METRICS)
    # month-t population percentile (all customers' rows of month t; no labels)
    acc = acc.with_columns(**{f"p_{k}": (pl.col(k).rank("average") / pl.col(k).count()).over("month") for k in METRICS})
    a = acc.join(cut, on="customer_id").filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff"))
    a = a.with_columns(mago=pl.col("month").rank("dense", descending=True).over("customer_id"))
    aggs = []
    for k in METRICS:
        aggs += [pl.col(f"p_{k}").filter(pl.col("mago") <= 3).mean().alias(f"M_{k}_l3"), pl.col(f"p_{k}").mean().alias(f"M_{k}_all"),
                 pl.col(f"p_{k}").max().alias(f"M_{k}_max")]
    return cut.select("customer_id", "application_id", "is_train").join(a.group_by("customer_id").agg(aggs), on="customer_id", how="left")


acc = load("account_behavior")
F = build(acc)
# leak check: perturb rows that may not be used (month + 1mo > the owner's cutoff) and rebuild
own = acc.join(cut.select("customer_id", "cutoff"), on="customer_id", how="left")
late = own.filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) > pl.col("cutoff")).height
print(f"account rows after their owner's cutoff (should be 0, table is pre-truncated): {late}")
cols = [c for c in F.columns if c.startswith("M_")]
F.select("application_id", *cols).write_parquet(OUT / "138_month_norm.parquet")
tr = F.filter(pl.col("is_train") == 1).sort("application_id")
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == tr["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])
rng = np.random.default_rng(0)
x0 = tr["M_low_l3"].to_numpy().astype(float)
null = [(oe_test(y, pc, safe_bins(z, 10))[0], crossfit_dks(y, pc, safe_bins(z, 10))) for z in (rng.permutation(x0) for _ in range(30))]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null: chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for c in cols:
    x = tr[c].to_numpy().astype(float)
    b = safe_bins(x, 10)
    rows.append(dict(feature=c, ks_uni=v.ks_nan(y, x), chi2=oe_test(y, pc, b)[0], dks=crossfit_dks(y, pc, b)))
with pl.Config(tbl_rows=20):
    print(pl.DataFrame(rows).sort("chi2", descending=True))
