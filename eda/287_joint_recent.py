"""Bet 15 (from 285): recent deterioration across channels, measured against each customer's own baseline, and combined.

285: before the decision, defaulters' order rate surges (2.8x in the last day), their account net flow collapses in the
last ~4 months, their order values rise, and the bureau pull made for this application shows more delinquency.
features.parquet measures each channel alone; trees combine many weak additive signals poorly. Label-free:
  jr_surge7, jr_surge30   log((orders in last 7/30 d per day + e) / (orders 120-365 d before per day + e))
  jr_flow1, jr_flow3      net flow / income-like credits, last month and last 3 months, minus months 5-12
  jr_low1                 low-balance-day share of the last month minus months 5-12
  jr_price                mean item price z of the last 90 d minus older than 182 d (from 268 item z)
  jr_bdel, jr_bscore      selected bureau pull minus the mean of earlier reports (delinquent accounts, score)
  jr_index                mean of the available standardised signals, signs fixed a priori (worse = higher)
  jr_x_*                  pairwise products of the standardised surge, flow, price signals
Usage: cd eda && ../.venv/bin/python 287_joint_recent.py
"""
import importlib

import numpy as np
import polars as pl

import common as c

px = importlib.import_module("268_price_latent")


def days(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds() / 86400


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime")
    cut = apps.select("customer_id", "decision_datetime")
    o = (c.load("ecommerce_transactions").group_by("order_id").agg(pl.col("customer_id", "transaction_datetime").first())
         .join(cut, on="customer_id").with_columns(age=days("decision_datetime", "transaction_datetime")))
    eps = 1 / 60
    ec = o.group_by("customer_id").agg(
        n7=(pl.col("age") < 7).sum(), n30=(pl.col("age") < 30).sum(), nb=((pl.col("age") >= 120) & (pl.col("age") < 365)).sum()
    ).select("customer_id", jr_surge7=((pl.col("n7") / 7 + eps) / (pl.col("nb") / 245 + eps)).log(),
             jr_surge30=((pl.col("n30") / 30 + eps) / (pl.col("nb") / 245 + eps)).log())
    a = (c.load("account_behavior").with_columns(mago=pl.col("month").rank("dense", descending=True).over("customer_id"),
                                                 net=(pl.col("inflows") - pl.col("outflows")) / (pl.col("income_like_credits") + 1e5),
                                                 low=pl.col("low_balance_days") / 30.0))
    base = (pl.col("mago") >= 5) & (pl.col("mago") <= 12)
    ac = a.group_by("customer_id").agg(
        jr_flow1=pl.col("net").filter(pl.col("mago") == 1).mean() - pl.col("net").filter(base).mean(),
        jr_flow3=pl.col("net").filter(pl.col("mago") <= 3).mean() - pl.col("net").filter(base).mean(),
        jr_low1=pl.col("low").filter(pl.col("mago") == 1).mean() - pl.col("low").filter(base).mean())
    it = px.items(c.features().select("application_id", "customer_id", "decision_datetime")).with_columns(age=days("cutoff", "transaction_datetime"))
    pr = it.group_by("customer_id").agg(jr_price=pl.col("z").filter(pl.col("age") < 90).mean() - pl.col("z").filter(pl.col("age") > 182).mean())
    b = (c.load("bureau_observations").join(apps.select("customer_id", sel=pl.col("application_id")), on="customer_id")
         .with_columns(is_sel=pl.col("application_id") == pl.col("sel")))
    bu = b.group_by("customer_id").agg(
        jr_bdel=pl.col("delinquent_accounts_last_12m").filter("is_sel").mean() - pl.col("delinquent_accounts_last_12m").filter(~pl.col("is_sel")).mean(),
        jr_bscore=-(pl.col("score").filter("is_sel").mean() - pl.col("score").filter(~pl.col("is_sel")).mean()))
    d = apps.select("application_id", "customer_id")
    for part in (ec, ac, pr, bu):
        d = d.join(part, on="customer_id", how="left", maintain_order="left")
    d = d.with_columns(pl.col("jr_flow1", "jr_flow3") * -1)  # worse = higher
    sig = ["jr_surge30", "jr_flow3", "jr_low1", "jr_price", "jr_bdel", "jr_bscore"]
    z = {k: (pl.col(k) - pl.col(k).mean()) / pl.col(k).std() for k in sig}
    d = d.with_columns(**{f"{k}_z": e for k, e in z.items()})
    zc = [f"{k}_z" for k in sig]
    d = d.with_columns(jr_index=pl.mean_horizontal(zc),
                       jr_x_surge_flow=pl.col("jr_surge30_z") * pl.col("jr_flow3_z"),
                       jr_x_surge_price=pl.col("jr_surge30_z") * pl.col("jr_price_z"),
                       jr_x_flow_price=pl.col("jr_flow3_z") * pl.col("jr_price_z")).drop(zc + ["customer_id"])
    d = d.with_columns([pl.col(k).cast(pl.Float64).fill_nan(None) for k in d.columns if k != "application_id"])
    d.write_parquet(c.PQ / "f287_joint_recent.parquet")
    t = c.features().select("application_id", "target", "is_train", "has_history").join(d, on="application_id", maintain_order="left").filter(pl.col("is_train") == 1)
    for g in (0, 1):
        q = t.filter(pl.col("has_history") == g)
        y = q["target"].to_numpy()
        print(f"has_history={g}: " + "  ".join(f"{k[3:]} {c.ks(y[np.isfinite(q[k].to_numpy())], q[k].to_numpy()[np.isfinite(q[k].to_numpy())]):.3f}"
                                                for k in d.columns if k != "application_id" and np.isfinite(q[k].to_numpy()).sum() > 500))


if __name__ == "__main__":
    main()
