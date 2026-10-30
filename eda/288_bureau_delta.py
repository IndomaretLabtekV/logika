"""Bet 16 (from 287: jr_bscore chi2 14.1, the strongest screen result since the Markov anchor's 17.8).

Every application (current and previous) has its own bureau pulls from 1-3 providers with identical score scales.
features.parquet has the level of the current pulls and a slope over all pulls (bz_trend), not the change between the
pulls made for this application and those made for earlier applications. Label-free, customers with earlier pulls only.
  bd_<col>_vs_mean   current-application mean minus the mean over earlier applications' pulls
  bd_<col>_vs_last   current-application mean minus the most recent earlier application's mean
  bd_<col>_pp        same-provider change vs the last earlier pull of that provider, averaged over providers
  bd_<col>_rate      bd_<col>_vs_last per 365 days between the two applications
  bd_gap_d           days between the current and the last earlier application's pulls
for col in score, utilization, inquiries_last_6m, delinquent_accounts_last_12m, active_accounts, outstanding_balance (log).
Usage: cd eda && ../.venv/bin/python 288_bureau_delta.py
"""
import numpy as np
import polars as pl

import common as c

COLS = {"score": "score", "util": "utilization", "inq": "inquiries_last_6m", "del": "delinquent_accounts_last_12m",
        "act": "active_accounts", "debt": "outstanding_balance"}


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime")
    b = (c.load("bureau_observations").join(apps.select("customer_id", sel=pl.col("application_id")), on="customer_id")
         .with_columns(is_sel=pl.col("application_id") == pl.col("sel"), debt_l=pl.col("outstanding_balance").cast(pl.Float64).log1p()))
    cols = {k: ("debt_l" if k == "debt" else v) for k, v in COLS.items()}
    # application-level means (one row per customer x application)
    al = b.group_by("customer_id", "application_id").agg(
        is_sel=pl.col("is_sel").first(), t=pl.col("report_datetime").mean(), **{k: pl.col(v).cast(pl.Float64).mean() for k, v in cols.items()})
    cur = al.filter("is_sel").drop("is_sel", "application_id")
    prev = al.filter(~pl.col("is_sel")).sort("customer_id", "t")
    pm = prev.group_by("customer_id").agg(**{f"{k}_pm": pl.col(k).mean() for k in cols}, **{f"{k}_pl": pl.col(k).last() for k in cols}, t_pl=pl.col("t").last())
    d = cur.join(pm, on="customer_id", how="inner").with_columns(bd_gap_d=(pl.col("t") - pl.col("t_pl")).dt.total_days().cast(pl.Float64))
    d = d.select("customer_id", "bd_gap_d",
                 *[(pl.col(k) - pl.col(f"{k}_pm")).alias(f"bd_{k}_vs_mean") for k in cols],
                 *[(pl.col(k) - pl.col(f"{k}_pl")).alias(f"bd_{k}_vs_last") for k in cols],
                 *[((pl.col(k) - pl.col(f"{k}_pl")) / (pl.col("bd_gap_d").clip(lower_bound=30) / 365)).alias(f"bd_{k}_rate") for k in cols])
    # same-provider change vs the provider's last earlier pull
    bp = b.sort("customer_id", "bureau_provider", "report_datetime")
    last_prev = bp.filter(~pl.col("is_sel")).group_by("customer_id", "bureau_provider").agg(**{f"{k}_lp": pl.col(v).cast(pl.Float64).drop_nulls().last() for k, v in cols.items()})
    cur_p = bp.filter("is_sel").group_by("customer_id", "bureau_provider").agg(**{k: pl.col(v).cast(pl.Float64).mean() for k, v in cols.items()})
    pp = cur_p.join(last_prev, on=["customer_id", "bureau_provider"]).group_by("customer_id").agg(
        **{f"bd_{k}_pp": (pl.col(k) - pl.col(f"{k}_lp")).mean() for k in cols})
    out = apps.select("application_id", "customer_id").join(d, on="customer_id", how="left", maintain_order="left").join(
        pp, on="customer_id", how="left", maintain_order="left").drop("customer_id")
    out = out.with_columns([pl.col(k).cast(pl.Float64).fill_nan(None) for k in out.columns if k != "application_id"])
    out.write_parquet(c.PQ / "f288_bureau_delta.parquet")
    t = c.features().select("application_id", "target", "is_train", "bz_trend").join(out, on="application_id", maintain_order="left").filter(pl.col("is_train") == 1)
    y = t["target"].to_numpy()
    print(f"rows with earlier pulls: {t['bd_score_vs_mean'].is_not_null().mean():.3f}")
    for k in ["bz_trend"] + [x for x in out.columns if x.startswith("bd_score") or x == "bd_gap_d"]:
        s = t[k].to_numpy()
        f = np.isfinite(s)
        print(f"   KS {k:18s} {c.ks(y[f], s[f]):.4f}")


if __name__ == "__main__":
    main()
