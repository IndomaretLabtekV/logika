"""Bet 5 (from 265): observed minus expected lateness per customer.

265 showed the generator makes an installment late more often later in the loan (idx 1: 0.20, idx 5+: 0.37), with
amount_due and loan order, on top of the customer's own risk; and that this risk drifts (two loans of one customer
correlate 0.47 within 90 days, 0.26 at 540-730 days). Raw counts (and f251) mix the customer signal with the loan mix.

A label-free population LightGBM gives e = P(late | prev state, idx, log amount_due, loan order, due month) with no
customer information. Per customer, from rows before the decision (all repayment rows are, by construction):
  oe_z          sum(late - e) / sqrt(sum e(1-e))
  oe_theta      one Newton step for a log-odds offset with a N(0, 1/4) prior: sum(late - e) / (sum e(1-e) + 4)
  oe_theta_tXXX the same with weights exp(-days before decision / XXX)
  oe_entry_*    only rows whose previous installment was on time; oe_persist_* only rows after a late one
Censored rows (the last installment of a loan whose length is not 6, 12 or 18) are left out.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

import common as c

EDGES = [0, 29, 59, 89]
PRIOR = 4.0
TAUS = [180, 365, 730]


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime")
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > e).cast(pl.Int8) for e in EDGES]),
                       idx=pl.col("due_date").rank("ordinal").over("loan_id").cast(pl.Int32),
                       nidx=pl.len().over("loan_id"),
                       start=pl.col("due_date").min().over("loan_id"))
         .with_columns(prev=pl.col("s").shift(1).over("loan_id").fill_null(-1),
                       order=pl.col("start").rank("dense").over("customer_id").cast(pl.Int32),
                       late=(pl.col("s") > 0).cast(pl.Int8))
         .filter(~((pl.col("idx") == pl.col("nidx")) & ~pl.col("nidx").is_in([6, 12, 18]))))
    X = r.select(prev=pl.col("prev"), idx=pl.col("idx"), lamt=pl.col("amount_due").cast(pl.Float64).log1p(),
                 order=pl.col("order"), month=pl.col("due_date").dt.month()).to_pandas()
    y = r["late"].to_numpy()
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=500, verbose=-1),
                  lgb.Dataset(X, y, categorical_feature=["prev"]), 300)
    e = m.predict(X)
    print(f"{len(y)} rows, late rate {y.mean():.4f}, population model logloss "
          f"{-np.mean(y * np.log(e) + (1 - y) * np.log(1 - e)):.4f} vs constant "
          f"{-np.mean(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean())):.4f}")

    r = (r.with_columns(e=pl.Series(e), late_f=pl.col("late").cast(pl.Float64))
         .with_columns(res=pl.col("late_f") - pl.col("e"), info=pl.col("e") * (1 - pl.col("e")))
         .join(apps.select("customer_id", "decision_datetime"), on="customer_id")
         .with_columns(age=(pl.col("decision_datetime") - pl.col("due_date")).dt.total_days().cast(pl.Float64)))

    def stats(tag, flt):
        out = [(pl.col("res").filter(flt).sum() / pl.col("info").filter(flt).sum().sqrt()).alias(f"oe_{tag}z"),
               (pl.col("res").filter(flt).sum() / (pl.col("info").filter(flt).sum() + PRIOR)).alias(f"oe_{tag}theta")]
        for t in TAUS:
            w = (-pl.col("age") / t).exp()
            out.append(((w * pl.col("res")).filter(flt).sum() / ((w * pl.col("info")).filter(flt).sum() + PRIOR))
                       .alias(f"oe_{tag}theta_t{t}"))
        return out

    agg = r.group_by("customer_id").agg(
        *stats("", pl.lit(True)), *stats("entry_", pl.col("prev") == 0), *stats("persist_", pl.col("prev") > 0),
        oe_info=pl.col("info").sum())
    out = apps.join(agg, on="customer_id", how="left").drop("customer_id", "decision_datetime")
    out = out.with_columns([pl.col(k).cast(pl.Float64).fill_nan(None) for k in out.columns if k != "application_id"])
    out.write_parquet(c.PQ / "f266_expected_late.parquet")

    f = c.features().filter(pl.col("is_train") == 1).select("application_id", "target").join(
        out, on="application_id", how="left", maintain_order="left").join(
        pl.read_parquet(c.PQ / "f251_repayment_markov.parquet").select("application_id", "mk_entry", "mk_hit3"),
        on="application_id", how="left", maintain_order="left")
    yy = f["target"].to_numpy()
    has = f["oe_theta"].is_not_null().to_numpy()
    print(f"train apps with history {has.sum()}")
    for k in [x for x in f.columns if x.startswith(("oe_", "mk_"))]:
        print(f"   KS {k:22s} {c.ks(yy[has], np.nan_to_num(f[k].to_numpy()[has], nan=-9)):.4f}")


if __name__ == "__main__":
    main()
