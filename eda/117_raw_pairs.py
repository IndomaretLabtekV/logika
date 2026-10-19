"""Raw records of matched pairs: same v31 OOF score band (rank 0.70-0.75, near the KS cut), has_history = 1, one bad
and one good. Prints every table compactly so patterns the features miss can be spotted by eye.
Usage: cd eda && ../.venv/bin/python 117_raw_pairs.py [n_pairs]"""
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

from common import ROOT, features, load

R = ROOT / "results"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
f = features().filter(pl.col("is_train") == 1).sort("application_id")
ids = f["application_id"]


def oof(path, col):
    return pl.read_parquet(path).sort("application_id")[col].to_numpy()


trees = np.mean([rankdata(oof(R / f"v22/v22/gbdt/oof_{m}.parquet", m)) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
s = (rankdata(trees) + rankdata(oof(R / "v22/v22/fm/oof_causilo.parquet", "causilo"))
     + rankdata(oof(R / "v28m/oof_kumo_large_c56000.parquet", "kumo_large_c56000"))) / 3 / len(ids)
f = f.with_columns(score=pl.Series(s))
band = f.filter((pl.col("score") > 0.70) & (pl.col("score") < 0.75) & (pl.col("has_history") == 1))
rng = np.random.default_rng(1)
T = {t: load(t) for t in ["customer_profiles", "previous_applications", "repayment_history", "account_behavior",
                          "bureau_observations", "ecommerce_transactions", "train_applications"]}
def show(d):
    print(d.with_columns(pl.col(pl.Datetime).dt.strftime("%y-%m-%d %H:%M")).write_csv().strip())


for k in range(n):
    for tgt in (1, 0):
        row = band.filter(pl.col("target") == tgt).sample(1, seed=int(rng.integers(1e9)))
        cid, cut = row["customer_id"][0], row["decision_datetime"][0]
        print(f"\n{'=' * 30} pair {k} target {tgt} customer {cid} score {row['score'][0]:.3f} cutoff {cut}")
        show(T["train_applications"].filter(pl.col("customer_id") == cid).drop("customer_id", "application_id"))
        show(T["customer_profiles"].filter(pl.col("customer_id") == cid).drop("customer_id"))
        show(T["previous_applications"].filter(pl.col("customer_id") == cid).sort("application_datetime")
              .drop("customer_id", "application_id", "loan_id"))
        rp = (T["repayment_history"].filter(pl.col("customer_id") == cid).sort("due_date")
              .select(pl.col("loan_id").str.slice(-4), "due_date", "amount_due", "amount_paid", "last_payment_datetime",
                      "days_past_due", "max_days_past_due"))
        show(rp.tail(24))
        show(T["account_behavior"].filter(pl.col("customer_id") == cid).sort("month").drop("customer_id").tail(12))
        show(T["bureau_observations"].filter(pl.col("customer_id") == cid).sort("report_datetime")
              .drop("customer_id", "application_id", "bureau_observation_id").tail(9))
        ec = (T["ecommerce_transactions"].filter(pl.col("customer_id") == cid)
              .group_by("order_id").agg(pl.col("transaction_datetime", "order_total", "payment_method", "order_status").first(),
                                        n=pl.len(), cats=pl.col("product_category").str.slice(0, 4).str.join(","))
              .sort("transaction_datetime").drop("order_id"))
        print(f"orders {ec.height}; last 15:"); show(ec.tail(15))
