"""Row-level distant supervision ("series OOF", AmEx/Home Credit 1st place): every history row of a
customer gets that customer's target; a LightGBM on label-free row features learns which individual
events signal default; out-of-fold row predictions are aggregated per customer into new features.

Fully nested per outer fold (validation_v2, seed 42): GroupKFold(5) by customer inside outer-train gives
row OOF for outer-train customers; a model on all outer-train rows scores outer-val and test rows.
Outer-val labels are never used. Rows are restricted to timestamps before the customer's decision.
Row weight 1/n_rows(customer). Output (atomic): results/v66/series_oof/<table>_f<k>.parquet with
per-application aggregates for all 100k applications.
Usage: cd eda && ../.venv/bin/python 237_series_oof.py <ecom|repay> <folds e.g. 0>
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import valid2 as v
from common import ROOT

OUT = ROOT / "results/v66/series_oof"
ROW = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.8,
           bagging_fraction=0.5, bagging_freq=1, lambda_l2=10, verbose=-1, n_jobs=4, seed=237)  # 4 threads: 3 hangs under 8-thread CPU load
ROUNDS = 400


def apps():
    cols = ["application_id", "customer_id", "decision_datetime"]
    tr = pl.read_csv(ROOT / "data/train_applications.csv", try_parse_dates=True).select(*cols, "target")
    te = pl.read_csv(ROOT / "data/test_applications.csv", try_parse_dates=True).select(cols).with_columns(pl.lit(None, pl.Int64).alias("target"))
    return pl.concat([tr, te])


def ecom_rows(a):
    e = pl.read_csv(ROOT / "data/ecommerce_transactions.csv", try_parse_dates=True)
    e = e.join(a.select("customer_id", "decision_datetime"), on="customer_id").filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
    e = e.with_columns(unit=(pl.col("item_total") / pl.col("quantity").clip(1)),
                       month=pl.col("transaction_datetime").dt.strftime("%Y-%m"))
    # label-free market reference: median unit price per category x month over all customers (covariates only)
    e = e.with_columns(rel=(pl.col("unit").log() - pl.col("unit").log().median().over("product_category", "month")),
                       items=pl.len().over("order_id"),
                       share=pl.col("item_total") / pl.col("order_total").clip(1),
                       age_d=(pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                       hour=pl.col("transaction_datetime").dt.hour(), wday=pl.col("transaction_datetime").dt.weekday(),
                       cat=pl.col("product_category").cast(pl.Categorical).to_physical(),
                       pay=pl.col("payment_method").cast(pl.Categorical).to_physical(),
                       cancel=(pl.col("order_status") == "cancelled").cast(pl.Int8))
    feats = ["quantity", "item_total", "unit", "rel", "order_total", "items", "share", "age_d", "hour", "wday", "cat", "pay", "cancel"]
    return e.select("customer_id", "age_d", *[f for f in feats if f != "age_d"]), feats, ["cat", "pay"]


def repay_rows(a):
    r = pl.read_csv(ROOT / "data/repayment_history.csv", try_parse_dates=True)
    r = r.join(a.select("customer_id", "decision_datetime"), on="customer_id").filter(pl.col("due_date") < pl.col("decision_datetime"))
    r = r.sort("customer_id", "loan_id", "due_date").with_columns(
        inst_no=pl.int_range(pl.len()).over("loan_id"), n_inst=pl.len().over("loan_id"),
        paid_ratio=pl.col("amount_paid") / pl.col("amount_due").clip(1),
        delay_d=(pl.col("last_payment_datetime") - pl.col("due_date")).dt.total_seconds() / 86400,
        age_d=(pl.col("decision_datetime") - pl.col("due_date")).dt.total_seconds() / 86400,
        paid_after_cutoff=(pl.col("last_payment_datetime").is_null() | (pl.col("last_payment_datetime") >= pl.col("decision_datetime"))).cast(pl.Int8))
    # payments/DPD observed only up to the decision: mask post-cutoff payment info
    r = r.with_columns(delay_d=pl.when(pl.col("paid_after_cutoff") == 1).then(None).otherwise(pl.col("delay_d")),
                       prev_dpd=pl.col("max_days_past_due").shift(1).over("loan_id"))
    feats = ["amount_due", "paid_ratio", "days_past_due", "max_days_past_due", "delay_d", "age_d", "inst_no", "n_inst",
             "paid_after_cutoff", "prev_dpd"]
    return r.select("customer_id", *feats), feats, []


def aggregate(rows, pred, a):
    d = rows.select("customer_id", "age_d").with_columns(pl.Series("p", pred))
    g = d.group_by("customer_id").agg(
        pl.col("p").mean().alias("mean"), pl.col("p").max().alias("max"), pl.col("p").quantile(.9).alias("p90"),
        pl.col("p").std().alias("std"), pl.len().alias("n"),
        pl.col("p").filter(pl.col("age_d") <= 90).mean().alias("mean90"),
        ((pl.col("p") * (-pl.col("age_d") / 180).exp()).sum() / (-pl.col("age_d") / 180).exp().sum()).alias("rw"))
    return a.select("application_id", "customer_id").join(g, on="customer_id", how="left").drop("customer_id")


def main():
    table, folds = sys.argv[1], [int(f) for f in sys.argv[2].split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    a = apps()
    t = time.time()
    rows, feats, cats = {"ecom": ecom_rows, "repay": repay_rows}[table](a)
    print(table, "rows", rows.height, "prep", round(time.time() - t), "s", flush=True)
    train = a.filter(pl.col("target").is_not_null()).sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    # integer customer codes: string ops on 4.4M object rows were the bottleneck
    code = {c: i for i, c in enumerate(a["customer_id"].to_list())}
    rows = rows.with_columns(cid=pl.col("customer_id").replace_strict(code, return_dtype=pl.Int64))
    rc = rows["cid"].to_numpy()
    train_cid = np.array([code[c] for c in train["customer_id"].to_list()])
    yc = np.full(len(code), -1)
    yc[train_cid] = y
    X = rows.select(feats).to_numpy().astype(np.float32)
    w = (1.0 / np.bincount(rc, minlength=len(code))[rc]).astype(np.float32)
    yrow = yc[rc]
    for k in folds:
        path = OUT / f"{table}_f{k}.parquet"
        if path.exists():
            print("cached", path.name, flush=True)
            continue
        tr, _ = v.outer(y, 42)[k]
        mask = np.zeros(len(code), bool)
        mask[train_cid[tr]] = True
        in_tr = mask[rc]
        pred = np.full(len(rc), np.nan)
        idx = np.flatnonzero(in_tr)
        for j, (a_i, b_i) in enumerate(GroupKFold(5).split(idx, groups=rc[idx])):
            m = lgb.train(ROW, lgb.Dataset(X[idx[a_i]], yrow[idx[a_i]], weight=w[idx[a_i]], categorical_feature=[feats.index(c) for c in cats]), ROUNDS)
            pred[idx[b_i]] = m.predict(X[idx[b_i]])
            print(table, "fold", k, "inner", j, round(time.time() - t), "s", flush=True)
        m = lgb.train(ROW, lgb.Dataset(X[idx], yrow[idx], weight=w[idx], categorical_feature=[feats.index(c) for c in cats]), ROUNDS)
        rest = np.flatnonzero(~in_tr)  # outer-val and test customers: never in any row-model training set
        pred[rest] = m.predict(X[rest])
        assert np.isfinite(pred).all()
        agg = aggregate(rows, pred, a).rename({c: f"so_{table}_{c}" for c in ["mean", "max", "p90", "std", "n", "mean90", "rw"]})
        tmp = path.with_name(path.name + ".tmp")
        agg.write_parquet(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(path)
        print(table, "fold", k, "done", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
