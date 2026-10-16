"""v21: LightGBM feature ablation: + Astra mature-history features (A3), redundancy pruning (A6). Kaggle CPU only."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp

LGB = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, metric="auc")
SEEDS = (0, 1)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(np.arange(len(y)), valid), valid


def prune(X, y, num_cols):
    """Drop numeric columns whose rank correlation with a stronger (higher training KS) column is >= 0.95."""
    sub = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)
    score = {}
    for c in num_cols:
        v = X[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        score[c] = ks(y[m], v[m]) if m.sum() > 100 and len(np.unique(y[m])) == 2 and np.nanstd(v[m]) > 0 else 0
    R = X.iloc[sub][num_cols].rank()
    keep, kept = [], []
    for c in sorted(num_cols, key=lambda c: -score[c]):
        r = R[c].fillna(R[c].median()).to_numpy()
        if all(abs(np.corrcoef(r, q)[0, 1]) < 0.95 for q in kept if np.std(r) > 0) or np.std(r) == 0:
            keep.append(c)
            kept.append(r)
    return keep


def main_v21(build_all):
    assert Path("/kaggle/input").exists()
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    mature = pl.read_parquet(next(Path("/kaggle/input").rglob("mature_history.parquet"))).drop("customer_id")
    out = Path("/kaggle/working/v21")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id").join(mature, on="application_id", how="left", validate="1:1")
    train, test = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    mcols = [c for c in mature.columns if c != "application_id"]
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    X, Xt = train.select(cols).to_pandas(), test.select(cols).to_pandas()
    for c in cols:
        if features[c].dtype == pl.String:
            cats = sorted(X[c].dropna().unique())
            X[c], Xt[c] = pd.Categorical(X[c], categories=cats), pd.Categorical(Xt[c], categories=cats)
        else:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan).astype(float)
            Xt[c] = Xt[c].replace([np.inf, -np.inf], np.nan).astype(float)
    hand = [c for c in cols if c not in mcols]
    nums = [c for c in hand if features[c].dtype != pl.String]
    y = train["target"].to_numpy().astype(int)
    variants = ["base", "mature", "pruned"]
    oof = {v: np.zeros(len(y)) for v in variants}
    tst = {v: np.zeros(test.height) for v in variants}
    log = []
    for fold, (tr, va) in enumerate(splits(y, 5, 42)):
        fit, stop = next(splits(y[tr], 7, 300 + fold))
        kept = prune(X.iloc[tr], y[tr], nums)
        feats = {"base": hand, "mature": hand + mcols, "pruned": kept + [c for c in hand if c not in nums]}
        for v in variants:
            for s in SEEDS:
                F = feats[v]
                dtr = lgb.Dataset(X.iloc[tr[fit]][F], y[tr[fit]])
                dst = lgb.Dataset(X.iloc[tr[stop]][F], y[tr[stop]], reference=dtr)
                m = lgb.train(dict(LGB, seed=s), dtr, 5000, valid_sets=[dst], callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[v][va] += m.predict(X.iloc[va][F], num_iteration=m.best_iteration) / len(SEEDS)
                tst[v] += m.predict(Xt[F], num_iteration=m.best_iteration) / len(SEEDS) / 5
            log.append(dict(fold=fold, variant=v, n_feat=len(feats[v]), ks=ks(y[va], oof[v][va])))
            print(log[-1], flush=True)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    res = {v: ks(y, oof[v]) for v in variants}
    print(res)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pl.DataFrame({"application_id": train["application_id"], "target": y, **oof}).write_parquet(out / "oof.parquet")
    pl.DataFrame({"application_id": test["application_id"], **tst}).write_parquet(out / "test.parquet")
