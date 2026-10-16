"""v19: pseudo-label self-training for LightGBM (fold-clean teacher = the fold's own base model). Kaggle CPU only."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp

LGB = dict(learning_rate=0.02, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, metric="auc")
# name -> (weight of pseudo rows, which pseudo rows: "all" soft labels or "conf" hard labels on confident tails)
VARIANTS = {"pl_soft1": (1.0, "all"), "pl_soft03": (0.3, "all"), "pl_conf": (1.0, "conf")}


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(np.arange(len(y)), valid), valid


def train(X, y, w, Xs, ys, objective):
    dtr = lgb.Dataset(X, y, weight=w)
    dst = lgb.Dataset(Xs, ys, reference=dtr)
    m = lgb.train(dict(LGB, objective=objective), dtr, 5000, valid_sets=[dst],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def main_v19(build_all):
    assert Path("/kaggle/input").exists()
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    out = Path("/kaggle/working/v19")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id")
    train_df, test_df = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    X, Xt = train_df.select(cols).to_pandas(), test_df.select(cols).to_pandas()
    for c in cols:
        if features[c].dtype == pl.String:
            cats = sorted(X[c].dropna().unique())
            X[c], Xt[c] = pd.Categorical(X[c], categories=cats), pd.Categorical(Xt[c], categories=cats)
        else:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            Xt[c] = Xt[c].replace([np.inf, -np.inf], np.nan)
    y = train_df["target"].to_numpy().astype(int)
    allv = ["base", *VARIANTS]
    oof = {v: np.full(len(y), np.nan) for v in allv}
    tst = {v: np.zeros(test_df.height) for v in allv}
    log = []
    for fold, (tr, va) in enumerate(splits(y, 5, 42)):
        fit, stop = next(splits(y[tr], 7, 300 + fold))
        Xf, yf, Xs, ys = X.iloc[tr[fit]], y[tr[fit]], X.iloc[tr[stop]], y[tr[stop]]
        t0 = time.monotonic()
        m = train(Xf, yf, None, Xs, ys, "binary")
        teach = m.predict(Xt, num_iteration=m.best_iteration)  # sees only tr[fit]: no valid-fold labels
        preds = {"base": m}
        for name, (w, mode) in VARIANTS.items():
            if mode == "all":
                Xp, yp = Xt, teach
            else:
                lo_, hi_ = np.quantile(teach, [0.3, 0.97])
                keep = (teach <= lo_) | (teach >= hi_)
                Xp, yp = Xt[keep], (teach[keep] >= hi_).astype(float)
            XA = pd.concat([Xf, Xp], ignore_index=True)
            yA = np.concatenate([yf.astype(float), yp])
            wA = np.concatenate([np.ones(len(yf)), np.full(len(yp), w)])
            preds[name] = train(XA, yA, wA, Xs, ys.astype(float), "cross_entropy")
        for name, mdl in preds.items():
            pv = mdl.predict(X.iloc[va], num_iteration=mdl.best_iteration)
            oof[name][va] = pv
            tst[name] += mdl.predict(Xt, num_iteration=mdl.best_iteration) / 5
            log.append(dict(fold=fold, variant=name, iters=mdl.best_iteration, ks=ks(y[va], pv)))
            print(log[-1], flush=True)
        print(f"fold {fold} {time.monotonic() - t0:.0f}s", flush=True)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    res = {n: ks(y, oof[n]) for n in allv}
    print(res)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pl.DataFrame({"application_id": train_df["application_id"], "target": y, **oof}).write_parquet(out / "oof.parquet")
    pl.DataFrame({"application_id": test_df["application_id"], **tst}).write_parquet(out / "test.parquet")
