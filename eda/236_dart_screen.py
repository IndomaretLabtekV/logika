"""LightGBM DART (AmEx 1st-place parameters: feature_fraction 0.05, 64 leaves, min_data 256, l2 30,
max_bin 63, lr 0.035) vs the v22 LGB recipe on the main 232 features, validation_v2 folds (seed 42).
DART early stopping is unreliable, so the round count is picked on the inner 1/7 stop split from
checkpoints every 250 rounds (max 4500), then refit x1.15 on the outer-train fold.
Usage: cd eda && ../.venv/bin/python 236_dart_screen.py <folds e.g. 0 or 0,1,2,3,4>
"""
import math
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT

DART = dict(objective="binary", metric="auc", boosting="dart", num_leaves=64, learning_rate=0.035,
            bagging_freq=5, bagging_fraction=0.75, feature_fraction=0.05, min_data_in_leaf=256, max_bin=63,
            min_data_in_bin=256, lambda_l1=0.1, lambda_l2=30, verbose=-1, n_jobs=8)
OUT = ROOT / "results/v66/dart"


def dart_fit(X, y, tr, fi, st, rest, seed):
    p = dict(DART, seed=seed)
    d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]], params={"max_bin": 63})
    m = lgb.train(p, d_fit, 4500)
    best, best_ks = 250, -1
    for n in range(250, 4501, 250):  # DART: predict with the first n trees (no dropout at inference)
        k = v.ks(y[tr[st]], m.predict(X.iloc[tr[st]], num_iteration=n))
        if k > best_ks:
            best, best_ks = n, k
    n = math.ceil(best * v.REFIT)
    m = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr], params={"max_bin": 63}), n)
    return [m.predict(r) for r in rest], n


if __name__ == "__main__":
    folds = [int(f) for f in sys.argv[1].split(",")]
    X, Xt, y, ids, test_ids = v.frames(pl.read_parquet(ROOT / "data_pq/features.parquet"))
    print("features", X.shape, flush=True)
    t = time.time()
    v.run("lgb_base", v.lgb_fit(dict(v.LGB, n_jobs=8)), X, Xt, y, ids, test_ids, OUT, seeds=[42], folds=folds, test=False)
    v.run("lgb_dart", dart_fit, X, Xt, y, ids, test_ids, OUT, seeds=[42], folds=folds, test=False)
    print("elapsed", round(time.time() - t), flush=True)
