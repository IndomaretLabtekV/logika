"""Hyperparameter screen for the trees+Markov feature set (features + f251): the v22 LGB parameters predate f251.
Single LightGBM, validation_v2 protocol (inner 1/7 early stop, refit x1.15), seed 42 folds; compared fold by fold with
results/v71/block_on_markov/oof_lgb_ref_s42 (v22 LGB on the same features). Configs are fixed in advance, not tuned on LB.
Usage: cd eda && ../.venv/bin/python 273_lgb_tune.py name[,name...] [seed] [n_jobs]
"""
import sys

import numpy as np
import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v73/lgb_tune"
CONFIGS = {
    "lr01": dict(learning_rate=0.01),
    "l15": dict(num_leaves=15, min_child_samples=200),
    "l63": dict(num_leaves=63, min_child_samples=300, lambda_l2=10.0),
    "ff03": dict(feature_fraction=0.3),
    "mcs400": dict(min_child_samples=400, lambda_l2=20.0),
    "slow_small": dict(learning_rate=0.01, num_leaves=15, feature_fraction=0.3, min_child_samples=300),
    "bin63": dict(max_bin=63),
    "goss_free": dict(bagging_fraction=0.6, feature_fraction=0.4, lambda_l1=1.0),
    "l7d3": dict(num_leaves=7, max_depth=3, min_child_samples=200, feature_fraction=0.3, lambda_l2=10.0, learning_rate=0.01),
}


def main():
    names = sys.argv[1].split(",")
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    ref = pl.read_parquet(ROOT / f"results/v71/block_on_markov/oof_lgb_ref_s{seed}.parquet")
    assert (ref["application_id"] == ids).all()
    rv = ref[ref.columns[-1]].to_numpy()
    for name in names:
        oof, _, log = v.run(f"lgb_{name}_s{seed}", v.lgb_fit(dict(v.LGB, **CONFIGS[name], n_jobs=jobs)), X, Xt, y, ids, test_ids,
                            OUT, seeds=[seed], test=False)
        d = [v.ks(y[va], oof[va]) - v.ks(y[va], rv[va]) for _, va in v.outer(y, seed)]
        print(f"RESULT {name:12s} KS {v.ks(y, oof):.5f} vs ref {v.ks(y, rv):.5f} delta {v.ks(y, oof) - v.ks(y, rv):+.5f} "
              f"folds won {sum(x > 0 for x in d)}/5 per fold {np.round(d, 4)} trees {[r['budget'] for r in log]}", flush=True)


if __name__ == "__main__":
    main()
