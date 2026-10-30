"""Second stage on top of the 8-seed trees+Markov blend: LightGBM boosting from logit(calibrated OOF) with only the
f277 auxiliary repayment-latent columns (+ has_history). 270-style residual test, repeated over several fold seeds.
Controls: the same stage with f271 columns (teammate spec) or with all current features gave no consistent gain.
Usage: cd eda && ../.venv/bin/python 278_stack_aux.py block[,block] seed[,seed...] [-]  ("-" = also write test preds)
"""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import common as c
import valid2 as v

ea = importlib.import_module("270_error_analysis")
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=200, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, n_jobs=4)


def block(names, ids):
    d = pl.DataFrame({"application_id": ids})
    for b in names:
        d = d.join(pl.read_parquet(c.PQ / f"{b}.parquet"), on="application_id", how="left", maintain_order="left")
    return d.drop("application_id").to_pandas()


def stage(B, y, base, seed):
    deltas, trees = [], []
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(B, y)):
        fi, st = next(StratifiedKFold(7, shuffle=True, random_state=1000 + k).split(tr, y[tr]))
        a, b = tr[fi], tr[st]
        m = lgb.train(PARAMS, lgb.Dataset(B.iloc[a], y[a], init_score=base[a]), 3000,
                      valid_sets=[lgb.Dataset(B.iloc[b], y[b], init_score=base[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
        o = base[va] + m.predict(B.iloc[va], raw_score=True)
        deltas.append(v.ks(y[va], o) - v.ks(y[va], base[va]))
        trees.append(m.best_iteration)
    return deltas, trees


def main():
    names = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")]
    feats = pl.read_parquet(c.PQ / "features.parquet").join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, _, y, ids, _ = v.frames(feats)
    B = block(names, ids)
    B["has_history"] = X["has_history"].to_numpy()
    s = ea.oof8(ids)
    p = ea.calibrate(s, y)
    base = np.log(p / (1 - p))
    alld = []
    for sd in seeds:
        d, t = stage(B, y, base, sd)
        alld += d
        print(f"{'+'.join(names)} seed {sd}: per-fold delta {np.round(d, 5)} mean {np.mean(d):+.5f} won {sum(x > 0 for x in d)}/5 trees {t}", flush=True)
    alld = np.array(alld)
    print(f"OVERALL {'+'.join(names)}: mean per-fold delta {alld.mean():+.5f} (se {alld.std(ddof=1) / np.sqrt(len(alld)):.5f}), won {(alld > 0).sum()}/{len(alld)}")


if __name__ == "__main__":
    main()
