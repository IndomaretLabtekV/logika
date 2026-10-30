"""Bet 13: nearest-neighbour structure. A generator that draws customers around prototypes leaves label information
in neighbourhoods that per-feature splits cannot rebuild.

Space: the 40 most important numeric features of the v22 LGB (gain on features + f251), rank-gauss transformed over
train + test (label-free), missing -> 0 plus nothing else. Features:
  kn_tm{k}   mean target of the k nearest TRAIN neighbours (k = 20, 100); cross-fitted on the 5 outer folds of seed 42:
             a train row only sees neighbours from the other 4 folds; test rows see all train rows
  kn_d{k}    distance to the k-th neighbour among train + test (label-free density)
Usage: cd eda && ../.venv/bin/python 284_knn_target.py
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import erfinv
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

import common as c
import valid2 as v


def rank_gauss(x):
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        f = np.isfinite(col)
        r = (rankdata(col[f]) - 0.5) / f.sum()
        out[f, j] = np.sqrt(2) * erfinv(2 * r - 1)
    return out


def main():
    feats = pl.read_parquet(c.PQ / "features.parquet").join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    m = lgb.train(dict(v.LGB, n_jobs=8, seed=0), lgb.Dataset(X, y), 600)
    imp = m.feature_importance("gain")
    num = [k for k in X.columns if str(X[k].dtype) != "category"]
    top = [k for k in X.columns[np.argsort(-imp)] if k in num][:40]
    print("space:", ", ".join(top[:12]), "...")
    Z = rank_gauss(np.vstack([X[top].to_numpy(float), Xt[top].to_numpy(float)]))
    Ztr, Zte = Z[: len(y)], Z[len(y):]
    out_tr, out_te = {}, {}
    for k in (20, 100):
        tm = np.zeros(len(y))
        for tr, va in v.outer(y, 42):
            nn = NearestNeighbors(n_neighbors=k, n_jobs=8).fit(Ztr[tr])
            _, idx = nn.kneighbors(Ztr[va])
            tm[va] = y[tr][idx].mean(1)
        nn = NearestNeighbors(n_neighbors=k, n_jobs=8).fit(Ztr)
        _, idx = nn.kneighbors(Zte)
        out_tr[f"kn_tm{k}"], out_te[f"kn_tm{k}"] = tm, y[idx].mean(1)
        print(f"kn_tm{k}: train KS {v.ks(y, tm):.4f}", flush=True)
    nn = NearestNeighbors(n_neighbors=101, n_jobs=8).fit(Z)
    d, _ = nn.kneighbors(Z)
    for k in (10, 100):
        out_tr[f"kn_d{k}"], out_te[f"kn_d{k}"] = d[: len(y), k], d[len(y):, k]
        print(f"kn_d{k}: train KS {v.ks(y, out_tr[f'kn_d{k}']):.4f}")
    a = pl.DataFrame({"application_id": ids, **out_tr})
    b = pl.DataFrame({"application_id": test_ids, **out_te})
    pl.concat([a, b]).write_parquet(c.PQ / "f284_knn_target.parquet")


if __name__ == "__main__":
    main()
