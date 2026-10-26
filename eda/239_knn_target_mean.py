"""Home Credit 1st-place style neighbour target mean, fully nested per outer fold (seed 42).
Inside outer-train: pick the 5 strongest numeric main232 features by univariate KS (fitting labels only),
rank-transform with outer-train quantiles (NaN -> 0.5), then the mean target of the k nearest outer-train
neighbours (k=100, 500): 5-fold OOF for outer-train rows, all outer-train neighbours for outer-val/test.
Output (atomic) in 237's format: results/v66/series_oof/knn_f<k>.parquet, 100k applications.
Usage: cd eda && ../.venv/bin/python 239_knn_target_mean.py <folds e.g. 0>
"""
import os
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import valid2 as v
from common import ROOT

OUT = ROOT / "results/v66/series_oof"
KS = (100, 500)


def rank_by(ref, x):
    srt = np.sort(ref[np.isfinite(ref)])
    r = np.searchsorted(srt, x, side="right") / max(len(srt), 1)
    return np.where(np.isfinite(x), r, .5)


def main():
    folds = [int(f) for f in sys.argv[1].split(",")]
    X, Xt, y, ids, test_ids = v.frames(pl.read_parquet(ROOT / "data_pq/features.parquet"))
    num = [c for c in X.columns if c not in v.CAT_COLS]
    A = np.vstack([X[num].to_numpy(float), Xt[num].to_numpy(float)])
    all_ids = np.r_[ids.to_numpy(), test_ids.to_numpy()]
    for k in folds:
        path = OUT / f"knn_f{k}.parquet"
        if path.exists():
            print("cached", path.name)
            continue
        tr, _ = v.outer(y, 42)[k]
        score = [v.ks_nan(y[tr], X[c].to_numpy(float)[tr]) for c in num]
        top = [num[i] for i in np.argsort(score)[::-1][:5]]
        cols = [num.index(c) for c in top]
        Z = np.column_stack([rank_by(A[tr, j], A[:, j]) for j in cols])
        out = {kk: np.full(len(A), np.nan) for kk in KS}
        for a, b in StratifiedKFold(5, shuffle=True, random_state=239 + k).split(tr, y[tr]):
            nn = NearestNeighbors(n_neighbors=max(KS)).fit(Z[tr[a]])
            _, ind = nn.kneighbors(Z[tr[b]])
            for kk in KS:
                out[kk][tr[b]] = y[tr[a]][ind[:, :kk]].mean(1)
        rest = np.setdiff1d(np.arange(len(A)), tr)  # outer-val + test
        nn = NearestNeighbors(n_neighbors=max(KS)).fit(Z[tr])
        _, ind = nn.kneighbors(Z[rest])
        for kk in KS:
            out[kk][rest] = y[tr][ind[:, :kk]].mean(1)
        df = pl.DataFrame({"application_id": all_ids, **{f"knn_tm{kk}": out[kk] for kk in KS}})
        tmp = path.with_name(path.name + ".tmp")
        df.write_parquet(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(path)
        print("fold", k, "top features", top, flush=True)


if __name__ == "__main__":
    main()
