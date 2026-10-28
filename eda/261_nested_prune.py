"""Bet 6: drop weak features inside each outer fold (no outer label used). v22 LGB on main + f251.
Per fold: fit on the inner fit split with early stopping, rank features by total gain, keep the top share KEEP,
then the usual validation_v2 early-stop + refit on the kept features only.
Usage: cd eda && ../.venv/bin/python 261_nested_prune.py 0.7 [seed]
"""
import math
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v71/block_on_markov"


def pruned_fit(keep):
    def fit(X, y, tr, fi, st, rest, seed):
        p = dict(v.LGB, n_jobs=8, seed=seed)
        d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]])
        m = lgb.train(p, d_fit, v.N_ROUNDS, valid_sets=[lgb.Dataset(X.iloc[tr[st]], y[tr[st]], reference=d_fit)],
                      callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        g = m.feature_importance("gain", iteration=m.best_iteration)
        cols = [X.columns[i] for i in np.argsort(-g)[: math.ceil(keep * X.shape[1])]]
        Xk = X[cols]
        d_fit = lgb.Dataset(Xk.iloc[tr[fi]], y[tr[fi]])
        m = lgb.train(p, d_fit, v.N_ROUNDS, valid_sets=[lgb.Dataset(Xk.iloc[tr[st]], y[tr[st]], reference=d_fit)],
                      callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        m = lgb.train(p, lgb.Dataset(Xk.iloc[tr], y[tr]), n)
        return [m.predict(r[cols]) for r in rest], n
    return fit


def main():
    keep = float(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    ref = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(ref)
    rn = f"lgb_ref_s{seed}"
    if not (OUT / f"oof_{rn}.parquet").exists():
        v.run(rn, v.lgb_fit(dict(v.LGB, n_jobs=8)), X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
    tag = f"lgb_prune{int(keep * 100)}_s{seed}"
    v.run(tag, pruned_fit(keep), X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
    b = pl.read_parquet(OUT / f"oof_{rn}.parquet")[rn].to_numpy()
    c = pl.read_parquet(OUT / f"oof_{tag}.parquet")[tag].to_numpy()
    wins = sum(v.ks(y[va], c[va]) > v.ks(y[va], b[va]) for _, va in v.outer(y, seed))
    print(f"{tag}: ref {v.ks(y, b):.5f} new {v.ks(y, c):.5f} delta {v.ks(y, c) - v.ks(y, b):+.5f} folds won {wins}/5")


if __name__ == "__main__":
    main()
