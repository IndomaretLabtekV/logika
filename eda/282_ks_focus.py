"""Bet 11: KS only scores the ordering around one cut (top ~31% of scores on our OOF, plateau 28-36%).
Logloss spreads capacity over the whole ranking; here training rows near the cut get more weight.

Inside every outer fold a quick 4-fold inner LightGBM ranks the training rows (training labels only); a row at inner
rank u gets weight 1 + a * exp(-((u - c) / s)^2 / 2) with c = 0.687 (the cut). Then the v22 LGB is fitted as in
valid2 with those weights. Compared fold by fold with a plain v22 LGB on the same seed (run here if missing).
Usage: cd eda && ../.venv/bin/python 282_ks_focus.py a s [seed]   (a=0 runs the plain reference)
"""
import math
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v73/ks_focus"
CUT = 0.687
QUICK = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=3)


def focus_fit(params, a, s):
    def fit(X, y, tr, fi, st, rest, seed):
        Xtr, ytr = X.iloc[tr], y[tr]
        wt = np.ones(len(tr))
        if a > 0:
            r = np.zeros(len(tr))
            for i, j in StratifiedKFold(4, shuffle=True, random_state=seed).split(Xtr, ytr):
                r[j] = lgb.train(dict(QUICK, seed=seed), lgb.Dataset(Xtr.iloc[i], ytr[i]), 400).predict(Xtr.iloc[j])
            u = rankdata(r) / len(r)
            wt = 1 + a * np.exp(-((u - CUT) / s) ** 2 / 2)
        pos = {j: i for i, j in enumerate(tr)}
        wf = wt[[pos[j] for j in tr[fi]]]
        p = dict(params, seed=seed)
        d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]], weight=wf)
        d_st = lgb.Dataset(X.iloc[tr[st]], y[tr[st]], reference=d_fit)
        m = lgb.train(p, d_fit, v.N_ROUNDS, valid_sets=[d_st], callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=wt), n)
        return [m.predict(x) for x in rest], n
    return fit


def main():
    a, s = float(sys.argv[1]), float(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    ref_path = OUT / f"oof_ref_s{seed}.parquet"
    if a == 0 or not ref_path.exists():
        v.run(f"ref_s{seed}", focus_fit(dict(v.LGB, n_jobs=3), 0, 1), X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
        if a == 0:
            return
    rv = pl.read_parquet(ref_path)[f"ref_s{seed}"].to_numpy()
    name = f"focus_a{a}_s{s}_s{seed}"
    oof, _, _ = v.run(name, focus_fit(dict(v.LGB, n_jobs=3), a, s), X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
    d = [v.ks(y[va], oof[va]) - v.ks(y[va], rv[va]) for _, va in v.outer(y, seed)]
    print(f"RESULT {name}: KS {v.ks(y, oof):.5f} vs ref {v.ks(y, rv):.5f} delta {v.ks(y, oof) - v.ks(y, rv):+.5f} "
          f"won {sum(x > 0 for x in d)}/5 per fold {np.round(d, 4)}", flush=True)


if __name__ == "__main__":
    main()
