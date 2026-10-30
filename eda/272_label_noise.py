"""Bet 9: train labels look noisier than test labels; down-weight suspicious training labels.

Evidence: test is a random split (adversarial AUC 0.501, same months), yet public KS sits ~0.04 above OOF for every
model family (trees +0.039, Kumo +0.035). Feature-independent label flips scale KS by (1 - a - b), where a, b are the
shares of wrong labels among observed bads and goods; 0.526 / 0.568 = 0.926 fits ~1-1.5% of goods flipped to bad.
Under such noise the OOF KS on observed labels stays proportional to the clean KS, so OOF can judge the fix.

Inside every outer fold: a quick 4-fold inner LightGBM ranks the training rows (training labels only); bads ranked
below quantile qb and goods above qg get weight w. Early stopping and refit as valid2 (single v22 LGB, seed 42),
compared with results/v71/block_on_markov/oof_lgb_ref_s42 (same features: features + f251).
Usage: cd eda && ../.venv/bin/python 272_label_noise.py qb qg w [seed]
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

OUT = ROOT / "results/v73/label_noise"
QUICK = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=8)


def noisy_fit(params, qb, qg, w):
    def fit(X, y, tr, fi, st, rest, seed):
        Xtr, ytr = X.iloc[tr], y[tr]
        r = np.zeros(len(tr))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=seed).split(Xtr, ytr):
            r[b] = lgb.train(dict(QUICK, seed=seed), lgb.Dataset(Xtr.iloc[a], ytr[a]), 400).predict(Xtr.iloc[b])
        u = rankdata(r) / len(r)
        wt = np.where(((ytr == 1) & (u < qb)) | ((ytr == 0) & (u > qg)), w, 1.0)
        fit.flagged.append(((ytr == 1) & (u < qb)).sum() + ((ytr == 0) & (u > qg)).sum())
        pos = {j: i for i, j in enumerate(tr)}
        wf = wt[[pos[j] for j in tr[fi]]]
        p = dict(params, seed=seed)
        d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]], weight=wf)
        d_st = lgb.Dataset(X.iloc[tr[st]], y[tr[st]], reference=d_fit)
        m = lgb.train(p, d_fit, v.N_ROUNDS, valid_sets=[d_st], callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=wt), n)
        return [m.predict(x) for x in rest], n
    fit.flagged = []
    return fit


def main():
    qb, qg, w = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    name = f"lgb_noise_b{qb}_g{qg}_w{w}_s{seed}"
    fit = noisy_fit(dict(v.LGB, n_jobs=8), qb, qg, w)
    oof, _, log = v.run(name, fit, X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
    ref = pl.read_parquet(ROOT / f"results/v71/block_on_markov/oof_lgb_ref_s{seed}.parquet")
    assert (ref["application_id"] == ids).all()
    rv = ref[ref.columns[-1]].to_numpy()
    print(f"{name}: flagged per fold {fit.flagged}")
    d = []
    for k, (_, va) in enumerate(v.outer(y, seed)):
        d.append(v.ks(y[va], oof[va]) - v.ks(y[va], rv[va]))
    print(f"pooled: ref {v.ks(y, rv):.5f} new {v.ks(y, oof):.5f} delta {v.ks(y, oof) - v.ks(y, rv):+.5f}; per fold {np.round(d, 5)}")


if __name__ == "__main__":
    main()
