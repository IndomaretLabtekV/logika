"""Distil an ICL teacher into LightGBM (validation_v2-clean; --folds 0 = screen, default all 5 folds, seed 42).
Teacher soft labels for the outer training rows are nested: Causilo fitted on 4/5 of the training fold predicts the
other 1/5 (5 inner folds), so no outer-validation label reaches the student through a teacher context. Student =
LGB (valid2.LGB params, objective cross_entropy) on target a*y + (1-a)*p_teacher; budget from the inner stop split
(early stopping on the hard labels of the stop rows, AUC), refit on the whole training fold. Outer fold scored once.
Usage: cd eda && ../.venv/bin/python 118_distill.py [--alphas 1,0.7,0.5,0.3]"""
import argparse
import math

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--alphas", default="1,0.7,0.5,0.3")
ap.add_argument("--folds", default="0,1,2,3,4")
a = ap.parse_args()

X, Xt, y, ids, tids = v.frames(features())
alphas = [float(s) for s in a.alphas.split(",")]
oof = {al: np.full(len(y), np.nan) for al in alphas}
for k in [int(s) for s in a.folds.split(",")]:
    tr, va = v.outer(y, 42)[k]
    fi, st = v.inner(y[tr], 42, k)
    cache = OUT / f"118_teacher_f{k}.parquet"
    if cache.exists():
        teach = pl.read_parquet(cache)["p"].to_numpy()
    else:
        teach = np.full(len(y), np.nan)
        fit = v.causilo_fit()
        for j, (a_, b_) in enumerate(StratifiedKFold(5, shuffle=True, random_state=99 + k).split(tr, y[tr])):
            (p,), _ = fit(X, y, tr[a_], None, None, [X.iloc[tr[b_]]], 42)
            teach[tr[b_]] = p
        pl.DataFrame({"p": teach}).write_parquet(cache)
    print(f"fold {k}: teacher nested OOF on training fold KS {v.ks(y[tr], teach[tr]):.5f}", flush=True)
    for alpha in alphas:
        t = alpha * y + (1 - alpha) * np.nan_to_num(teach)
        p = dict(v.LGB, objective="cross_entropy", seed=42)
        m = lgb.train(p, lgb.Dataset(X.iloc[tr[fi]], t[tr[fi]]), v.N_ROUNDS,
                      valid_sets=[lgb.Dataset(X.iloc[tr[st]], y[tr[st]])],
                      callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        oof[alpha][va] = lgb.train(p, lgb.Dataset(X.iloc[tr], t[tr]), n).predict(X.iloc[va])
        print(f"  alpha {alpha:g}: rounds {n}, outer fold KS {v.ks(y[va], oof[alpha][va]):.5f}", flush=True)
tag = "" if a.folds == "0" else "_5f"
for alpha, o in oof.items():
    pl.DataFrame({"application_id": ids, "target": y, f"distill_a{alpha:g}": o}).write_parquet(
        OUT / f"118_oof_distill_a{alpha:g}{tag}.parquet")
