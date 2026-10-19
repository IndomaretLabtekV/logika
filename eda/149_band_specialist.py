"""Band specialist (nested, seed 42). Pooled KS is decided around the global cut (rank ~0.69, eda/127/148), where
every model has AUC ~0.53. Per outer fold: stage 1 = LGB; training-fold rows get nested 5-fold OOF stage-1 scores,
validation rows the full-training-fold stage-1 model. Band = stage-1 rank in [lo, hi] among the training fold's OOF
(the same cut points applied to validation scores). Stage 2 = LGB trained ONLY on training-fold band rows with the
stage-1 logit as an extra feature (inner stop split inside the band rows). Validation band rows are re-ordered by
w * rank(stage2) + (1 - w) * rank(stage1) while keeping the stage-1 score values (quantile mapping inside the band),
rows outside the band untouched. Compared paired with stage 1 alone (= plain LGB seed 42).
Usage: cd eda && ../.venv/bin/python 149_band_specialist.py [--lo 0.55 --hi 0.85]"""
import argparse
import math

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--lo", type=float, default=0.55)
ap.add_argument("--hi", type=float, default=0.85)
a = ap.parse_args()
X, Xt, y, ids, tids = v.frames(features())
P = dict(v.LGB, seed=42)
W = (0.3, 0.5, 1.0)
base = np.full(len(y), np.nan)
out = {w: np.full(len(y), np.nan) for w in W}


def fit_lgb(Xf, yf, rows_fit, rows_stop, n_max=v.N_ROUNDS):
    m = lgb.train(P, lgb.Dataset(Xf.iloc[rows_fit], yf[rows_fit]), n_max, valid_sets=[lgb.Dataset(Xf.iloc[rows_stop], yf[rows_stop])],
                  callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
    return m.best_iteration


for k, (tr, va) in enumerate(v.outer(y, 42)):
    fi, st = v.inner(y[tr], 42, k)
    n1 = math.ceil(fit_lgb(X, y, tr[fi], tr[st]) * v.REFIT)
    s1_va = lgb.train(P, lgb.Dataset(X.iloc[tr], y[tr]), n1).predict(X.iloc[va], raw_score=True)
    s1_tr = np.zeros(len(tr))
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=900 + k).split(tr, y[tr]):
        s1_tr[b_] = lgb.train(P, lgb.Dataset(X.iloc[tr[a_]], y[tr[a_]]), n1).predict(X.iloc[tr[b_]], raw_score=True)
    lo_v, hi_v = np.quantile(s1_tr, [a.lo, a.hi])
    band_tr = (s1_tr >= lo_v) & (s1_tr <= hi_v)
    Xs = X.assign(s1=np.nan)
    Xs.iloc[tr, Xs.columns.get_loc("s1")] = s1_tr
    Xs.iloc[va, Xs.columns.get_loc("s1")] = s1_va
    rows = tr[band_tr]
    f2, s2 = v.inner(y[rows], 42, 50 + k)
    n2 = math.ceil(fit_lgb(Xs, y, rows[f2], rows[s2]) * v.REFIT)
    spec = lgb.train(P, lgb.Dataset(Xs.iloc[rows], y[rows]), n2)
    band_va = (s1_va >= lo_v) & (s1_va <= hi_v)
    s2_va = spec.predict(Xs.iloc[va[band_va]], raw_score=True)
    base[va] = s1_va
    for w in W:
        o = s1_va.copy()
        comb = w * rankdata(s2_va) + (1 - w) * rankdata(s1_va[band_va])
        o[band_va] = np.sort(s1_va[band_va])[np.argsort(np.argsort(comb))]
        out[w][va] = o
    print(f"fold {k}: band rows train {band_tr.sum()} (bad {y[rows].mean():.3f}), val {band_va.sum()}; stage-1 KS "
          f"{v.ks(y[va], s1_va):.4f}; band AUC-ish KS stage1 {v.ks(y[va][band_va], s1_va[band_va]):.4f} specialist "
          f"{v.ks(y[va][band_va], s2_va):.4f}; " + " ".join(f"w{w}: {v.ks(y[va], out[w][va]):.4f}" for w in W), flush=True)
folds = v.outer(y, 42)
for w in W:
    o = out[w]
    r = np.random.default_rng(0)
    d = np.array([(lambda i: v.ks(y[i], o[i]) - v.ks(y[i], base[i]))(r.choice(len(y), len(y))) for _ in range(500)])
    wins = sum(v.ks(y[va], o[va]) > v.ks(y[va], base[va]) for _, va in folds)
    print(f"w {w}: pooled {v.ks(y, o):.5f} vs stage1 {v.ks(y, base):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5")
pl.DataFrame({"application_id": ids, "target": y, "stage1": base, **{f"band_w{w}": out[w] for w in W}}).write_parquet(OUT / "149_band_specialist.parquet")
