"""KS-targeted boosting (fold-0 screen, validation_v2-clean). LightGBM custom objective = logloss + lam * KS surrogate,
KS surrogate = -(mean_bad sigma((s - t)/tau) - mean_good sigma((s - t)/tau)) with t = the current q-quantile of the raw
scores (the pooled KS cut sits near rank 0.69, eda/127), so the extra gradient concentrates on rows near the cut.
Budget from the inner stop split (early stopping on stop-row KS), refit on the whole training fold, outer fold scored
once. lam = 0 reproduces plain LGB (check). Writes eda/out/128_oof_ks_lam*.parquet for screen2.
Usage: cd eda && ../.venv/bin/python 128_ks_objective.py [--lams 0,1,3] [--tau 0.5] [--q 0.69]"""
import argparse
import math

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import expit

import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--lams", default="0,1,3")
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--q", type=float, default=0.69)
ap.add_argument("--folds", default="0")
a = ap.parse_args()

X, Xt, y, ids, tids = v.frames(features())


def objective(lam):
    def f(score, data):
        yy = data.get_label()
        p = expit(score)
        g, h = p - yy, np.maximum(p * (1 - p), 1e-6)
        if lam:
            t = np.quantile(score, a.q)
            z = expit((score - t) / a.tau)
            dz = z * (1 - z) / a.tau
            nb, ng = yy.sum(), len(yy) - yy.sum()
            w = np.where(yy == 1, -1 / nb, 1 / ng) * len(yy)  # per-row scale comparable to logloss gradients
            g = g + lam * w * dz
            h = h + lam * np.abs(w) * dz * np.abs(1 - 2 * z) / a.tau
        return g, h
    return f


def ks_eval(score, data):
    return "ks", v.ks(data.get_label(), score), True


lams = [float(s) for s in a.lams.split(",")]
oof = {lam: np.full(len(y), np.nan) for lam in lams}
for k in [int(s) for s in a.folds.split(",")]:
    tr, va = v.outer(y, 42)[k]
    fi, st = v.inner(y[tr], 42, k)
    for lam in lams:
        p = dict(v.LGB, objective=objective(lam), metric="None", seed=42)
        d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]], init_score=np.full(len(fi), np.log(y[tr].mean() / (1 - y[tr].mean()))))
        d_st = lgb.Dataset(X.iloc[tr[st]], y[tr[st]], reference=d_fit)
        m = lgb.train(p, d_fit, v.N_ROUNDS, valid_sets=[d_st], feval=ks_eval,
                      callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        full = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr], init_score=np.full(len(tr), np.log(y[tr].mean() / (1 - y[tr].mean())))), n)
        oof[lam][va] = full.predict(X.iloc[va], raw_score=True)
        print(f"fold {k} lam {lam:g}: rounds {n}, outer KS {v.ks(y[va], oof[lam][va]):.5f}", flush=True)
tag = "" if a.folds == "0" else "_5f"
for lam, o in oof.items():
    pl.DataFrame({"application_id": ids, "target": y, f"ks_lam{lam:g}": o}).write_parquet(OUT / f"128_oof_ks_lam{lam:g}{tag}.parquet")
