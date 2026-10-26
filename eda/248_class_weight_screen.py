"""Class weighting for KS (not in the null catalogue: v10 tuned capacity/LR only). v22 LGB on main232, 5-fold seed 42,
inner-stop budget, paired with lgb_base4 (same params/threads, results/v66/series_oof_eval). Variants:
scale_pos_weight 2 and 6.4 (~balanced), and cross_entropy on 0.02-smoothed labels.
Usage: cd eda && ../.venv/bin/python 248_class_weight_screen.py
"""
import math

import lightgbm as lgb
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT

OUT = ROOT / "results/v66/class_weight"


def smoothed_fit(params, eps=.02):
    def fit(X, y, tr, fi, st, rest, seed):
        p = dict(params, seed=seed, objective="cross_entropy")
        ys = y.astype(float) * (1 - 2 * eps) + eps
        m = lgb.train(p, lgb.Dataset(X.iloc[tr[fi]], ys[tr[fi]]), v.N_ROUNDS,
                      valid_sets=[lgb.Dataset(X.iloc[tr[st]], y[tr[st]].astype(float))],
                      callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        m = lgb.train(p, lgb.Dataset(X.iloc[tr], ys[tr]), n)
        return [m.predict(r) for r in rest], n
    return fit


X, Xt, y, ids, test_ids = v.frames(pl.read_parquet(ROOT / "data_pq/features.parquet"))
base = pl.read_parquet(ROOT / "results/v66/series_oof_eval/oof_lgb_base4.parquet")["lgb_base4"].to_numpy()
variants = {"spw2": v.lgb_fit(dict(v.LGB, n_jobs=4, scale_pos_weight=2.0)),
            "spw6": v.lgb_fit(dict(v.LGB, n_jobs=4, scale_pos_weight=6.4)),
            "xent_smooth": smoothed_fit(dict(v.LGB, n_jobs=4))}
rng = np.random.default_rng(248)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500)]
for name, fit in variants.items():
    oof, _, _ = v.run(name, fit, X, Xt, y, ids, test_ids, OUT, seeds=[42], test=False)
    d = np.array([v.ks(y[r], oof[r]) - v.ks(y[r], base[r]) for r in boots])
    fw = sum(v.ks(y[va], oof[va]) > v.ks(y[va], base[va]) for _, va in v.outer(y, 42))
    print(f"RESULT {name}: {v.ks(y, oof):.5f} vs base4 {v.ks(y, base):.5f} delta {v.ks(y, oof) - v.ks(y, base):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {(d > 0).mean():.3f} folds {fw}/5", flush=True)
