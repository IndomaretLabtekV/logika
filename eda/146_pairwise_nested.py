"""Nested (selection-leak-free) version of eda/145. The +0.0036 LGB gain of the top-30 pairwise features may be a
winner's-curse artefact: they were chosen among 7,080 using residuals of ALL training labels, including the outer
fold rows they are then scored on. Here, per outer fold (seed 42): top-60 features by LGB gain fitted on the training
fold only; reference = inner 5-fold LGB OOF on the training fold (bin-calibrated on the training fold); the 7,080
pairs are scanned on training-fold rows only; the top 30 are appended and LGB / LGB_s (validation_v2 fit: inner stop,
refit x REFIT) predict the outer fold. Compared paired with plain LGB / LGB_s seed 42 (results/screen/seeds/s42).
Usage: cd eda && ../.venv/bin/python 146_pairwise_nested.py [--top 30]"""
import argparse
import itertools

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm, rankdata
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import OUT, features
from common9 import calibrate, oe_test, safe_bins

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()
X, Xt, y, ids, tids = v.frames(features())
nums = [c for c in X.columns if c not in v.CAT_COLS]


def g(col):
    out = np.full(len(col), np.nan)
    mk = np.isfinite(col)
    out[mk] = norm.ppf((rankdata(col[mk]) - 0.5) / mk.sum())
    return out


OPS = {"diff": lambda ga, gb: ga - gb, "prod": lambda ga, gb: norm.cdf(ga) * norm.cdf(gb),
       "max": lambda ga, gb: np.fmax(norm.cdf(ga), norm.cdf(gb)), "ratio": lambda ga, gb: (norm.cdf(ga) + 0.01) / (norm.cdf(gb) + 0.01)}
fits = {"lgb": v.lgb_fit(v.LGB), "lgb_s": v.lgb_fit(v.LGB_S)}
oof = {k: np.full(len(y), np.nan) for k in fits}
for k, (tr, va) in enumerate(v.outer(y, 42)):
    P = dict(v.LGB, seed=42)
    m = lgb.train(P, lgb.Dataset(X.iloc[tr][nums], y[tr]), 400)
    top = [c for c, _ in sorted(zip(nums, m.feature_importance("gain")), key=lambda t: -t[1])[:60]]
    inner = np.full(len(tr), np.nan)
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=700 + k).split(tr, y[tr]):
        inner[b_] = lgb.train(P, lgb.Dataset(X.iloc[tr[a_]], y[tr[a_]]), 600).predict(X.iloc[tr[b_]])
    pc = calibrate(y[tr], inner)
    G = {c: g(np.concatenate([X[c].to_numpy(float), Xt[c].to_numpy(float)])) for c in top}  # unlabelled rank-gauss
    scores = []
    for c1, c2 in itertools.combinations(top, 2):
        for op, fn in OPS.items():
            x = fn(G[c1][: len(y)][tr], G[c2][: len(y)][tr])
            scores.append((oe_test(y[tr], pc, safe_bins(x, 10))[0], c1, op, c2))
    scores.sort(key=lambda t: -t[0])
    chosen = scores[: a.top]
    add = {f"pw{j}": OPS[op](G[c1], G[c2]) for j, (_, c1, op, c2) in enumerate(chosen)}
    Xa = X.assign(**{n: col[: len(y)] for n, col in add.items()})
    Xta = Xt.assign(**{n: col[len(y):] for n, col in add.items()})
    fi, st = v.inner(y[tr], 42, k)
    for name, fit in fits.items():
        (p,), _ = fit(Xa, y, tr, fi, st, [Xa.iloc[va]], 42)
        oof[name][va] = p
    print(f"fold {k}: top pair chi2 {chosen[0][0]:.2f} ({chosen[0][1]} {chosen[0][2]} {chosen[0][3]}); "
          + " ".join(f"{n} {v.ks(y[va], oof[n][va]):.4f}" for n in fits), flush=True)
for name in fits:
    base = pl.read_parquet(f"../results/screen/seeds/s42/oof_{name}.parquet").sort("application_id")[name].to_numpy()
    o = oof[name]
    r = np.random.default_rng(0)
    d = np.array([(lambda i: v.ks(y[i], o[i]) - v.ks(y[i], base[i]))(r.choice(len(y), len(y))) for _ in range(500)])
    w = sum(v.ks(y[va], o[va]) > v.ks(y[va], base[va]) for _, va in v.outer(y, 42))
    print(f"{name}: nested pw{a.top} {v.ks(y, o):.5f} vs base {v.ks(y, base):.5f} d {d.mean():+.5f} sd {d.std():.5f} "
          f"P>0 {(d > 0).mean():.2f} folds {w}/5")
    pl.DataFrame({"application_id": ids, "target": y, f"pwn_{name}": o}).write_parquet(OUT / f"146_oof_pwn_{name}.parquet")
