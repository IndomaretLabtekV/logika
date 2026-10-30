"""Blend scheme on the existing 8-seed OOF (v72 + v73) of the 6 trees+Markov models, plus the 3 v72-only extra models.
Compares equal rank mean, equal mean of calibrated logits (isotonic, cross-fitted), and weights fitted on folds
(nonnegative, sum 1, coordinate search on 4 folds, scored on the 5th). Paired bootstrap of the KS change vs equal rank.
Usage: cd eda && ../.venv/bin/python 275_blend_scheme.py
"""
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import valid2 as v

R = Path(__file__).resolve().parent.parent / "results"
SIX = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
EXTRA = ["lgb_x", "xgb_d8", "cat_d8"]


def load():
    a = pl.read_parquet(R / "v72k_out/v72/preds/oof_lgb.parquet").select("application_id", "target")
    y, cols = a["target"].to_numpy(), {}
    for m in SIX + EXTRA:
        x = pl.read_parquet(R / f"v72k_out/v72/preds/oof_{m}.parquet")[m].to_numpy()
        if m in SIX:
            x = (3 * x + 5 * pl.read_parquet(R / f"v73k_out/v72/preds/oof_{m}.parquet")[m].to_numpy()) / 8
        cols[m] = rankdata(x) / len(x)
    return y, cols


def logit_cal(s, y):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=275).split(s, y):
        p[va] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s[tr], y[tr]).predict(s[va])
    return np.log(p / (1 - p))


def fit_weights(M, y, iters=3, grid=np.linspace(0, 1, 11)):
    w = np.ones(M.shape[1]) / M.shape[1]
    for _ in range(iters):
        for j in range(M.shape[1]):
            best = None
            for g in grid:
                u = w.copy()
                u[j] = g
                if u.sum() == 0:
                    continue
                k = v.ks(y, M @ (u / u.sum()))
                if best is None or k > best[0]:
                    best = (k, u / u.sum())
            w = best[1]
    return w


def boot(y, a, b, n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(v.ks(y[i], a[i]) - v.ks(y[i], b[i]))
    return np.mean(d), np.mean(np.array(d) > 0)


def main():
    y, cols = load()
    for names, tag in ((SIX, "six"), (SIX + EXTRA, "nine")):
        M = np.column_stack([cols[m] for m in names])
        base = M.mean(1)
        lg = np.column_stack([logit_cal(cols[m], y) for m in names]).mean(1)
        cv = np.zeros(len(y))
        ws = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=2750).split(M, y):
            w = fit_weights(M[tr], y[tr])
            ws.append(w)
            cv[va] = M[va] @ w
        print(f"{tag}: equal rank KS {v.ks(y, base):.5f}")
        for nm, s in (("logit mean", lg), ("fold-fitted weights", cv)):
            mu, pp = boot(y, s, base)
            print(f"   {nm:20s} KS {v.ks(y, s):.5f}  boot delta {mu:+.5f} P>0 {pp:.2f}")
        print("   mean fitted weights", dict(zip(names, np.round(np.mean(ws, 0), 3))))
    for m in SIX + EXTRA:
        print(f"   single {m:7s} KS {v.ks(y, cols[m]):.5f}")


if __name__ == "__main__":
    main()
