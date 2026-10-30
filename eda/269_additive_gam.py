"""Bet 7: an additive model next to the trees.

265/268 suggest every table is a noisy reading of one customer latent (price z shifts every category equally, lateness
follows a drifting customer rate). If the target is a smooth function of a sum of such readings, an additive logistic
GAM can capture it with less variance than trees, and its errors differ from the trees'.

Model: per numeric feature a quantile-clipped cubic spline (5 knots) + a missing flag, one-hot categoricals, L2 logistic.
Same outer folds as valid2 (StratifiedKFold 5, seed), OOF only. Reports the GAM OOF KS and the rank blend with the v72
six-model trees+Markov OOF at several weights (blend weights chosen on folds of the OOF, never on the LB).
Usage: cd eda && ../.venv/bin/python 269_additive_gam.py [C] [seed]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

import common as c
import valid2 as v

V72 = Path(__file__).resolve().parent.parent / "results/v72k_out/v72/preds"
SIX = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]


def design(X, tr):
    """Fit transforms on rows tr, return the full design matrix."""
    cols = []
    cats = [k for k in X.columns if str(X[k].dtype) == "category"]
    for k in X.columns:
        if k in cats:
            continue
        x = X[k].to_numpy().astype(float)
        nan = ~np.isfinite(x)
        if np.unique(x[~nan]).size < 3:
            cols.append(np.where(nan, 0.0, x)[:, None])
            cols.append(nan[:, None].astype(float))
            continue
        lo, hi = np.nanquantile(x[tr], [0.01, 0.99])
        xc = np.clip(np.where(nan, np.nanmedian(x[tr]), x), lo, hi)
        st = SplineTransformer(n_knots=5, degree=3, knots="quantile").fit(xc[tr][:, None])
        cols.append(st.transform(xc[:, None]))
        if nan.any():
            cols.append(nan[:, None].astype(float))
    for k in cats:
        x = X[k].astype(str).to_numpy()
        for lev in np.unique(x[tr])[1:]:
            cols.append((x == lev)[:, None].astype(float))
    D = np.hstack(cols)
    mu, sd = D[tr].mean(0), D[tr].std(0) + 1e-9
    return (D - mu) / sd


def main():
    C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    feats = pl.read_parquet(c.PQ / "features.parquet").join(
        pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    XA = pd.concat([X, Xt], ignore_index=True)
    oof, test = np.zeros(len(y)), np.zeros(len(Xt))
    for k, (tr, va) in enumerate(v.outer(y, seed)):
        D = design(XA, tr)
        m = LogisticRegression(C=C, max_iter=3000).fit(D[tr], y[tr])
        oof[va] = m.decision_function(D[va])
        test += rankdata(m.decision_function(D[len(y):])) / len(Xt) / v.N_FOLDS
        print(f"fold {k}: {D.shape[1]} columns, KS {v.ks(y[va], oof[va]):.5f}", flush=True)
    print(f"GAM C={C} seed {seed}: OOF KS {v.ks(y, oof):.5f}")
    out = Path(__file__).resolve().parent.parent / "results/v73/gam"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"application_id": ids, "target": y, "gam": oof}).write_parquet(out / f"oof_gam_C{C}_s{seed}.parquet")
    pl.DataFrame({"application_id": test_ids, "gam": test}).write_parquet(out / f"test_gam_C{C}_s{seed}.parquet")

    t = pl.DataFrame({"application_id": ids})
    for mname in SIX:
        t = t.join(pl.read_parquet(V72 / f"oof_{mname}.parquet").select("application_id", mname), on="application_id",
                   how="left", maintain_order="left")
    trees = np.mean([rankdata(t[mname].to_numpy()) for mname in SIX], 0) / len(y)
    g = rankdata(oof) / len(y)
    print(f"trees OOF KS {v.ks(y, trees):.5f}, spearman trees~gam {np.corrcoef(rankdata(trees), rankdata(g))[0, 1]:.4f}")
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        b = (1 - w) * trees + w * g
        per = [v.ks(y[va], b[va]) - v.ks(y[va], trees[va]) for _, va in v.outer(y, 2026)]
        print(f"   w_gam {w:.2f}: OOF KS {v.ks(y, b):.5f}  AUC {roc_auc_score(y, b):.5f}  per-fold KS delta {np.round(per, 5)}")


if __name__ == "__main__":
    main()
