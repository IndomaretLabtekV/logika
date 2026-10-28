"""Sharper screen than within-decile KS: likelihood-ratio of adding a candidate to a logistic model on the OOF reference.
Model: y ~ natural-spline basis of logit(rank(ref)) [+ rank(candidate) + isnan(candidate)]. Reports deviance gain (chi2, 2 df)
and the cross-fitted KS of the combined score. Anchor: mk_hit3 against the base family (a block that gave +0.003 public).
Usage: cd eda && ../.venv/bin/python 256_lr_screen.py f255_process_chains[,other]
"""
import importlib
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

import common as c

scr = importlib.import_module("255_residual_screen")


def design(ref, s=None):
    u = (rankdata(ref) - 0.5) / len(ref)
    X = SplineTransformer(n_knots=8, degree=3).fit_transform(np.log(u / (1 - u))[:, None])
    if s is None:
        return X
    nan = ~np.isfinite(s)
    r = np.zeros(len(s))
    r[~nan] = (rankdata(s[~nan]) - 0.5) / (~nan).sum() - 0.5
    return np.c_[X, r, nan.astype(float)]


def dev(X, y):
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def cv_ks(X, y):
    o = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=256).split(X, y):
        o[va] = LogisticRegression(C=1e4, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return c.ks(y, o)


def main():
    ids, ref = scr.family("f251_repayment_markov")
    _, base = scr.family("base")
    d = pl.DataFrame({"application_id": ids}).join(c.features().select("application_id", "target"), on="application_id", how="left")
    d = d.join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    blocks = sys.argv[1].split(",") if len(sys.argv) > 1 else ["f255_process_chains"]
    cand = []
    for b in blocks:
        e = pl.read_parquet(c.PQ / f"{b}.parquet")
        d = d.join(e, on="application_id", how="left")
        cand += [x for x in e.columns if x != "application_id"]
    y = d["target"].to_numpy()
    d0, d0b = dev(design(ref), y), dev(design(base), y)
    k0 = cv_ks(design(ref), y)
    s = d["mk_hit3"].to_numpy().astype(float)
    print(f"anchor mk_hit3 on base ref: chi2 {d0b - dev(design(base, s), y):8.1f}  cvKS {cv_ks(design(base), y):.5f} -> {cv_ks(design(base, s), y):.5f}")
    print(f"current ref cvKS {k0:.5f}")
    print(f"{'feature':18s} {'chi2':>8s} {'cvKS delta':>11s}")
    for k in cand:
        s = d[k].to_numpy().astype(float)
        X = design(ref, s)
        print(f"{k:18s} {d0 - dev(X, y):8.1f} {cv_ks(X, y) - k0:+11.5f}", flush=True)


if __name__ == "__main__":
    main()
