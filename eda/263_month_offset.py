"""Bet 8: calendar offset on top of trees+Markov. Logistic model y ~ spline(logit rank(ref)) + group dummies (L2),
cross-fitted over the outer folds (seed 42, 7, 2026 splits): offsets learnt on 4 folds, scored on the 5th.
Compared with the same model without dummies, so the gain is the offset alone. Groups: decision year-month, has_history.
"""
import importlib

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

import common as c
import valid2 as v

scr = importlib.import_module("255_residual_screen")


def spline(ref):
    u = (rankdata(ref) - 0.5) / len(ref)
    return SplineTransformer(n_knots=8, degree=3).fit_transform(np.log(u / (1 - u))[:, None])


def dummies(s):
    levels = sorted(s.unique().to_list())
    return np.column_stack([(s == l).to_numpy().astype(float) for l in levels])


def crossfit(X, y, C, seed):
    o = np.zeros(len(y))
    for tr, va in v.outer(y, seed):
        o[va] = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr]).decision_function(X[va])
    return o


def main():
    ids, ref = scr.family("f251_repayment_markov")
    f = c.features().select("application_id", "target", "has_history", ym=pl.col("decision_datetime").dt.strftime("%Y-%m"))
    d = pl.DataFrame({"application_id": ids}).join(f, on="application_id", how="left", maintain_order="left")
    assert (d["application_id"] == ids).all()
    y = d["target"].to_numpy()
    S = spline(ref)
    print(f"ref KS {v.ks(y, ref):.5f}")
    print(d.group_by("ym").agg(pl.len(), pl.col("target").mean()).sort("ym"))
    groups = {"month": dummies(d["ym"]), "has_history": dummies(d["has_history"]),
              "month+hist": np.c_[dummies(d["ym"]), dummies(d["has_history"])]}
    for seed in (42, 7, 2026):
        base = crossfit(S, y, 1e4, seed)
        kb = [v.ks(y[va], base[va]) for _, va in v.outer(y, seed)]
        for name, G in groups.items():
            for C in (0.03, 0.3, 3.0):
                X = np.c_[S * 100, G]  # large scale on spline so the L2 effectively only shrinks the offsets
                o = crossfit(X, y, C, seed)
                kn = [v.ks(y[va], o[va]) for _, va in v.outer(y, seed)]
                dl = np.array(kn) - np.array(kb)
                print(f"seed {seed} {name:12s} C {C:<5} pooled {v.ks(y, o) - v.ks(y, base):+.5f} "
                      f"fold-mean {dl.mean():+.5f} wins {(dl > 0).sum()}/5")


if __name__ == "__main__":
    main()
