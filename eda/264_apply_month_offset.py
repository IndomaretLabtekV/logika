"""Applies the 263 calendar offset to a rank-mean submission. Fit on train OOF (trees+Markov family, 18 fits):
logit P(y) = spline(logit rank) + month offset (L2, C=0.3). Test: same spline on the test rank, plus the month offset.
Usage: cd eda && ../.venv/bin/python 264_apply_month_offset.py <in.csv> <out.csv>
"""
import importlib
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

import common as c

scr = importlib.import_module("255_residual_screen")
C = 0.3


def logit_rank(s):
    u = (rankdata(s) - 0.5) / len(s)
    return np.log(u / (1 - u))[:, None]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ids, ref = scr.family("f251_repayment_markov")
    ym = c.features().select("application_id", ym=pl.col("decision_datetime").dt.strftime("%Y-%m"), y=pl.col("target"))
    tr = pl.DataFrame({"application_id": ids, "ref": ref}).join(ym, on="application_id", how="left", maintain_order="left")
    te = pl.read_csv(src).join(ym, on="application_id", how="left", maintain_order="left")
    months = sorted(tr["ym"].unique().to_list())
    assert set(te["ym"].unique().to_list()) <= set(months)
    sp = SplineTransformer(n_knots=8, degree=3, extrapolation="linear").fit(logit_rank(tr["ref"].to_numpy()))
    D = lambda s: np.column_stack([(s == m).to_numpy().astype(float) for m in months])
    Xtr = np.c_[sp.transform(logit_rank(tr["ref"].to_numpy())) * 100, D(tr["ym"])]
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, tr["y"].to_numpy())
    off = dict(zip(months, m.coef_[0][-len(months):].round(4)))
    print("month offsets:", off)
    Xte = np.c_[sp.transform(logit_rank(te["default_probability"].to_numpy())) * 100, D(te["ym"])]
    p = 1 / (1 + np.exp(-m.decision_function(Xte)))
    out = te.select("application_id").with_columns(default_probability=pl.Series(p))
    assert out.height == 30000 and out["application_id"].n_unique() == 30000 and out.null_count().sum_horizontal()[0] == 0
    from scipy.stats import spearmanr
    print("spearman vs input", spearmanr(p, te["default_probability"].to_numpy())[0])
    out.write_csv(dst)


if __name__ == "__main__":
    main()
