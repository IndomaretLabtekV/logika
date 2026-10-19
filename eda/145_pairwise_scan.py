"""OpenFE-style pairwise scan against the clean TREES-ONLY OOF (eda/65 did ~1,200 against the biased v8 OOF).
The 60 numeric features with the highest LGB gain (fitted on all training rows, 400 rounds — used only to choose
which pairs to try) -> every pair (a, b): a - b and a / b on rank-gauss scale, rank(a) * rank(b), max(rank a, rank b)
-> O/E chi2/df per safe bin vs the bin-calibrated trees OOF. Family-wise threshold from a permuted-feature null max.
Top 30 are written to eda/out/145_pairwise_top.parquet for an LGB --extra check.
Usage: cd eda && ../.venv/bin/python 145_pairwise_scan.py"""
import io
import contextlib
import itertools

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm, rankdata

import blend2
import valid2 as v
from common import OUT, features
from common9 import calibrate, oe_test, safe_bins

X, Xt, y, ids, tids = v.frames(features())
nums = [c for c in X.columns if c not in v.CAT_COLS]
m = lgb.train(dict(v.LGB, seed=0), lgb.Dataset(X[nums], y), 400)
top = [c for c, _ in sorted(zip(nums, m.feature_importance("gain")), key=lambda t: -t[1])[:60]]
allX = np.vstack([X[top].to_numpy(float), Xt[top].to_numpy(float)])


def g(col):
    """Rank-gauss on train+test values of the column (unlabelled); NaN stays NaN."""
    out = np.full(len(col), np.nan)
    mk = np.isfinite(col)
    out[mk] = norm.ppf((rankdata(col[mk]) - 0.5) / mk.sum())
    return out


Gs = np.column_stack([g(allX[:, j]) for j in range(len(top))])[: len(y)]
Rk = norm.cdf(Gs)
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == ids).all()
pc = calibrate(y, res["equal_trees"][0])
ops = {"diff": lambda a, b: Gs[:, a] - Gs[:, b], "prod": lambda a, b: Rk[:, a] * Rk[:, b],
       "max": lambda a, b: np.fmax(Rk[:, a], Rk[:, b]), "ratio": lambda a, b: (Rk[:, a] + 0.01) / (Rk[:, b] + 0.01)}
rows = []
for a, b in itertools.combinations(range(len(top)), 2):
    for name, fn in ops.items():
        x = fn(a, b)
        rows.append((f"{top[a]}{'|'}{name}{'|'}{top[b]}", oe_test(y, pc, safe_bins(x, 10))[0]))
d = pl.DataFrame(rows, schema=["feature", "chi2"], orient="row").sort("chi2", descending=True)
rng = np.random.default_rng(0)
null = []
for _ in range(300):
    a, b = rng.choice(len(top), 2, replace=False)
    null.append(oe_test(y, pc, safe_bins(rng.permutation(ops["diff"](a, b)), 10))[0])
null = np.array(null)
thr = np.quantile(null, 0.95) + np.std(null) * (np.sqrt(2 * np.log(d.height)) - 1.645)
print(f"{d.height} pairwise candidates; permuted chi2/df p95 {np.quantile(null, .95):.2f}, family-wise thr ~{thr:.2f}; "
      f"above thr {d.filter(pl.col('chi2') > thr).height}")
with pl.Config(tbl_rows=30, fmt_str_lengths=80, tbl_width_chars=140):
    print(d.head(25))
d.write_csv(OUT / "145_pairwise_scan.csv")
cols = {}
for name in d.head(30)["feature"]:
    a_, op, b_ = name.split("|")
    full = np.vstack([X[[a_, b_]].to_numpy(float), Xt[[a_, b_]].to_numpy(float)])
    ga, gb = g(full[:, 0]), g(full[:, 1])
    ra, rb = norm.cdf(ga), norm.cdf(gb)
    cols["pw_" + name.replace("|", "_")] = {"diff": ga - gb, "prod": ra * rb, "max": np.fmax(ra, rb), "ratio": (ra + 0.01) / (rb + 0.01)}[op]
pl.DataFrame({"application_id": pl.concat([ids, tids]), **cols}).write_parquet(OUT / "145_pairwise_top.parquet")
