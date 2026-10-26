"""Trees with a small NESTED selection of teammate prefix features (customer-local t__*, label-free builders).
Per outer fold (seed 42): LGB on main232 + all t__ columns fitted on the inner-fit split ranks t__ columns by
gain; the top 30 are added to main232 and the usual v22 LGB (inner-stop budget, refit) is trained. Outer labels
are never used for selection. Paired with lgb_base4. Untested combination: full union and univariate-KS
selection were tested (both hurt); model-gain selection of a few complementary columns was not.
Usage: cd eda && ../.venv/bin/python 249_nested_teammate_select.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import valid2 as v
from common import ROOT

TOP = 30
X, Xt, y, ids, test_ids = v.frames(pl.read_parquet(ROOT / "data_pq/features.parquet"))
pool = pl.read_parquet(ROOT / "results/v60/strict_tfm/pool_outer_f0/strict_features.parquet")
tcols = [c for c in pool.columns if c.startswith("t__")]
T = pool.filter(pl.col("application_id").is_in(ids.to_list())).sort("application_id")
assert T["application_id"].to_list() == ids.to_list()
T = T.select(tcols).to_pandas().astype(float).replace([np.inf, -np.inf], np.nan)
XA = pd.concat([X.reset_index(drop=True), T], axis=1)
base_fit = v.lgb_fit(dict(v.LGB, n_jobs=4))
chosen = {}


def fit(Xf, yy, tr, fi, st, rest, seed):
    m = lgb.train(dict(v.LGB, n_jobs=4, seed=seed), lgb.Dataset(XA.iloc[tr[fi]], yy[tr[fi]]), 600)
    gain = dict(zip(XA.columns, m.feature_importance("gain")))
    top = sorted(tcols, key=lambda c: -gain[c])[:TOP]
    chosen[len(chosen)] = top
    cols = list(X.columns) + top
    Xs = XA[cols]
    return base_fit(Xs, yy, tr, fi, st, [Xs.iloc[np.asarray(r.index)] for r in rest], seed)


oof, _, _ = v.run("lgb_tsel30", fit, X, Xt, y, ids, test_ids, ROOT / "results/v66/teammate_select", seeds=[42], test=False)
base = pl.read_parquet(ROOT / "results/v66/series_oof_eval/oof_lgb_base4.parquet")["lgb_base4"].to_numpy()
rng = np.random.default_rng(249)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
d = np.array([v.ks(y[r], oof[r]) - v.ks(y[r], base[r]) for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))])
fw = sum(v.ks(y[va], oof[va]) > v.ks(y[va], base[va]) for _, va in v.outer(y, 42))
print(f"RESULT tsel30: {v.ks(y, oof):.5f} vs base4 {v.ks(y, base):.5f} delta {v.ks(y, oof) - v.ks(y, base):+.5f} "
      f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {(d > 0).mean():.3f} folds {fw}/5")
common = set.intersection(*map(set, chosen.values()))
print("columns chosen in all folds:", sorted(common)[:15])
