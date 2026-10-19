"""Snapshot-delta features (eda/142) in LGB / LGB_s over 3 outer seeds (42/7/2026) vs the same seeds without them
(results/screen/seeds/s*), paired bootstrap + folds won; then at tree-family level (rank mean of 6 GBDT, members with
a delta run replaced by their 3-seed delta version, others from v22). Optional extra members as args (xgb, cat, ...)
if results/screen/snapd*/oof_<m>.parquet exist for all three seeds.
Usage: cd eda && ../.venv/bin/python 143_snapd_eval.py"""
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import ROOT

R = ROOT / "results"
D = {42: "snapd", 7: "snapd_s7", 2026: "snapd_s2026"}


def load(p):
    d = pl.read_parquet(p).sort("application_id")
    return d["target"].to_numpy(), d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


y, _ = load(R / "v22/v22/gbdt/oof_lgb.parquet")
folds = v.outer(y, 42)


def cmp(a, b, n=500):
    r = np.random.default_rng(0)
    d = np.array([(lambda i: v.ks(y[i], a[i]) - v.ks(y[i], b[i]))(r.choice(len(y), len(y))) for _ in range(n)])
    w = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"{v.ks(y, a):.5f} vs {v.ks(y, b):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {w}/5"


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
G = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
new = {}
for m in G:
    if all((R / f"screen/{d}/oof_{m}.parquet").exists() for d in D.values()):
        a = np.mean([load(R / f"screen/{d}/oof_{m}.parquet")[1] for d in D.values()], axis=0)
        b = np.mean([load(R / f"screen/seeds/s{s}/oof_{m}.parquet")[1] for s in D], axis=0) if m in ("lgb", "lgb_s") \
            else load(R / f"v22/v22/gbdt/oof_{m}.parquet")[1]
        print(f"{m:6s} 3 seeds +deltas vs without: {cmp(a, b)}")
        new[m] = a
old = {m: load(R / f"v22/v22/gbdt/oof_{m}.parquet")[1] for m in G}
fam_old = rk(np.mean([rk(old[m]) for m in G], axis=0))
fam_new = rk(np.mean([rk(new.get(m, old[m])) for m in G], axis=0))
print(f"tree family (replaced {sorted(new)}) vs v22 trees: {cmp(fam_new, fam_old)}")
