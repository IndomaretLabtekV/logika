"""Distilled LGB (eda/118, 5-fold seed 42) as the tree component of v31, full clean OOF.
Blends (rank average, fixed family weights, nothing fitted): v31 = trees(6 GBDT v22) + Causilo + Kumo, 1/3 each;
candidates replace LGB inside the tree family or the whole tree family with the distilled LGB. Paired bootstrap of
pooled KS over rows + outer folds won, against v31 and against plain LGB.
Usage: cd eda && ../.venv/bin/python 119_distill_eval.py out/118_oof_distill_a0.3_5f.parquet [...]"""
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import ROOT

R = ROOT / "results"
ids = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
y, ids = ids["target"].to_numpy(), ids["application_id"]


def load(path):
    d = pl.read_parquet(path).sort("application_id")
    assert (d["application_id"] == ids).all()
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert not np.isnan(x).any(), path
    return rankdata(x) / len(x)


G = {m: load(R / f"v22/v22/gbdt/oof_{m}.parquet") for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]}
cau, kumo = load(R / "v22/v22/fm/oof_causilo.parquet"), load(R / "v28m/oof_kumo_large_c56000.parquet")
trees = np.mean(list(G.values()), axis=0)
v31 = (trees + cau + kumo) / 3
folds = v.outer(y, 42)


def compare(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f}  d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


print(f"v31 {v.ks(y, v31):.5f}  trees {v.ks(y, trees):.5f}  lgb(v22, 3 seeds) {v.ks(y, G['lgb']):.5f}")
for p in sys.argv[1:]:
    d = load(p)
    name = p.split("/")[-1]
    print(f"\n{name}: alone {v.ks(y, d):.5f}")
    print(f"  vs lgb v22                   {compare(d, G['lgb'])}")
    t2 = (trees * 6 - G["lgb"] + d) / 6
    print(f"  trees lgb->distill vs trees  {compare(t2, trees)}")
    print(f"  v31 lgb->distill vs v31      {compare((t2 + cau + kumo) / 3, v31)}")
    print(f"  v31 trees->distill vs v31    {compare((d + cau + kumo) / 3, v31)}")
    print(f"  v31 trees+distill/2 vs v31   {compare(((trees + d) / 2 + cau + kumo) / 3, v31)}")
