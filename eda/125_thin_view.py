"""Thin-view model: LGB/LGB_s trained on ALL training-fold rows but only on the features thin-file customers have
(113 always-missing columns + has_history dropped; results/screen/thinview, seed 42). Thin-file rows then get a
logit average of the full model and the thin-view model; history rows keep the full model. Paired against the full
model (results/screen/seeds/s42, same folds): KS on thin rows and pooled, bootstrap over rows + folds won.
Usage: cd eda && ../.venv/bin/python 125_thin_view.py"""
import numpy as np
import polars as pl
from scipy.special import expit, logit

import valid2 as v
from common import ROOT, features

R = ROOT / "results"
f = features().filter(pl.col("is_train") == 1).sort("application_id")
ids, y = f["application_id"], f["target"].to_numpy()
thin = f["has_history"].to_numpy() == 0


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


folds = v.outer(y, 42)


def cmp(a, b, m, n=1000):
    rng = np.random.default_rng(0)
    idx = np.where(m)[0]
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(idx, len(idx)) for _ in range(n))])
    wins = sum(v.ks(y[va[m[va]]], a[va[m[va]]]) > v.ks(y[va[m[va]]], b[va[m[va]]]) for _, va in folds)
    return f"KS {v.ks(y[m], a[m]):.5f} vs {v.ks(y[m], b[m]):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


allr = np.ones(len(y), bool)
for m in ("lgb", "lgb_s"):
    full = np.clip(raw(R / f"screen/seeds/s42/oof_{m}.parquet"), 1e-6, 1 - 1e-6)
    tv = np.clip(raw(R / f"screen/thinview/oof_{m}.parquet"), 1e-6, 1 - 1e-6)
    print(f"\n{m}: thin rows {thin.sum()}, thin-view vs full on thin rows: {cmp(tv, full, thin)}")
    print(f"{m}: thin-view vs full on history rows: {cmp(tv, full, ~thin)}")
    for w in (0.5, 1.0):
        comb = full.copy()
        comb[thin] = expit((1 - w) * logit(full[thin]) + w * logit(tv[thin]))
        print(f"{m}: thin rows use {w:.1f} thin-view (logit) -> thin {cmp(comb, full, thin)}")
        print(f"{m}:   pooled {cmp(comb, full, allr)}")
