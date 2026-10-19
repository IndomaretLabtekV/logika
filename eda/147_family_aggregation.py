"""How should the 6 GBDT be aggregated into the tree family? Rank mean (current) vs probability mean vs logit mean
(both keep cross-segment calibration, which pooled KS depends on near the cut, eda/126). Clean v22 OOF, paired
bootstrap + folds won; same for the v31-style blend with each tree aggregation.
Usage: cd eda && ../.venv/bin/python 147_family_aggregation.py"""
import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata

import valid2 as v
from common import ROOT

R = ROOT / "results"


def load(p):
    d = pl.read_parquet(p).sort("application_id")
    return d["target"].to_numpy(), d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


G = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
y, _ = load(R / "v22/v22/gbdt/oof_lgb.parquet")
P = np.array([np.clip(load(R / f"v22/v22/gbdt/oof_{m}.parquet")[1], 1e-6, 1 - 1e-6) for m in G])
rk = lambda x: rankdata(x) / len(x)  # noqa: E731
agg = {"rank": rk(np.mean([rk(p) for p in P], axis=0)), "prob": rk(P.mean(0)), "logit": rk(expit(logit(P).mean(0)))}
folds = v.outer(y, 42)


def cmp(a, b, n=1000):
    r = np.random.default_rng(0)
    d = np.array([(lambda i: v.ks(y[i], a[i]) - v.ks(y[i], b[i]))(r.choice(len(y), len(y))) for _ in range(n)])
    w = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"{v.ks(y, a):.5f} vs {v.ks(y, b):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {w}/5"


for k in ("prob", "logit"):
    print(f"trees {k} vs rank: {cmp(agg[k], agg['rank'])}")
cau, kum = rk(load(R / "v22/v22/fm/oof_causilo.parquet")[1]), rk(load(R / "v28m/oof_kumo_large_c56000.parquet")[1])
v31 = (agg["rank"] + cau + kum) / 3
for k in ("prob", "logit"):
    print(f"v31 with trees-{k} vs v31: {cmp((agg[k] + cau + kum) / 3, v31)}")
