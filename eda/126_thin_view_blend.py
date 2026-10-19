"""v31 with thin-view models on thin-file rows (eda/125). For every model that has a thin-view run
(results/screen/thinview, seed 42): thin-file rows get logit-mean(full OOF, thin-view OOF), history rows keep the full
OOF. Families then rank-averaged as in v31 (trees 1/3 = 6 GBDT, Causilo 1/3, Kumo 1/3; Kumo has no thin view unless
present). Paired bootstrap pooled KS + folds won vs v31 and vs the clean v8 recipe; per-segment KS printed.
Usage: cd eda && ../.venv/bin/python 126_thin_view_blend.py"""
import io
import contextlib

import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata

import blend2
import valid2 as v
from common import ROOT, features

R = ROOT / "results"
f = features().filter(pl.col("is_train") == 1).sort("application_id")
ids, y = f["application_id"], f["target"].to_numpy()
thin = f["has_history"].to_numpy() == 0


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert not np.isnan(x).any(), p
    return np.clip(x, 1e-6, 1 - 1e-6)


def tv(full, name):
    p = R / f"screen/thinview/oof_{name}.parquet"
    if not p.exists():
        return full, False
    out = full.copy()
    out[thin] = expit((logit(full[thin]) + logit(raw(p)[thin])) / 2)
    return out, True


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
G = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
full = {m: raw(R / f"v22/v22/gbdt/oof_{m}.parquet") for m in G}
full["causilo"] = raw(R / "v22/v22/fm/oof_causilo.parquet")
full["kumo_large_c56000"] = raw(R / "v28m/oof_kumo_large_c56000.parquet")
new, has = {}, {}
for m, x in full.items():
    new[m], has[m] = tv(x, m)
print("thin view available:", [m for m, h in has.items() if h])


def blend(P):
    trees = rk(np.mean([rk(P[m]) for m in G], axis=0))
    return (trees + rk(P["causilo"]) + rk(P["kumo_large_c56000"])) / 3, trees


v31, t_old = blend(full)
v31t, t_new = blend(new)
with contextlib.redirect_stdout(io.StringIO()):
    _, _, _, _, _, res = blend2.main([str(R / "v22/v22")])
v8 = res["v8_recipe"][0]
folds = v.outer(y, 42)


def cmp(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


for m in full:
    if has[m]:
        print(f"{m:18s} thin KS {v.ks(y[thin], full[m][thin]):.5f} -> {v.ks(y[thin], new[m][thin]):.5f}; pooled {v.ks(y, full[m]):.5f} -> {v.ks(y, new[m]):.5f}")
print("trees family thin-view vs trees:", cmp(t_new, t_old))
print("v31 thin-view vs v31:           ", cmp(v31t, v31))
print("v31 thin-view vs v8 recipe:     ", cmp(v31t, v8))
for nm, m in [("thin", thin), ("history", ~thin)]:
    print(f"segment {nm}: v31 {v.ks(y[m], v31[m]):.5f} -> {v.ks(y[m], v31t[m]):.5f}")


def within(seg_new, base, m):
    """Keep base's score values inside segment m, reorder them by seg_new (no labels used)."""
    out = base.copy()
    idx = np.where(m)[0]
    out[idx] = np.sort(base[idx])[np.argsort(np.argsort(seg_new[idx]))]
    return out


v31q = within(v31t, v31, thin)
print("v31 + thin reorder (quantile-kept) vs v31:", cmp(v31q, v31))
print("v31 + thin reorder vs v8 recipe:          ", cmp(v31q, v8))
print(f"  thin {v.ks(y[thin], v31q[thin]):.5f} history {v.ks(y[~thin], v31q[~thin]):.5f}")
