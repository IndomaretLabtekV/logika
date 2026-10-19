"""v31-type blend with every component averaged over as many outer-fold seeds as we have (closer to how the test
predictions average models): LGB/LGB_s 8 seeds, XGB/XGB_s/Cat/Cat_s 3 seeds (v22), Causilo 5 seeds, Kumo 3 seeds.
Compared paired with the single-seed-ICL v31 and the clean v8 recipe; tree-share curve printed as a diagnostic only
(weights are not chosen from it). Saves eda/out/123_averaged_blend.png.
Usage: cd eda && ../.venv/bin/python 123_averaged_blend.py"""
import io
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import blend2
import valid2 as v
from common import OUT, ROOT

R = ROOT / "results"
ids = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
y, ids = ids["target"].to_numpy(), ids["application_id"]


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert not np.isnan(x).any(), p
    return x


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
S8 = [1, 2, 3, 4, 5, 7, 42, 2026]
g = {m: rk(np.mean([raw(R / f"screen/seeds/s{s}/oof_{m}.parquet") for s in S8], axis=0)) for m in ("lgb", "lgb_s")}
g |= {m: rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ("xgb", "xgb_s", "cat", "cat_s")}
trees_avg = rk(np.mean(list(g.values()), axis=0))
trees_v22 = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
cau5 = rk(np.mean([raw(R / "v22/v22/fm/oof_causilo.parquet")] + [raw(R / f"screen/seeds/s{s}/oof_causilo.parquet") for s in (1, 2, 7, 2026)], axis=0))
kum3 = rk((raw(R / "v28m/oof_kumo_large_c56000.parquet") + 2 * raw(R / "v38/oof_kumo_large_c56000.parquet")) / 3)
cau1, kum1 = rk(raw(R / "v22/v22/fm/oof_causilo.parquet")), rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))
with contextlib.redirect_stdout(io.StringIO()):
    _, _, _, _, _, res = blend2.main([str(R / "v22/v22")])
v8 = res["v8_recipe"][0]
folds = v.outer(y, 42)


def cmp(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


v31 = (trees_v22 + cau1 + kum1) / 3
v31a = (trees_avg + cau5 + kum3) / 3
print(f"trees v22 {v.ks(y, trees_v22):.5f} -> averaged {v.ks(y, trees_avg):.5f}; causilo 1 seed {v.ks(y, cau1):.5f} -> 5 seeds "
      f"{v.ks(y, cau5):.5f}; kumo 1 {v.ks(y, kum1):.5f} -> 3 {v.ks(y, kum3):.5f}")
print("v31 averaged vs v31:        ", cmp(v31a, v31))
print("v31 averaged vs v8 recipe:  ", cmp(v31a, v8))
print("trees averaged vs v31 avg:  ", cmp(trees_avg, v31a))
print("trees + kumo3 (1/3:2/3) vs v31 avg:", cmp((trees_avg + 2 * kum3) / 3, v31a))
ws = np.linspace(0, 1, 11)
curve = [v.ks(y, w * trees_avg + (1 - w) * (cau5 + kum3) / 2) for w in ws]
print("tree share curve (diagnostic): " + " ".join(f"{w:.1f}:{c:.5f}" for w, c in zip(ws, curve)))
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ws, curve, "o-")
ax.set(xlabel="tree share (rest Causilo/Kumo 1:1)", ylabel="pooled OOF KS", title="Averaged components")
fig.tight_layout()
fig.savefig(OUT / "123_averaged_blend.png", dpi=110)
