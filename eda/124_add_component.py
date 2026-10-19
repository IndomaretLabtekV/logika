"""Add one new clean OOF model (full 5 folds) to v31 as an extra family, fixed weights (rank average, nothing fitted):
v31 = trees + Causilo + Kumo (1/3 each); candidates: + new at weight w (others scaled 1-w), keeping trees >= 1/3 of the
non-new part. Paired bootstrap of pooled KS over rows + outer folds won, vs v31 and vs the clean v8 recipe; also v31
built from 3/5-seed ICL (eda/123) to avoid the seed-42 luck.
Usage: cd eda && ../.venv/bin/python 124_add_component.py ../results/v40/oof_tabr.parquet"""
import io
import contextlib
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

import blend2
import valid2 as v
from common import ROOT

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
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
cau1, kum1 = rk(raw(R / "v22/v22/fm/oof_causilo.parquet")), rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))
cau5 = rk(np.mean([raw(R / "v22/v22/fm/oof_causilo.parquet")] + [raw(R / f"screen/seeds/s{s}/oof_causilo.parquet") for s in (1, 2, 7, 2026)], axis=0))
kum3 = rk((raw(R / "v28m/oof_kumo_large_c56000.parquet") + 2 * raw(R / "v38/oof_kumo_large_c56000.parquet")) / 3)
new = rk(raw(sys.argv[1]))
with contextlib.redirect_stdout(io.StringIO()):
    _, _, _, _, _, res = blend2.main([str(R / "v22/v22")])
v8 = res["v8_recipe"][0]
folds = v.outer(y, 42)


def cmp(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f} d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


print(f"new alone {v.ks(y, new):.5f}; spearman vs trees {np.corrcoef(new, trees)[0, 1]:.4f}, causilo {np.corrcoef(new, cau1)[0, 1]:.4f}, "
      f"kumo {np.corrcoef(new, kum1)[0, 1]:.4f}")
for tag, c, k in [("seed-42 ICL", cau1, kum1), ("multi-seed ICL", cau5, kum3)]:
    base = (trees + c + k) / 3
    print(f"[{tag}] v31 {v.ks(y, base):.5f}")
    for w in (0.1, 0.2, 0.25):
        cand = (1 - w) * base + w * new
        print(f"  + new {w:.2f} vs v31: {cmp(cand, base)}")
        print(f"  + new {w:.2f} vs v8:  {cmp(cand, v8)}")
