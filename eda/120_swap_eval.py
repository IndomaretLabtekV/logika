"""Swap one v31 family for another clean OOF (full 5 folds) and compare paired with v31 and the clean v8 recipe.
A family is given as weight*path terms; each path's OOF (probability, mean over its seeds) is combined by weighted mean
of probabilities (= more seeds of the same model), then ranked. v31 = trees(6 GBDT v22) + Causilo(v22) + Kumo(v28m).
Usage: cd eda && ../.venv/bin/python 120_swap_eval.py causilo 1:../results/v22/v22/fm/oof_causilo.parquet 2:../results/v37/oof_causilo.parquet"""
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
fam, terms = sys.argv[1], sys.argv[2:]
ids = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
y, ids = ids["target"].to_numpy(), ids["application_id"]


def raw(path):
    d = pl.read_parquet(path).sort("application_id")
    assert (d["application_id"] == ids).all()
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert not np.isnan(x).any(), path
    return x


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
F = {"trees": rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in
                          ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)),
     "causilo": rk(raw(R / "v22/v22/fm/oof_causilo.parquet")),
     "kumo": rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))}
w = [float(t.split(":", 1)[0]) for t in terms]
new = rk(sum(wi * raw(t.split(":", 1)[1]) for wi, t in zip(w, terms)) / sum(w))
v31 = (F["trees"] + F["causilo"] + F["kumo"]) / 3
cand = (sum(F[k] for k in F if k != fam) + new) / 3
with contextlib.redirect_stdout(io.StringIO()):
    oof, test, names, Rm, Rt, res = blend2.main([str(R / "v22/v22")])
assert (oof["application_id"] == ids).all()
v8 = res["v8_recipe"][0] if "v8_recipe" in res else None
folds = v.outer(y, 42)


def compare(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f}  d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


print(f"{fam} new {v.ks(y, new):.5f} vs old {v.ks(y, F[fam]):.5f}; spearman {np.corrcoef(new, F[fam])[0, 1]:.5f}")
print(f"family new vs old:   {compare(new, F[fam])}")
print(f"v31 swapped vs v31:  {compare(cand, v31)}")
if v8 is not None:
    print(f"v31 swapped vs v8:   {compare(cand, v8)}")
    print(f"v31 vs v8:           {compare(v31, v8)}")
print("blend2 result keys:", list(res)[:12])
