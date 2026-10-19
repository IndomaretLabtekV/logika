"""Feature bagging for the ICL families, full 5-fold clean OOF (seed 42). A bag = rank mean of the base ICL model
(fm_select top 100) and its random-subset members (fm_select rand=k). Compared paired (bootstrap of pooled KS over
rows, plus outer folds won) with: the single base model, and v31 (trees + Causilo + Kumo, 1/3 each) with that family
replaced by its bag.
Usage: cd eda && ../.venv/bin/python 116_fbag_eval.py causilo ../results/screen/fbag/oof_causilo_r1.parquet [...]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, ROOT

R = ROOT / "results"
fam, paths = sys.argv[1], sys.argv[2:]
ids = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
y = ids["target"].to_numpy()
ids = ids["application_id"]


def load(path):
    d = pl.read_parquet(path).sort("application_id")
    assert (d["application_id"] == ids).all()
    x = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    assert not np.isnan(x).any(), path
    return rankdata(x) / len(x)


F = {"trees": rankdata(np.mean([load(R / f"v22/v22/gbdt/oof_{m}.parquet") for m in
                                ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)) / len(y),
     "causilo": load(R / "v22/v22/fm/oof_causilo.parquet"),
     "kumo": load(R / "v28m/oof_kumo_large_c56000.parquet")}
members = [load(p) for p in paths]
bag = np.mean([F[fam], *members], axis=0)
folds = v.outer(y, 42)


def compare(a, b, n=1000):
    rng = np.random.default_rng(0)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    wins = sum(v.ks(y[va], a[va]) > v.ks(y[va], b[va]) for _, va in folds)
    return f"KS {v.ks(y, a):.5f} vs {v.ks(y, b):.5f}  d {d.mean():+.5f} sd {d.std():.5f} P>0 {(d > 0).mean():.2f} folds {wins}/5"


for p, m in zip(paths, members):
    print(f"member {p.split('/')[-1]:32s} KS {v.ks(y, m):.5f}  spearman vs base {np.corrcoef(m, F[fam])[0, 1]:.4f}")
v31 = (F["trees"] + F["causilo"] + F["kumo"]) / 3
v31b = (F["trees"] + (bag if fam == "causilo" else F["causilo"]) + (bag if fam == "kumo" else F["kumo"])) / 3
print(f"{fam} bag ({1 + len(members)}) vs {fam}:  {compare(bag, F[fam])}")
print(f"v31 with {fam} bag vs v31:     {compare(v31b, v31)}")
# member-count curve
ks_curve = [v.ks(y, np.mean([F[fam], *members[:k]], axis=0)) for k in range(len(members) + 1)]
print("bag KS by number of random members:", " ".join(f"{k}:{s:.5f}" for k, s in enumerate(ks_curve)))
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(range(len(ks_curve)), ks_curve, "o-")
ax.set(xlabel="random-subset members added", ylabel="pooled OOF KS", title=f"{fam} feature bagging")
fig.tight_layout()
fig.savefig(OUT / f"116_fbag_{fam}.png", dpi=110)
