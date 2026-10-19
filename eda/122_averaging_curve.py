"""Does averaging models over outer-fold seeds help trees but not ICL? (Explains 'public prefers trees': test
predictions of trees average 15 models, OOF only 3; ICL barely varies with its context.)
Trees: LGB and LGB_s run one seed at a time (results/screen/seeds/s<seed>); KS of the probability mean over k seeds,
averaged over 30 random seed orders (removes which-seed luck). ICL: Causilo / Kumo single seed 42 vs mean of 3 seeds
(v22+v37, v28m+v38). Also the spread of single-seed KS (seed luck). Saves eda/out/122_averaging_curve.png.
Usage: cd eda && ../.venv/bin/python 122_averaging_curve.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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


seeds = sorted(int(p.name[1:]) for p in (R / "screen/seeds").glob("s*") if (p / "oof_lgb_s.parquet").exists())
print("seeds available:", seeds)
fig, ax = plt.subplots(figsize=(6, 4))
rng = np.random.default_rng(0)
for m in ("lgb", "lgb_s"):
    P = np.array([raw(R / f"screen/seeds/s{s}/oof_{m}.parquet") for s in seeds])
    single = [v.ks(y, p) for p in P]
    curve = []
    for k in range(1, len(seeds) + 1):
        curve.append(np.mean([v.ks(y, P[rng.permutation(len(seeds))[:k]].mean(0)) for _ in range(30 if k < len(seeds) else 1)]))
    print(f"{m}: single-seed KS mean {np.mean(single):.5f} sd {np.std(single):.5f} range [{min(single):.5f}, {max(single):.5f}]")
    print(f"{m}: mean-over-k curve " + " ".join(f"{k + 1}:{c:.5f}" for k, c in enumerate(curve)))
    ax.plot(range(1, len(seeds) + 1), curve, "o-", label=m)
cs = [s for s in seeds if (R / f"screen/seeds/s{s}/oof_causilo.parquet").exists()]
if cs:
    P = np.array([raw(R / "v22/v22/fm/oof_causilo.parquet")] + [raw(R / f"screen/seeds/s{s}/oof_causilo.parquet") for s in cs])
    single = [v.ks(y, p) for p in P]
    curve = [np.mean([v.ks(y, P[rng.permutation(len(P))[:k]].mean(0)) for _ in range(30 if k < len(P) else 1)])
             for k in range(1, len(P) + 1)]
    print(f"causilo seeds [42, {cs}]: single KS mean {np.mean(single):.5f} sd {np.std(single):.5f} "
          f"range [{min(single):.5f}, {max(single):.5f}]")
    print("causilo: mean-over-k curve " + " ".join(f"{k + 1}:{c:.5f}" for k, c in enumerate(curve)))
    ax.plot(range(1, len(P) + 1), curve, "o-", label="causilo")
    lg = np.array([raw(R / f"screen/seeds/s{s}/oof_lgb.parquet") for s in seeds])
    print(f"mean pairwise corr between single seeds: lgb {np.mean(np.corrcoef(lg)[np.triu_indices(len(lg), 1)]):.5f}, "
          f"causilo {np.mean(np.corrcoef(P)[np.triu_indices(len(P), 1)]):.5f}")
for name, one, three in [("causilo", R / "v22/v22/fm/oof_causilo.parquet", [R / "v37/oof_causilo.parquet"]),
                         ("kumo", R / "v28m/oof_kumo_large_c56000.parquet", [R / "v38/oof_kumo_large_c56000.parquet"])]:
    a, b = raw(one), raw(three[0])
    print(f"{name}: seed 42 {v.ks(y, a):.5f}; other 2 seeds (mean) {v.ks(y, b):.5f}; mean of 3 {v.ks(y, (a + 2 * b) / 3):.5f}; "
          f"corr(seed42, other) {np.corrcoef(a, b)[0, 1]:.5f}")
    ax.plot([1, 3], [v.ks(y, a), v.ks(y, (a + 2 * b) / 3)], "s--", label=f"{name} (seed 42 -> 3 seeds)")
ax.set(xlabel="outer-fold seeds averaged per row", ylabel="pooled OOF KS", title="Seed averaging: trees vs ICL")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "122_averaging_curve.png", dpi=110)
