"""Pre-registration numbers for an informative probe: clean trees only (6 GBDT v22, 3 seeds, rank mean) vs v31.
Research question: how much do the ICL models (Causilo + Kumo) add on the public LB? CV says +0.0017 but the seed-42
ICL OOFs were lucky (~+0.001-0.0016, eda/120). Distribution of KS(trees_only) - KS(v31) on random 30% subsets of the
clean OOF (public = 9k of 30k rows), with v31 built from seed-42 ICL and from 3-seed ICL. Saves eda/out/121_*.png.
Usage: cd eda && ../.venv/bin/python 121_trees_only_estimate.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, ROOT

R = ROOT / "results"
ids = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
y, ids = ids["target"].to_numpy(), ids["application_id"]


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
cau1, kum1 = rk(raw(R / "v22/v22/fm/oof_causilo.parquet")), rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))
cau3 = rk((raw(R / "v22/v22/fm/oof_causilo.parquet") + 2 * raw(R / "v37/oof_causilo.parquet")) / 3)
kum3 = rk((raw(R / "v28m/oof_kumo_large_c56000.parquet") + 2 * raw(R / "v38/oof_kumo_large_c56000.parquet")) / 3)
S = {"v31_seed42": (trees + cau1 + kum1) / 3, "v31_3seed": (trees + cau3 + kum3) / 3}
rng = np.random.default_rng(0)
n_pub = int(0.3 * len(y))
fig, ax = plt.subplots(figsize=(6, 4))
for name, ref in S.items():
    d = []
    for _ in range(2000):
        i = rng.choice(len(y), n_pub, replace=False)
        d.append(v.ks(y[i], trees[i]) - v.ks(y[i], ref[i]))
    d = np.array(d)
    full = v.ks(y, trees) - v.ks(y, ref)
    print(f"trees_only - {name}: full OOF {full:+.5f}; 30% subset mean {d.mean():+.5f} sd {d.std():.5f}; "
          f"P(subset diff > 0) {(d > 0).mean():.2f}; 5-95% [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]")
    ax.hist(d, 60, alpha=.5, label=f"trees_only - {name}")
ax.axvline(0, color="k", lw=.5)
ax.set(xlabel="KS difference on a random 30% subset", title="trees only vs v31 (clean OOF)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "121_trees_only_estimate.png", dpi=110)
