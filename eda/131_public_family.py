"""Single-family public probes (19 Okt) vs clean CV: public - pooled OOF offset per family, and the observed public
gap to trees-only vs the gap expected from random 30% subsets of the clean OOF (sd = subset sd of the paired KS
difference). Saves eda/out/131_public_family.png.
Usage: cd eda && ../.venv/bin/python 131_public_family.py"""
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
fam = {  # name: (clean OOF of the submitted composition, public)
    "trees only (v39a)": (trees, 0.56337),
    "Kumo only (v31c)": (rk(raw(R / "v28m/oof_kumo_large_c56000.parquet")), 0.56056),
    "LR rank-gauss (v39c)": (rk(raw(R / "screen/lin_5f/oof_lr_q.parquet")), 0.55835),
    "Causilo 3 seeds (v39b)": (rk((raw(R / "v22/v22/fm/oof_causilo.parquet") + 2 * raw(R / "v37/oof_causilo.parquet")) / 3), 0.55333),
}
rng = np.random.default_rng(0)
n = int(0.3 * len(y))
subs = [rng.choice(len(y), n, replace=False) for _ in range(2000)]
names, offs, z = [], [], []
for name, (s, pub) in fam.items():
    cv = v.ks(y, s)
    d = np.array([v.ks(y[i], s[i]) - v.ks(y[i], trees[i]) for i in subs])
    gap = pub - 0.56337
    zz = (gap - d.mean()) / d.std() if d.std() > 0 else 0.0
    print(f"{name:24s} CV {cv:.5f} public {pub:.5f} offset {pub - cv:+.4f} | public gap to trees {gap:+.5f} "
          f"vs expected {d.mean():+.5f} +- {d.std():.5f} -> z {zz:+.1f}")
    names.append(name)
    offs.append(pub - cv)
    z.append(zz)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(names, offs)
ax.axvline(offs[0], color="k", lw=.5)
ax.set(xlabel="public - clean OOF KS", title="Family offsets on the public LB")
fig.tight_layout()
fig.savefig(OUT / "131_public_family.png", dpi=110)
