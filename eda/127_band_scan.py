"""Pooled KS only reacts to the ordering around one global cut (eda/126). Which features separate bad from good
INSIDE the band around the v31 KS cut, beyond the v31 score? Band = v31 OOF rank in [0.55, 0.80] (cut at ~0.67).
Per feature (safe bins): O/E chi2/df vs the bin-calibrated v31 probability, restricted to band rows, plus a cross-fit
dKS of the band-local adjustment measured on POOLED KS (the band adjustment applied, everything else untouched).
Null = same with a permuted feature. Saves eda/out/127_band_scan.png / .csv.
Usage: cd eda && ../.venv/bin/python 127_band_scan.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, ROOT, features
from common9 import calibrate, oe_test, safe_bins

R = ROOT / "results"
f = features().filter(pl.col("is_train") == 1).sort("application_id")
ids, y = f["application_id"], f["target"].to_numpy()


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == ids).all()
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
s = (trees + rk(raw(R / "v22/v22/fm/oof_causilo.parquet")) + rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))) / 3
r = rk(s)
p = calibrate(y, s)
band = (r >= 0.55) & (r <= 0.80)
cut = np.argmax([abs(np.mean(r[y == 1] <= t) - np.mean(r[y == 0] <= t)) for t in np.linspace(0, 1, 401)]) / 400
print(f"v31 KS {v.ks(y, s):.5f}; KS cut at rank {cut:.3f}; band rows {band.sum()} bad rate {y[band].mean():.3f} "
      f"thin share band {np.mean(f['has_history'].to_numpy()[band] == 0):.3f} vs all {np.mean(f['has_history'].to_numpy() == 0):.3f}")
lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def band_dks(b, reps=3, seed=0):
    """Cross-fit: bin effects (log O/E, +5 smoothing) fitted on band rows of one half, applied to band rows of the
    other half; gain in POOLED KS."""
    _, inv = np.unique(b, return_inverse=True)
    K = inv.max() + 1
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(reps):
        half = rng.random(len(y)) < 0.5
        adj = np.zeros(len(y))
        for fit in (half, ~half):
            m = fit & band
            o = np.bincount(inv[m], weights=y[m], minlength=K) + 5
            e = np.bincount(inv[m], weights=p[m], minlength=K) + 5
            app = ~fit & band
            adj[app] = np.log(o / e)[inv[app]]
        g.append(v.ks(y, lp + adj) - v.ks(y, lp))
    return float(np.mean(g))


rng = np.random.default_rng(1)
pp = f["pp_mean"].to_numpy()
null = [(oe_test(y[band], p[band], safe_bins(z[band], 10))[0], band_dks(safe_bins(z, 10)))
        for z in (rng.permutation(pp) for _ in range(30))]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null (permuted feature): band chi2/df p95 {c95:.2f}, pooled dKS p95 {d95:+.5f}")
rows = []
for c in f.columns:
    if c in v.ID_COLS or f[c].dtype == pl.String:
        continue
    x = f[c].cast(pl.Float64).to_numpy()
    b = safe_bins(x, 10)
    rows.append(dict(feature=c, chi2=oe_test(y[band], p[band], b[band])[0], dks=band_dks(b)))
d = pl.DataFrame(rows).sort("chi2", descending=True)
d.write_csv(OUT / "127_band_scan.csv")
print(d.head(20))
print("above both nulls:", d.filter((pl.col("chi2") > c95) & (pl.col("dks") > d95)).height, "of", d.height)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(d["chi2"], d["dks"], s=8)
ax.axvline(c95, color="r", lw=.5)
ax.axhline(d95, color="r", lw=.5)
for row in d.head(8).iter_rows(named=True):
    ax.annotate(row["feature"], (row["chi2"], row["dks"]), fontsize=6)
ax.set(xlabel="band chi2/df vs v31", ylabel="pooled cross-fit dKS (band-only adjustment)", title="Band-local residual scan")
fig.tight_layout()
fig.savefig(OUT / "127_band_scan.png", dpi=110)
