"""Diagnosis: univariate KS of legacy vs strict (fit-only reference) price features and of the
selected nested_small columns, on outer-train rows of fold 0 only (no outer-validation labels).
Figure: eda/out/223_strict_vs_legacy_pp.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import valid2 as v
from common import ROOT

train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
tr, _ = v.outer(y, 42)[0]
legacy = pl.read_parquet(ROOT / "data_pq/features.parquet").filter(pl.col("application_id").is_in(ids[tr].tolist())).sort("application_id")
strict = pl.read_parquet(ROOT / "results/v60/strict_tfm/pool_outer_f0/strict_features.parquet").filter(pl.col("application_id").is_in(ids[tr].tolist())).sort("application_id")
assert legacy["application_id"].to_list() == strict["application_id"].to_list() == sorted(ids[tr].tolist())
yt = y[tr]


def uks(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 100 or len(np.unique(x[ok])) < 2:
        return np.nan
    return v.ks(yt[ok], x[ok])


rows = []
for c in sorted(c for c in legacy.columns if c.startswith("pp_")):
    if c in strict.columns:
        a, b = legacy[c].to_numpy(), strict[c].to_numpy()
        both = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
        rho = np.corrcoef(np.argsort(np.argsort(a[both])), np.argsort(np.argsort(b[both])))[0, 1] if both.sum() > 2 else np.nan
        rows.append((c, uks(a), uks(b), rho, np.isnan(a.astype(float)).mean(), np.isnan(b.astype(float)).mean()))
print(f"{'col':28s} legacyKS strictKS spearman nanL nanS")
for r in rows:
    print(f"{r[0]:28s} {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}  {r[4]:.3f} {r[5]:.3f}")
d = np.array([r[2] - r[1] for r in rows], float)
print("pp columns:", len(rows), "mean strict-legacy univariate KS:", np.nanmean(d).round(5), "max drop:", np.nanmin(d).round(5))
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter([r[1] for r in rows], [r[2] for r in rows], s=12)
lim = [0, max(np.nanmax([r[1] for r in rows]), np.nanmax([r[2] for r in rows])) + .02]
ax.plot(lim, lim, "k--", lw=.8); ax.set_xlabel("legacy univariate KS"); ax.set_ylabel("strict univariate KS")
ax.set_title("pp_* price features, fold-0 outer-train rows"); fig.tight_layout()
fig.savefig(ROOT / "eda/out/223_strict_vs_legacy_pp.png", dpi=120)
