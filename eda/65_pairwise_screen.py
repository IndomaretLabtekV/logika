"""Two-feature signal the univariate residual screens cannot see.
 A) log-ratios of every pair of non-negative money features (trees split on one axis at a time and cannot form a
    ratio; the price premium was a ratio).
 B) 2-D interaction cells (5x5 quantile grid) for every pair among the 30 strongest features.
Each candidate is tested against the calibrated v6 blend; because hundreds of pairs are tested, the threshold is the
max over 200 random features (family-wise) and a hit must replicate on both halves of the training rows."""
import itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks_nan, qbin, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v6_oof())
rng = np.random.default_rng(21)
half = rng.random(len(y)) < 0.5


def test(x):
    a = resid_test(y[half], p[half], x[half], bins=10)
    b = resid_test(y[~half], p[~half], x[~half], bins=10)
    return a[0], b[0], a[2], b[2]


null = np.array([test(rng.random(len(y))) for _ in range(200)])
CHI_MAX, DKS_MAX = null[:, :2].max(), null[:, 2:].max()
print(f"family-wise noise (max of 200 random, per half): chi2/df {CHI_MAX:.2f}  dKS {DKS_MAX:+.4f}")

num = [c for c in f.columns if f[c].dtype not in (pl.String, pl.Datetime("us")) and c not in ("target", "is_train")]
M = {c: tr[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) for c in num}
money = [c for c in num if np.nanmin(M[c]) >= 0 and np.nanmax(M[c]) > 1e5]
print(f"A) {len(money)} money features -> {len(money) * (len(money) - 1) // 2} log-ratios")
rows = []
for a, b in itertools.combinations(money, 2):
    x = np.log1p(M[a]) - np.log1p(M[b])
    rows.append(("ratio", f"{a}/{b}", ks_nan(y, x), *test(x)))
ksu = sorted(((ks_nan(y, M[c]), c) for c in num), reverse=True)[:30]
top = [c for _, c in ksu]
print(f"B) top-30 features by KS -> {30 * 29 // 2} interaction grids")
for a, b in itertools.combinations(top, 2):
    x = (qbin(M[a], 5) * 6 + qbin(M[b], 5)).astype(float)  # NaN bin = 5 per axis; cells as one categorical axis
    # resid_test bins by quantile of x; with 36 discrete codes use the codes directly via a rank-preserving map
    rows.append(("grid", f"{a} x {b}", np.nan, *test(x)))
R = pl.DataFrame(rows, schema=["kind", "candidate", "ks", "chi_h1", "chi_h2", "dks_h1", "dks_h2"], orient="row")
R = R.with_columns(replicates=(pl.min_horizontal("chi_h1", "chi_h2") > CHI_MAX) & (pl.min_horizontal("dks_h1", "dks_h2") > DKS_MAX),
                   dks_min=pl.min_horizontal("dks_h1", "dks_h2")).sort("dks_min", descending=True)
R.write_csv(OUT / "65_pairwise_screen.csv")
with pl.Config(tbl_rows=25, float_precision=4, fmt_str_lengths=60):
    print(R.head(25))
print("replicated hits:", R["replicates"].sum(), "of", R.height)

fig, ax = plt.subplots(figsize=(8, 5))
for k, c in (("ratio", "C0"), ("grid", "C1")):
    q = R.filter(pl.col("kind") == k)
    ax.scatter(q["dks_h1"], q["dks_h2"], s=6, c=c, label=k)
ax.axvline(DKS_MAX, color="r", ls="--"); ax.axhline(DKS_MAX, color="r", ls="--")
ax.set_xlabel("dKS vs v6, half 1"); ax.set_ylabel("dKS vs v6, half 2"); ax.legend(); ax.set_title("pairwise candidates (red = family-wise noise)")
plt.tight_layout(); plt.savefig(OUT / "65_pairwise_screen.png", dpi=80)
