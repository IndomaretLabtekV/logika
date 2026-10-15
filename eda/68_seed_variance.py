"""How much KS does variance reduction (more seeds / fold splits) buy? v5 and v6 are nearly the same models fitted on
different seeds and splits (feature difference: time-safe price reference only), so averaging them measures the gain
of doubling the ensemble directly, on saved OOF. Weights are fixed (no fitting), so no selection bias.
Also: per model family, KS of the v6 3-seed average vs the v5 3-seed average vs their mean -> extrapolate to 5 seeds."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, features, ks, v5_oof, v6_oof

f = features(); y = f.filter(pl.col("is_train") == 1)["target"].to_numpy(); n = len(y)
L = lambda v, m: rankdata(np.load(ROOT / "results" / v / f"oof_{m}.npy")) / n  # noqa: E731
rows = []
for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabm"]:
    a, b = L("v5", m), L("v6", m)
    rows.append((m, ks(y, a), ks(y, b), ks(y, (a + b) / 2), float(np.corrcoef(a, b)[0, 1])))
R = pl.DataFrame(rows, schema=["model", "ks_v5", "ks_v6", "ks_mean_v5_v6", "rank_corr"], orient="row").with_columns(
    gain=pl.col("ks_mean_v5_v6") - (pl.col("ks_v5") + pl.col("ks_v6")) / 2)
with pl.Config(float_precision=4):
    print(R)
s5, s6 = v5_oof(), v6_oof()
tr5 = np.mean([L("v5", m) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], 0)
tr6 = np.mean([L("v6", m) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], 0)
print(f"equal trees: v5 {ks(y, tr5):.4f} v6 {ks(y, tr6):.4f} mean {ks(y, (tr5 + tr6) / 2):.4f}")
print(f"submitted blends: v5 {ks(y, s5):.4f} v6 {ks(y, s6):.4f} 50/50 {ks(y, rankdata(s5) + rankdata(s6)):.4f}")
rng = np.random.default_rng(0)
d = []
for _ in range(300):
    i = rng.integers(0, n, n)
    d.append(ks(y[i], (rankdata(s5) + rankdata(s6))[i]) - ks(y[i], s6[i]))
d = np.array(d); print(f"bootstrap 50/50 - v6: mean {d.mean():+.4f} p5 {np.percentile(d, 5):+.4f} P>0 {np.mean(d > 0):.2f}")
R.write_csv(OUT / "68_seed_variance.csv")
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(R.height)
for k, c in enumerate(["ks_v5", "ks_v6", "ks_mean_v5_v6"]):
    ax.bar(x + (k - 1) * 0.27, R[c], 0.27, label=c)
ax.set_xticks(x, R["model"]); ax.set_ylim(0.51, 0.535); ax.legend(); ax.set_title("OOF KS: v5, v6 and their average (2x seeds)")
plt.tight_layout(); plt.savefig(OUT / "68_seed_variance.png", dpi=80)
