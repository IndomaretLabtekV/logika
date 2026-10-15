"""Spearman correlation / redundancy clusters, and conditional KS: signal left *within* score deciles
(model-free proxy for incremental value). Also thin-file vs history segment ranking."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, features, ks_nan

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
num = [c for c, d in f.schema.items() if d.is_numeric() and c not in ("target", "is_train")]
uni = {c: ks_nan(y, tr[c].cast(pl.Float64).to_numpy()) for c in num}
top = sorted(num, key=lambda c: -uni[c])[:40]

R = tr.select([pl.col(c).rank() for c in top]).to_pandas().corr().to_numpy()
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar(im)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=7)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
ax.set_title("Spearman correlation, top-40 features by univariate KS")
plt.tight_layout(); plt.savefig(OUT / "11_corr.png", dpi=80)
print("highly redundant pairs (|rho|>0.95):")
for i in range(len(top)):
    for j in range(i + 1, len(top)):
        if abs(R[i, j]) > 0.95:
            print(f"  {top[i]:<22} {top[j]:<22} rho={R[i, j]:.3f}")

# conditional KS: average KS inside 10 bureau-score bins (weighted). High = adds info beyond the score.
sb = tr.with_columns(sb=pl.col("bc_score_mean").fill_null(0).rank("ordinal") * 10 // (pl.len() + 1))["sb"].to_numpy()
def cond_ks(c, groups):
    s = tr[c].cast(pl.Float64).to_numpy(); tot = 0.0
    for g in np.unique(groups):
        m = groups == g
        if y[m].min() != y[m].max():
            tot += m.mean() * ks_nan(y[m], s[m])
    return tot
rows = [(c, uni[c], cond_ks(c, sb)) for c in sorted(num, key=lambda c: -uni[c])[:45]]
res = pl.DataFrame(rows, schema=["feature", "ks", "ks_within_score_bins"], orient="row").sort("ks_within_score_bins", descending=True)
with pl.Config(tbl_rows=60, float_precision=4):
    print(res)

# segment: thin-file customers (no history) - what drives them?
thin = tr.filter(pl.col("has_history") == 0); yt = thin["target"].to_numpy()
seg = sorted([(c, ks_nan(yt, thin[c].cast(pl.Float64).to_numpy())) for c in num if thin[c].null_count() < thin.height],
             key=lambda t: -t[1])[:15]
print(f"thin-file n={thin.height} bad={yt.mean():.3f}; top KS:", [(c, round(v, 3)) for c, v in seg])
