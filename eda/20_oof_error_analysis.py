"""Where does v1 fail? Segment KS, residual tests of every v1 feature, and what separates the bads v1 missed.
Uses only the saved v1 OOF predictions (results/v1); nothing is trained."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, ks_nan, resid_test, v1_oof

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
s = v1_oof()
p = calibrate(y, s)
print(f"v1 GBDT rank-blend OOF KS={ks(y, s):.4f}  calibrated-p KS={ks(y, p):.4f}  mean p={p.mean():.4f} vs y={y.mean():.4f}")

# 1) null distribution of the residual test (pure noise features)
rng = np.random.default_rng(1)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(20)])
print(f"noise feature: chi2/df p95={np.percentile(null[:, 0], 95):.2f}  dKS p95={np.percentile(null[:, 2], 95):+.4f}")

# 2) segment KS: where is v1 weak?
tr = tr.with_columns(
    s=pl.Series(s),
    age_b=(pl.col("age") // 10 * 10),
    amt_q=pl.col("loan_amount").qcut(4, labels=["q1", "q2", "q3", "q4"]),
    month=pl.col("decision_datetime").dt.month(),
    ec_act=pl.col("ec_orders_90d").qcut(4, labels=["q1", "q2", "q3", "q4"], allow_duplicates=True),
    n_scored=pl.col("bc_n_scored"),
)
print("\n=== segment KS of v1 (n, bad rate, KS)")
for seg in ["has_history", "employment_type", "loan_purpose", "region", "housing_status", "annual_interest_rate",
            "age_b", "amt_q", "ec_act", "n_scored", "dependents"]:
    rows = []
    for k, g in tr.group_by(seg, maintain_order=True):
        if g.height < 300:
            continue
        rows.append((str(k[0]), g.height, g["target"].mean(), ks(g["target"].to_numpy(), g["s"].to_numpy())))
    rows.sort(key=lambda r: r[0])
    print(f"{seg:<22}", "  ".join(f"{a}:n={n},br={b:.3f},KS={c:.3f}" for a, n, b, c in rows))

# 3) residual test of every v1 feature (should be ~noise if the model used it fully)
ID = {"application_id", "customer_id", "decision_datetime", "target", "is_train"}
num = [c for c in f.columns if c not in ID and f[c].dtype != pl.String]
res = []
for c in num:
    chi, lift, dks = resid_test(y, p, tr[c].cast(pl.Float64).to_numpy())
    res.append((c, ks_nan(y, tr[c].cast(pl.Float64).to_numpy()), chi, lift, dks))
r = pl.DataFrame(res, schema=["feature", "ks", "chi2_df", "max_lift", "dKS_cf"], orient="row").sort("chi2_df", descending=True)
with pl.Config(tbl_rows=25, float_precision=4):
    print("\n=== v1 features: residual left on the table (top 25 by chi2/df)")
    print(r.head(25))
r.write_csv(OUT / "20_resid_v1_features.csv")

# 4) missed bads: bads in the bottom half of v1 score vs goods in that same half
lo = s < np.median(s)
mb, gg = lo & (y == 1), lo & (y == 0)
print(f"\nmissed bads (bottom-half score): {mb.sum()} of {y.sum()} bads ({mb.sum() / y.sum():.1%})")
diff = []
for c in num:
    x = tr[c].cast(pl.Float64).to_numpy()
    a, b = x[mb], x[gg]
    sd = np.nanstd(x[lo]) + 1e-12
    diff.append((c, (np.nanmedian(a) - np.nanmedian(b)) / sd, np.isnan(a).mean() - np.isnan(b).mean(),
                 ks_nan(np.r_[np.ones(mb.sum()), np.zeros(gg.sum())], np.r_[a, b])))
d = pl.DataFrame(diff, schema=["feature", "median_diff_sd", "nan_rate_diff", "ks_missed_vs_good"], orient="row")
with pl.Config(tbl_rows=20, float_precision=4):
    print("=== what separates missed bads from goods with the same (low) v1 score")
    print(d.sort("ks_missed_vs_good", descending=True).head(20))

# 5) figure: score distributions + KS by segment for has_history
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for lab, m in [("good", y == 0), ("bad", y == 1)]:
    ax[0].hist(s[m], bins=50, alpha=.5, density=True, label=lab)
ax[0].set_title("v1 OOF rank score"); ax[0].legend()
dec = np.minimum((s * 20).astype(int), 19)
ax[1].plot(np.bincount(dec, weights=y) / np.bincount(dec), "o-"); ax[1].set_title("bad rate by v1 score 5%-bin")
top = r.head(8)["feature"].to_list()
for c in top:
    x = tr[c].cast(pl.Float64).to_numpy()
    from common import qbin
    b = qbin(x)
    ax[2].plot((np.bincount(b, weights=y, minlength=11) + 1) / (np.bincount(b, weights=p, minlength=11) + 1), "o-", label=c)
ax[2].axhline(1, color="k", lw=1); ax[2].set_title("observed / expected bads per decile (10=NaN)")
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "20_oof_errors.png", dpi=80)
