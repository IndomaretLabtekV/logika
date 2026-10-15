"""KS is a global statistic: if the rank-averaged v6 blend puts segments (thin-file vs history, months, employment
types) on slightly different scales, the global KS is lower than what the same within-segment ordering allows.
Also: does the disagreement between models (TabM vs trees vs TabPFN) carry signal a better stacker would use?
Everything below is counting/binning on saved OOF predictions (no model is trained)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, calibrate, features, ks, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
s = v6_oof(); p = calibrate(y, s)
rng = np.random.default_rng(3)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"v6 KS {ks(y, s):.4f}; noise chi2/df p95={CHI:.2f} dKS p95={DKS:+.4f}")

names = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabpfn", "tabm", "tabm_p"]
Rk = {n: rankdata(np.load(ROOT / "results" / "v6" / f"oof_{n}.npy")) / len(y) for n in names}
trees = np.mean([Rk[n] for n in names[:6]], 0)
cands = {
    "seg_has_history": tr["has_history"].to_numpy().astype(float),
    "seg_dec_month": tr["dec_month"].to_numpy().astype(float),
    "seg_employment": tr["employment_type"].rank("dense").to_numpy().astype(float),
    "seg_region": tr["region"].rank("dense").to_numpy().astype(float),
    "seg_purpose": tr["loan_purpose"].rank("dense").to_numpy().astype(float),
    "seg_n_reports": tr["bc_n_reports"].to_numpy().astype(float),
    "seg_rate_tier": tr["annual_interest_rate"].to_numpy().astype(float),
    "dis_tabm_minus_trees": Rk["tabm"] - trees,
    "dis_tabpfn_minus_trees": Rk["tabpfn"] - trees,
    "dis_tabm_minus_tabpfn": Rk["tabm"] - Rk["tabpfn"],
    "dis_model_sd": np.std([Rk[n] for n in names], 0),
    "dis_lgb_minus_cat": Rk["lgb"] - Rk["cat"],
}
rows = [(k, *resid_test(y, p, x)) for k, x in cands.items()]
R = pl.DataFrame(rows, schema=["candidate", "chi2_df", "max_lift", "dKS_cf"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS)).sort("dKS_cf", descending=True)
with pl.Config(tbl_rows=30, float_precision=4):
    print(R)
R.write_csv(OUT / "62_blend_calibration.csv")

# within-segment KS vs global: how much does misalignment cost? Re-map each segment's scores to its own bad-rate
# (cross-fitted bins) and recompute global KS.
half = rng.random(len(y)) < 0.5
def seg_cal(seg, bins=20):
    out = np.zeros(len(y))
    for fit in (half, ~half):
        for g in np.unique(seg):
            m = seg == g
            b = np.minimum((rankdata(s[m]) * bins / (m.sum() + 1)).astype(int), bins - 1)
            r = (np.bincount(b[fit[m]], weights=y[m][fit[m]], minlength=bins) + 1) / (np.bincount(b[fit[m]], minlength=bins) + 7)
            idx = np.where(m & ~fit)[0]
            out[idx] = r[b[~fit[m]]] + 1e-6 * s[idx]
    return out
for k in ("seg_has_history", "seg_dec_month", "seg_employment", "seg_rate_tier"):
    print(f"segment-recalibrated by {k:16s}: KS {ks(y, seg_cal(cands[k])):.4f} (global-only recalibration {ks(y, seg_cal(np.zeros(len(y)))):.4f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for g, lbl in ((0, "thin"), (1, "history")):
    m = cands["seg_has_history"] == g
    q = np.quantile(s, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(q, s[m]) - 1, 0, 19)
    ax[0].plot(np.bincount(b, weights=y[m], minlength=20) / np.maximum(np.bincount(b, minlength=20), 1), "o-", label=lbl)
ax[0].set_title("bad rate per global score vigintile, by segment"); ax[0].legend(); ax[0].set_xlabel("global score vigintile")
ax[1].barh(R["candidate"], R["dKS_cf"]); ax[1].axvline(DKS, color="r", ls="--"); ax[1].set_title("dKS vs v6 (red = noise p95)")
plt.tight_layout(); plt.savefig(OUT / "62_blend_calibration.png", dpi=80)
