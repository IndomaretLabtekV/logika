"""Remove nuisance variance from the signal channels.

eda (within-class correlations): pp_mean correlates 0.26 with declared income inside BOTH classes, i.e. part of the
price premium is wealth, not stress. If income itself carries little target signal, pp_mean minus its income-predicted
part is a cleaner measurement of the stress latent. A tree needs many splits on two variables to approximate such a
difference. For each channel: regress its rank-normal score on nuisance variables (log income, age, tenure, dependents,
employment / housing / region dummies) using ALL 100k customers (no labels), and compare univariate KS of raw vs
residual and the O/E test of the residual against v8. Also reports the within-class R^2 of each channel on the
nuisance block (the share of non-target variance that could be removed)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm, rankdata
from common import ks_nan
from common9 import OUT, crossfit_dks, features, oe_test, safe_bins, v8_ref

f = features()
tr, y, p = v8_ref()
istr = f["is_train"].to_numpy() == 1
def rn(x):
    out = np.full(len(x), np.nan); m = np.isfinite(x)
    out[m] = norm.ppf((rankdata(x[m]) - 0.5) / m.sum()); return out
N = np.column_stack([np.log(f["declared_monthly_income"].to_numpy()), f["age"].to_numpy(), f["employment_tenure_months"].to_numpy(),
                     f["dependents"].to_numpy()]
                    + [(f[c] == v).to_numpy().astype(float) for c in ("employment_type", "housing_status", "region") for v in f[c].unique().sort().to_list()[1:]])
N = np.column_stack([np.ones(len(f)), (N - N.mean(0)) / N.std(0)])
CH = ["pp_mean", "pp_trim", "pp_done", "pp_90", "ec_accel_30d", "ec2_ewm28_exc", "ec_spend_per_month", "ec_aov", "rp_frac_dpd1",
      "rp_late_mean", "bc_score_mean", "bc_util_mean", "ac_lowdays_mean", "ac_income_mean", "income_verif_ratio", "dti_verified",
      "cf_nonincome_share", "ec_n_orders", "pp_qty_item"]
yy = np.full(len(f), np.nan); yy[istr] = f["target"].to_numpy()[istr]
pos = f.select("application_id").with_row_index("i").join(tr.select("application_id"), on="application_id", maintain_order="right")["i"].to_numpy()
rows = []
for c in CH:
    x = rn(f[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float))
    m = np.isfinite(x)
    beta, *_ = np.linalg.lstsq(N[m], x[m], rcond=None)
    res = np.full(len(x), np.nan); res[m] = x[m] - N[m] @ beta
    r2 = 1 - np.var(res[m]) / np.var(x[m])
    w0 = m & istr & (yy == 0)
    r2_w = 1 - np.var(res[w0]) / np.var(x[w0])
    xr, rr = x[pos], res[pos]
    b = safe_bins(rr, 10); chi, _ = oe_test(y, p, b)
    rows.append((c, r2, r2_w, ks_nan(y, xr), ks_nan(y, rr), ks_nan(y, rr) - ks_nan(y, xr), chi, crossfit_dks(y, p, b)))
R = pl.DataFrame(rows, schema=["channel", "R2_nuisance_all", "R2_within_good", "KS_raw", "KS_resid", "dKS_univ", "chi2_df_vs_v8", "dKS_cf"], orient="row")
with pl.Config(tbl_rows=30, float_precision=4):
    print(R.sort("dKS_univ", descending=True))
R.write_csv(OUT / "95_nuisance_residuals.csv")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(R["channel"], R["dKS_univ"]); ax.set_xlabel("univariate KS change after removing the nuisance block")
plt.tight_layout(); plt.savefig(OUT / "95_nuisance_residuals.png", dpi=80)
print("saved", OUT / "95_nuisance_residuals.png")
