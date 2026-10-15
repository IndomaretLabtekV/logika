"""Hypothesis: the simulated lender priced / approved past and current applications with information we do not see
(the customer's latent risk). Then the part of each decision NOT explained by the bureau pull linked to that same
application is a measurement of hidden risk:
  rate premium   = rate tier - E[rate | score bin of that application's own bureau pull]
  reject premium = rejected  - P(rejected | score bin, requested/income bin)
  cut premium    = approved/requested - E[... | score bin]
All references are label-free (fit on train+test applications). Tested against the v6 OOF blend with a noise baseline."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks_nan, load, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v6_oof())
rng = np.random.default_rng(5)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise vs v6: chi2/df p95={CHI:.2f}  dKS p95={DKS:+.4f}")

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
prof = load("customer_profiles").select("customer_id", inc="declared_monthly_income")
bu = load("bureau_observations")
pull = bu.group_by("application_id").agg(score=pl.col("score").mean(), util=pl.col("utilization").mean(),
                                         inq=pl.col("inquiries_last_6m").mean(), delq=pl.col("delinquent_accounts_last_12m").max())
pull = pull.with_columns(sbin=(pl.col("score") // 20).fill_null(-1))

# 1) what sets the rate of the selected application? (mapping strength)
sel = apps.join(pull, on="application_id", how="left").join(prof, on="customer_id")
g = sel.group_by("sbin").agg(n=pl.len(), rate_mean=pl.col("annual_interest_rate").mean(), rate_sd=pl.col("annual_interest_rate").std(),
                            n_tiers=pl.col("annual_interest_rate").n_unique()).sort("sbin")
with pl.Config(tbl_rows=40, float_precision=4):
    print("selected applications: rate by own-pull score bin\n", g)
tot = sel["annual_interest_rate"].var(); within = sel.with_columns(r=pl.col("annual_interest_rate") - pl.col("annual_interest_rate").mean().over("sbin"))["r"].var()
print(f"selected rate variance explained by own-pull score bin: {1 - within / tot:.3f}")
for extra in (["sbin", "delq"], ["sbin", "delq", "inq"], ["sbin", "delq", "loan_purpose"]):
    w = sel.with_columns(r=pl.col("annual_interest_rate") - pl.col("annual_interest_rate").mean().over(extra))["r"].var()
    print(f"  ... explained by {extra}: {1 - w / tot:.3f}")

# 2) premiums on the selected application and on past applications
sel = sel.with_columns(rate_prem=pl.col("annual_interest_rate") - pl.col("annual_interest_rate").mean().over("sbin", "delq"))
pv = load("previous_applications").join(pull, on="application_id", how="left").join(prof, on="customer_id")
pv = pv.with_columns(rbin=(pl.col("requested_amount") / pl.col("inc")).qcut(5, labels=[str(i) for i in range(5)], allow_duplicates=True),
                     rej=(pl.col("decision") == "rejected").cast(pl.Float64), cut_=pl.col("approved_amount") / pl.col("requested_amount"))
pv = pv.with_columns(rate_prem=pl.col("annual_interest_rate") - pl.col("annual_interest_rate").mean().over("sbin", "delq"),
                     rej_prem=pl.col("rej") - pl.col("rej").mean().over("sbin", "rbin"),
                     cut_prem=pl.col("cut_") - pl.col("cut_").mean().over("sbin"))
print("past applications with a linked bureau pull:", pv["score"].is_not_null().mean())
agg = pv.group_by("customer_id").agg(la_pv_rate_prem=pl.col("rate_prem").mean(), la_pv_rate_prem_last=pl.col("rate_prem").sort_by("decision_datetime", "application_id").drop_nulls().last(),
                                     la_pv_rej_prem=pl.col("rej_prem").mean(), la_pv_rej_prem_sum=pl.col("rej_prem").sum(),
                                     la_pv_cut_prem=pl.col("cut_prem").mean())
X = tr.select("customer_id").join(sel.select("customer_id", la_sel_rate_prem="rate_prem"), on="customer_id", how="left", maintain_order="left")
X = X.join(agg, on="customer_id", how="left", maintain_order="left")
rows = []
for c in X.columns[1:]:
    x = X[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    chi, lift, d = resid_test(y, p, x)
    rows.append((c, ks_nan(y, x), chi, d, float(np.mean(~np.isnan(x)))))
R = pl.DataFrame(rows, schema=["feature", "ks", "chi2_df", "dKS_cf", "coverage"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS))
with pl.Config(float_precision=4):
    print(R)
R.write_csv(OUT / "63_lender_assessment.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
gg = g.filter(pl.col("sbin") >= 0)
ax[0].errorbar(gg["sbin"] * 20, gg["rate_mean"], gg["rate_sd"], fmt="o-"); ax[0].set_title("selected rate vs own-pull score")
t = tr.select("target").with_columns(x=X["la_sel_rate_prem"]).drop_nulls()
q = t.with_columns(b=pl.col("x").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True)).group_by("b").agg(r=pl.col("target").mean()).sort("b")
ax[1].bar(q["b"].cast(str), q["r"]); ax[1].set_title("bad rate by selected rate-premium decile")
ax[2].barh(R["feature"], R["dKS_cf"]); ax[2].axvline(DKS, color="r", ls="--"); ax[2].set_title("dKS vs v6")
plt.tight_layout(); plt.savefig(OUT / "63_lender_assessment.png", dpi=80)
