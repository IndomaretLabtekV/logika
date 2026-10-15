"""Bureau scores as repeated noisy measurements of one latent risk: provider bias/scale, cross-provider and
over-time correlation, and denoised combinations (provider z-score, all-pull average, trend). Screened vs v1."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, load, screen

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
prev = load("previous_applications")
b = load("bureau_observations")
cut = apps.select("customer_id", sel_app="application_id", cutoff="decision_datetime")
b = b.join(cut, on="customer_id").filter(pl.col("available_datetime") <= pl.col("cutoff"))
b = b.with_columns(is_sel=pl.col("application_id") == pl.col("sel_app"),
                   age_d=(pl.col("cutoff") - pl.col("report_datetime")).dt.total_seconds() / 86400,
                   delay_d=(pl.col("available_datetime") - pl.col("report_datetime")).dt.total_seconds() / 86400)

# 1) which application does each bureau row belong to?
known = set(prev["application_id"]) | set(apps["application_id"])
print("bureau rows whose application_id is neither selected nor previous:", (~b["application_id"].is_in(list(known))).sum())
print("rows per (customer, application) pull:", b.group_by("customer_id", "application_id").len()["len"].value_counts().sort("len"))

# 2) provider profile
print("\n=== provider profile (all leak-safe rows)")
print(b.group_by("bureau_provider").agg(n=pl.len(), score_null=pl.col("score").is_null().mean(), mean=pl.col("score").mean(),
                                         sd=pl.col("score").std(), delay_med=pl.col("delay_d").median(),
                                         age_med=pl.col("age_d").median()).sort("bureau_provider"))

# 3) cross-provider correlation in the selected pull, and current vs historical
cur = b.filter("is_sel").pivot(on="bureau_provider", index="customer_id", values="score", aggregate_function="first")
print("\n=== Pearson corr of provider scores (selected pull)")
print(cur.drop("customer_id").corr())
hist = b.filter(~pl.col("is_sel"))
pair = (b.filter("is_sel").select("customer_id", "bureau_provider", s_cur="score")
        .join(hist.select("customer_id", "bureau_provider", s_h="score", gap="age_d"), on=["customer_id", "bureau_provider"])
        .drop_nulls())
pair = pair.with_columns(gap_b=(pl.col("gap") // 180 * 180).clip(upper_bound=900))
print("\n=== corr(current score, historical score same provider) by age of the historical report (days)")
print(pair.group_by("gap_b").agg(n=pl.len(), rho=pl.corr("s_cur", "s_h")).sort("gap_b"))

# 4) latent estimates. z-score per provider uses pooled train+test rows (unsupervised, no target)
st = b.group_by("bureau_provider").agg(mu=pl.col("score").mean(), sd=pl.col("score").std())
b = b.join(st, on="bureau_provider").with_columns(z=(pl.col("score") - pl.col("mu")) / pl.col("sd"))
w_decay = (-pl.col("age_d") / 365).exp()
lat = b.group_by("customer_id").agg(
    bz_cur_mean=pl.col("z").filter("is_sel").mean(),
    bz_cur_min=pl.col("z").filter("is_sel").min(),
    bz_all_mean=pl.col("z").mean(),
    bz_all_n=pl.col("z").is_not_null().sum(),
    bz_wmean=(pl.col("z") * w_decay).sum() / (w_decay * pl.col("z").is_not_null()).sum(),
    bz_hist_mean=pl.col("z").filter(~pl.col("is_sel")).mean(),
    bz_newest=pl.col("z").sort_by("report_datetime").drop_nulls().last(),
    bz_trend=pl.cov(pl.col("age_d"), pl.col("z")) / pl.col("age_d").var() * -365,
    bz_range=pl.col("z").max() - pl.col("z").min(),
    bz_cur_spread=pl.col("z").filter("is_sel").max() - pl.col("z").filter("is_sel").min(),
    b_unscored_frac=pl.col("score").is_null().mean(),
    b_cur_unscored=pl.col("score").filter("is_sel").is_null().sum(),
    b_cur_age_min=pl.col("age_d").filter("is_sel").min(),
    b_active_cur=pl.col("active_accounts").filter("is_sel").mean(),
    b_active_trend=pl.col("active_accounts").filter("is_sel").mean() - pl.col("active_accounts").filter(~pl.col("is_sel")).mean(),
    b_delinq_cur_mean=pl.col("delinquent_accounts_last_12m").filter("is_sel").mean(),
    b_delinq_all_mean=pl.col("delinquent_accounts_last_12m").mean(),
    b_inq_cur_mean=pl.col("inquiries_last_6m").filter("is_sel").mean(),
    b_inq_cur_max=pl.col("inquiries_last_6m").filter("is_sel").max(),
    b_util_cur_mean=pl.col("utilization").filter("is_sel").mean(),
    b_revlim_cur_mean=pl.col("revolving_credit_limit").filter("is_sel").mean(),
    b_out_cur_mean=pl.col("outstanding_balance").filter("is_sel").mean(),
    b_out_cur_min=pl.col("outstanding_balance").filter("is_sel").min(),
    b_out_growth=pl.col("outstanding_balance").filter("is_sel").mean()
    / (pl.col("outstanding_balance").filter(~pl.col("is_sel")).mean() + 1e6),
)
# inquiries not explained by own applications in the 6m window -> applications elsewhere
own6 = (prev.join(cut, on="customer_id").filter(pl.col("application_datetime") <= pl.col("cutoff"))
        .filter((pl.col("cutoff") - pl.col("application_datetime")).dt.total_days() <= 183)
        .group_by("customer_id").agg(own_apps_6m=pl.len()))
lat = lat.join(own6, on="customer_id", how="left").with_columns(
    pl.col("own_apps_6m").fill_null(0),
).with_columns(b_inq_external=pl.col("b_inq_cur_mean") - pl.col("own_apps_6m"))
r = screen(lat, "21_bureau")

# figure: correlation decay + bad rate by denoised score vs raw mean
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
g = pair.group_by("gap_b").agg(rho=pl.corr("s_cur", "s_h")).sort("gap_b")
ax[0].plot(g["gap_b"], g["rho"], "o-"); ax[0].set_xlabel("age of historical report (d)"); ax[0].set_title("corr(current, past score)")
ax[1].barh(r["feature"].head(15).to_list()[::-1], r["chi2_df"].head(15).to_list()[::-1])
ax[1].axvline(1.2, color="r", ls="--"); ax[1].set_title("residual chi2/df vs v1 (red = noise p95)")
plt.tight_layout(); plt.savefig(OUT / "21_bureau.png", dpi=80)
