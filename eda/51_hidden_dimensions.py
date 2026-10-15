"""The price premium was a per-row attribute (unit price) that no aggregate measured. Sweep the other per-row
attributes the same way, against the v5 residual: intra-month balance dynamics, previous-application timing,
repayment payment timing, e-commerce clock/weekday/payment style, bureau reporting lags, selected-application
details. Every candidate is label-free, cutoff-filtered, and scored with a noise baseline."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks_nan, load, resid_test, v5_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v5_oof())
rng = np.random.default_rng(11)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise vs v5: chi2/df p95={CHI:.2f}  dKS p95={DKS:+.4f}")
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime")
H = lambda c: pl.col(c).dt.hour() + pl.col(c).dt.minute() / 60  # noqa: E731
parts = []

# 1) account: intra-month dynamics (balance path shape, not levels)
a = load("account_behavior").join(cut, on="customer_id").filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us")) <= pl.col("cutoff"))
a = a.with_columns(
    mid=(pl.col("opening_balance") + pl.col("closing_balance")) / 2,
    turnover=(pl.col("inflows") + pl.col("outflows")) / (pl.col("average_balance").abs() + 1e5),
).with_columns(
    path_skew=(pl.col("average_balance") - pl.col("mid")) / (pl.col("mid").abs() + 1e5),
    dip=(pl.col("average_balance") - pl.col("minimum_balance")) / (pl.col("average_balance").abs() + 1e5),
    low_given_min=pl.col("low_balance_days") * (pl.col("minimum_balance") > 250_000).cast(pl.Int8),
    nonincome=(pl.col("inflows") - pl.col("income_like_credits")) / (pl.col("inflows") + 1e5),
)
parts.append(a.group_by("customer_id").agg(
    hd_acc_path_skew=pl.col("path_skew").mean(), hd_acc_dip=pl.col("dip").mean(), hd_acc_dip_max=pl.col("dip").max(),
    hd_acc_turnover=pl.col("turnover").mean(), hd_acc_low_given_min=pl.col("low_given_min").mean(),
    hd_acc_nonincome_cv=pl.col("nonincome").std() / (pl.col("nonincome").mean().abs() + 1e-3),
    hd_acc_income_regular=(pl.col("income_like_credits") > 0).mean(),
    hd_acc_inflow_cv=pl.col("inflows").std() / (pl.col("inflows").mean() + 1e5)))

# 2) previous applications: decision lag, application clock, gaps between applications, requested-amount rounding
pv = load("previous_applications").join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff")).sort("customer_id", "application_datetime", "application_id")
pv = pv.with_columns(lag_h=(pl.col("decision_datetime") - pl.col("application_datetime")).dt.total_seconds() / 3600,
                     hour=H("application_datetime"), gap_d=pl.col("application_datetime").diff().over("customer_id").dt.total_seconds() / 86400,
                     round_m=(pl.col("requested_amount") % 1_000_000 == 0).cast(pl.Int8))
parts.append(pv.group_by("customer_id").agg(
    hd_pv_lag_mean=pl.col("lag_h").mean(), hd_pv_lag_max=pl.col("lag_h").max(), hd_pv_night=(pl.col("hour") < 6).mean(),
    hd_pv_hour_mean=pl.col("hour").mean(), hd_pv_gap_min=pl.col("gap_d").min(), hd_pv_gap_median=pl.col("gap_d").median(),
    hd_pv_round_share=pl.col("round_m").mean(), hd_pv_term_last=pl.col("term_months").last()))

# 3) repayment: when in the day/month payments land, partial-payment depth, payment vs due amount shape
rp = load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
rp = rp.with_columns(pay_h=H("last_payment_datetime"), pay_dom=pl.col("last_payment_datetime").dt.day(),
                     due_dom=pl.col("due_date").dt.day(),
                     partial_depth=pl.when(pl.col("amount_paid") < pl.col("amount_due")).then(pl.col("amount_paid") / pl.col("amount_due")),
                     same_clock=(pl.col("last_payment_datetime").dt.time() == pl.col("due_date").dt.time()).cast(pl.Int8))
parts.append(rp.group_by("customer_id").agg(
    hd_rp_pay_hour=pl.col("pay_h").mean(), hd_rp_pay_night=(pl.col("pay_h") < 6).mean(),
    hd_rp_partial_depth=pl.col("partial_depth").mean(), hd_rp_same_clock=pl.col("same_clock").mean(),
    hd_rp_due_dom=pl.col("due_dom").mean(), hd_rp_pay_late_dom=pl.col("pay_dom").mean()))

# 4) e-commerce style: clock, weekday, payment-method mix by category, cancellations by price, basket composition
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.with_columns(h=H("transaction_datetime"), wd=pl.col("transaction_datetime").dt.weekday(), mday=pl.col("transaction_datetime").dt.day(),
                   lu=(pl.col("item_total") / pl.col("quantity")).log(), done=pl.col("order_status") == "completed")
e = e.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"))
o = e.group_by("order_id").agg(pl.col("customer_id", "h", "wd", "mday", "payment_method", "done").first(), z=pl.col("z").mean(),
                               ncat=pl.col("product_category").n_unique(), n=pl.len())
parts.append(o.group_by("customer_id").agg(
    hd_ec_hour_mean=pl.col("h").mean(), hd_ec_hour_sd=pl.col("h").std(), hd_ec_weekend=(pl.col("wd") >= 6).mean(),
    hd_ec_payday=(pl.col("mday") >= 25).mean(), hd_ec_month_start=(pl.col("mday") <= 5).mean(),
    hd_ec_z_cancelled_minus_done=pl.col("z").filter(~pl.col("done")).mean() - pl.col("z").filter("done").mean(),
    hd_ec_cc_share=(pl.col("payment_method") == "credit_card").mean(), hd_ec_cod_share=(pl.col("payment_method") == "COD").mean(),
    hd_ec_z_credit=pl.col("z").filter(pl.col("payment_method") == "credit_card").mean(),
    hd_ec_z_cod=pl.col("z").filter(pl.col("payment_method") == "COD").mean(),
    hd_ec_multi_cat=(pl.col("ncat") > 1).mean(), hd_ec_items_sd=pl.col("n").std()))

# 5) bureau: reporting lag and staleness spread across providers at the selected pull, score vs provider gap
b = load("bureau_observations").join(apps.select("customer_id", sel="application_id", cutoff="decision_datetime"), on="customer_id").filter(pl.col("available_datetime") <= pl.col("cutoff"))
b = b.with_columns(delay=(pl.col("available_datetime") - pl.col("report_datetime")).dt.total_seconds() / 86400,
                   is_sel=pl.col("application_id") == pl.col("sel"))
parts.append(b.group_by("customer_id").agg(
    hd_bu_delay_mean=pl.col("delay").mean(), hd_bu_delay_sel_max=pl.col("delay").filter("is_sel").max(),
    hd_bu_score_null_share=pl.col("score").is_null().mean(),
    hd_bu_active_vs_inq=(pl.col("active_accounts") / (pl.col("inquiries_last_6m") + 1)).mean()))

# 6) selected application: clock, amount rounding, installment rounding residue
sa = apps.select("customer_id", hd_app_hour=H("application_datetime"), hd_app_dom=pl.col("application_datetime").dt.day(),
                 hd_app_round_m=(pl.col("loan_amount") % 1_000_000 == 0).cast(pl.Int8), hd_dec_hour=H("decision_datetime"))
parts.append(sa)

X = tr.select("customer_id")
for q in parts:
    X = X.join(q, on="customer_id", how="left", maintain_order="left")
rows = []
for c in X.columns[1:]:
    x = X[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    if np.nanstd(x) == 0:
        continue
    chi, lift, d = resid_test(y, p, x)
    rows.append((c, c.split("_")[1], ks_nan(y, x), chi, d, float(np.mean(~np.isnan(x)))))
R = pl.DataFrame(rows, schema=["feature", "table", "ks", "chi2_df", "dKS_cf", "coverage"], orient="row").sort("chi2_df", descending=True)
R = R.with_columns(above_noise=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS))
R.write_csv(OUT / "51_hidden_dimensions.csv")
with pl.Config(tbl_rows=60, float_precision=4):
    print(R)
print(f"\nabove noise on both tests: {R['above_noise'].sum()} of {R.height}")
fig, ax = plt.subplots(figsize=(8, 5))
for t_ in R["table"].unique().to_list():
    s = R.filter(pl.col("table") == t_)
    ax.scatter(s["chi2_df"], s["dKS_cf"], label=t_, s=20)
ax.axvline(CHI, color="r", ls="--"); ax.axhline(DKS, color="r", ls="--")
ax.set_xlabel("residual chi2/df vs v5"); ax.set_ylabel("cross-fit dKS vs v5"); ax.legend(); ax.set_title("hidden per-row dimensions vs the v5 blend")
plt.tight_layout(); plt.savefig(OUT / "51_hidden_dimensions.png", dpi=80)
