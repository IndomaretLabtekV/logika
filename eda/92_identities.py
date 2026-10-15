"""Arithmetic identities between columns, and whether their residuals carry signal ("revealing the unknowns").

When a column is a deterministic function of other columns plus a hidden term, the hidden term is a candidate
latent (the unit-price premium of v5 was one). For every identity: share of rows where it holds exactly, the
residual distribution, and, where a residual exists, the customer-level mean residual tested against the v8 score
(observed-row KS and the O/E score test with zero kept in its own bin)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common9 import OUT, crossfit_dks, load, oe_test, safe_bins, v8_ref
from common import ks

tr, y, p = v8_ref()
cut = pl.concat([load("train_applications").drop("target"), load("test_applications")]).select("customer_id", cutoff="decision_datetime")
bu, ec, rp = load("bureau_observations"), load("ecommerce_transactions"), load("repayment_history")
ac, pv = load("account_behavior"), load("previous_applications")
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])

checks = {}
# 1 utilization = revolving balance / limit
d = bu.filter(pl.col("revolving_credit_limit") > 0).with_columns(res=pl.col("utilization") - pl.col("revolving_outstanding_balance") / pl.col("revolving_credit_limit"))
checks["bureau utilization - bal/limit"] = d.select("customer_id", "res")
# 2 order_total = sum(item_total)
o = ec.group_by("order_id").agg(pl.col("customer_id").first(), t=pl.col("order_total").first(), s=pl.col("item_total").sum())
checks["order_total - sum(items)"] = o.select("customer_id", res=(pl.col("t") - pl.col("s")).cast(pl.Float64))
# 3 item_total multiple of quantity? (unit price integer rounding)
checks["item_total mod (1000*qty)"] = ec.select("customer_id", res=(pl.col("item_total") % (1000 * pl.col("quantity"))).cast(pl.Float64))
# 4 amount_due vs scheduled installment of its loan
rd = rp.join(pv.select("loan_id", "scheduled_monthly_installment", "term_months", "approved_amount", "annual_interest_rate"), on="loan_id", how="left")
rd = rd.with_columns(k=pl.col("due_date").rank("ordinal").over("loan_id"))
checks["amount_due - scheduled (non-final)"] = rd.filter(pl.col("k") < pl.col("term_months")).select(
    "customer_id", res=(pl.col("amount_due") - pl.col("scheduled_monthly_installment")).cast(pl.Float64))
checks["final installment - scheduled"] = rd.filter(pl.col("k") == pl.col("term_months")).select(
    "customer_id", res=(pl.col("amount_due") - pl.col("scheduled_monthly_installment")).cast(pl.Float64))
# 5 due dates = decision + k months (same clock time)?
rd2 = rd.join(pv.select("loan_id", dec="decision_datetime"), on="loan_id", how="left")
checks["due - (decision + k months) [h]"] = rd2.select(
    "customer_id", res=((pl.col("due_date") - pl.col("dec").dt.offset_by(pl.format("{}mo", pl.col("k")))).dt.total_seconds() / 3600))
# 6 paid installments: max_dpd vs lateness of last payment
paid = rp.filter((pl.col("amount_paid") >= pl.col("amount_due")) & pl.col("last_payment_datetime").is_not_null()).with_columns(
    late=(pl.col("last_payment_datetime") - pl.col("due_date")).dt.total_seconds() / 86400)
checks["paid: max_dpd - floor(late)"] = paid.select("customer_id", res=(pl.col("max_days_past_due") - pl.col("late").floor().clip(lower_bound=0)))
checks["paid: late days fractional part"] = paid.filter(pl.col("late") != 0).select("customer_id", res=pl.col("late") - pl.col("late").floor())
checks["paid: early days (late<0)"] = paid.select("customer_id", res=(-pl.col("late")).clip(lower_bound=0))
# 7 partial payments: amount_paid / amount_due
part = rp.filter((pl.col("amount_paid") > 0) & (pl.col("amount_paid") < pl.col("amount_due")))
checks["partial paid share"] = part.select("customer_id", res=pl.col("amount_paid") / pl.col("amount_due"))
# 8 account: average balance vs open/close midpoint, min balance vs min(open, close), low days vs min balance
checks["avg_bal - (open+close)/2"] = ac.select("customer_id", res=(pl.col("average_balance") - (pl.col("opening_balance") + pl.col("closing_balance")) / 2)
                                               / (pl.col("inflows") + 1))
checks["min(open,close) - min_bal"] = ac.select("customer_id", res=(pl.min_horizontal("opening_balance", "closing_balance") - pl.col("minimum_balance"))
                                                / (pl.col("inflows") + 1))
checks["low_days>0 while min_bal>=250k"] = ac.select("customer_id", res=((pl.col("low_balance_days") > 0) & (pl.col("minimum_balance") >= 250_000)).cast(pl.Float64))
checks["income_like > inflows"] = ac.select("customer_id", res=(pl.col("income_like_credits") > pl.col("inflows")).cast(pl.Float64))
# 9 previous: approved / requested; rounding of amounts
checks["approved / requested"] = pv.filter(pl.col("approved_amount").is_not_null()).select("customer_id", res=pl.col("approved_amount") / pl.col("requested_amount"))
checks["requested mod 100k"] = pv.select("customer_id", res=(pl.col("requested_amount") % 100_000).cast(pl.Float64))
checks["selected loan mod 100k"] = apps.select("customer_id", res=(pl.col("loan_amount") % 100_000).cast(pl.Float64))
checks["declared income mod 1000"] = load("customer_profiles").select("customer_id", res=(pl.col("declared_monthly_income") % 1000).cast(pl.Float64))
# 10 bureau: report -> available delay, active accounts vs outstanding
checks["bureau delay days"] = bu.select("customer_id", res=(pl.col("available_datetime") - pl.col("report_datetime")).dt.total_seconds() / 86400)
checks["outstanding>0 with 0 active"] = bu.select("customer_id", res=((pl.col("outstanding_balance") > 0) & (pl.col("active_accounts") == 0)).cast(pl.Float64))
checks["rev balance > outstanding"] = bu.select("customer_id", res=(pl.col("revolving_outstanding_balance") > pl.col("outstanding_balance")).cast(pl.Float64))

rows, ex = [], []
for name, d in checks.items():
    d = d.filter(pl.col("res").is_finite())
    r = d["res"].to_numpy()
    exact = float(np.mean(np.abs(r) < 1e-9)) if len(r) else np.nan
    cm = d.group_by("customer_id").agg(m=pl.col("res").mean())
    x = tr.select("customer_id").join(cm, on="customer_id", how="left", maintain_order="left")["m"].to_numpy().astype(float)
    obs = np.isfinite(x)
    kso = ks(y[obs], x[obs]) if obs.sum() > 500 and np.nanstd(x) > 0 else np.nan
    b = safe_bins(x)
    chi, _ = oe_test(y, p, b)
    rows.append((name, len(r), exact, float(np.quantile(r, 0.01)) if len(r) else np.nan, float(np.median(r)) if len(r) else np.nan,
                 float(np.quantile(r, 0.99)) if len(r) else np.nan, float(obs.mean()), kso, chi, crossfit_dks(y, p, b)))
    ex.append((name, r[:200000]))
R = pl.DataFrame(rows, schema=["identity", "rows", "exact_share", "p01", "median", "p99", "cust_cov", "KS_obs", "chi2_df_vs_v8", "dKS_cf"], orient="row")
with pl.Config(tbl_rows=40, fmt_str_lengths=40, tbl_width_chars=220, float_precision=4):
    print("chi2/df vs v8: ~1 = nothing left beyond v8 (p95 about 1.9 for 8 bins); dKS_cf: cross-fitted KS gain")
    print(R)
R.write_csv(OUT / "92_identities.csv")
fig, axs = plt.subplots(5, 5, figsize=(20, 14))
for ax, (n, r) in zip(axs.ravel(), ex):
    r = r[np.isfinite(r)]
    if len(r):
        lo, hi = np.quantile(r, [0.005, 0.995])
        ax.hist(np.clip(r, lo, hi), 60)
    ax.set_title(n, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "92_identities.png", dpi=70)
print("saved", OUT / "92_identities.png")
