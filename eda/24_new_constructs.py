"""New constructs that v1 does not have: active debt burden from previous_applications (incl. loans approved but not yet
due), loan stacking, lateness sequence (streaks, trend, first-payment, latest loan), partial payments, free cash flow
vs installment. Screened vs v1 residual."""
import numpy as np
import polars as pl
from common import load, screen

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime", inst_new="proposed_monthly_installment",
                  amt_new="loan_amount", rate_new="annual_interest_rate")
prof = load("customer_profiles").select("customer_id", "declared_monthly_income")
prev, rep, acc = load("previous_applications"), load("repayment_history"), load("account_behavior")

# (a) active obligations straight from previous_applications
p = prev.join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff"))
p = p.with_columns(m_el=((pl.col("cutoff") - pl.col("decision_datetime")).dt.total_days() / 30.44),
                   age_d=(pl.col("cutoff") - pl.col("decision_datetime")).dt.total_days())
ap = p.filter(pl.col("decision") == "approved").with_columns(
    rem=(pl.col("term_months") - pl.col("m_el").floor()).clip(lower_bound=0))
A = ap.group_by("customer_id").agg(
    ob_n_active=(pl.col("rem") > 0).sum(),
    ob_inst_active=pl.col("scheduled_monthly_installment").filter(pl.col("rem") > 0).sum(),
    ob_remaining=(pl.col("scheduled_monthly_installment") * pl.col("rem")).sum(),
    ob_days_since_appr=pl.col("age_d").min(),
    ob_n_appr_30d=(pl.col("age_d") <= 30).sum(), ob_n_appr_90d=(pl.col("age_d") <= 90).sum(),
    ob_amt_90d=pl.col("approved_amount").filter(pl.col("age_d") <= 90).sum(),
    ob_n_appr_365d=(pl.col("age_d") <= 365).sum(),
    ob_term_mean=pl.col("term_months").mean(),
)
P = p.group_by("customer_id").agg(
    ob_apps_30d=(pl.col("age_d") <= 30).sum(),
    ob_canc_365d=((pl.col("decision") == "cancelled") & (pl.col("age_d") <= 365)).sum(),
    ob_req_trend=pl.col("requested_amount").sort_by("application_datetime").last() / pl.col("requested_amount").mean(),
)
A = P.join(A, on="customer_id", how="left").join(cut, on="customer_id").join(prof, on="customer_id")

# (b) repayment sequence
r = rep.join(cut.select("customer_id", "cutoff"), on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
r = r.sort("customer_id", "due_date").with_columns(
    late=(pl.col("max_days_past_due") > 0).cast(pl.Int32),
    partial=((pl.col("amount_paid") > 0) & (pl.col("amount_paid") < pl.col("amount_due"))).cast(pl.Int32),
    k=pl.col("due_date").rank("ordinal").over("loan_id"),
    age_d=(pl.col("cutoff") - pl.col("due_date")).dt.total_days(),
    loan_start=pl.col("due_date").min().over("loan_id"))
r = r.with_columns(streak_id=(pl.col("late") != pl.col("late").shift(1)).fill_null(True).cum_sum().over("customer_id"))
streak = r.filter(pl.col("late") == 1).group_by("customer_id", "streak_id").agg(L=pl.len()).group_by("customer_id").agg(rq_streak_max=pl.col("L").max())
last_loan = r.filter(pl.col("loan_start") == pl.col("loan_start").max().over("customer_id"))
LL = last_loan.group_by("customer_id").agg(rq_lastloan_late=pl.col("late").mean(), rq_lastloan_maxdpd=pl.col("max_days_past_due").max(),
                                            rq_lastloan_n=pl.len())
R = r.group_by("customer_id").agg(
    rq_partial_frac=pl.col("partial").mean(), rq_partial_n=pl.col("partial").sum(),
    rq_first_inst_late=pl.col("late").filter(pl.col("k") == 1).mean(),
    rq_first_inst_dpd=pl.col("max_days_past_due").filter(pl.col("k") == 1).mean(),
    rq_late_last6=pl.col("late").tail(6).mean(), rq_late_prior=pl.col("late").head(pl.len() - 6).mean(),
    rq_cur_streak=(pl.col("late").reverse().cum_min()).sum(),
    rq_dpd_p75=pl.col("max_days_past_due").quantile(0.75), rq_dpd_median=pl.col("max_days_past_due").median(),
    rq_frac_1_29=((pl.col("max_days_past_due") > 0) & (pl.col("max_days_past_due") < 30)).mean(),
    rq_amt_wtd_late=(pl.col("late") * pl.col("amount_due")).sum() / pl.col("amount_due").sum(),
    rq_dpd_sum=pl.col("max_days_past_due").sum(),
    rq_dpd_log_mean=(pl.col("max_days_past_due") + 1).log().mean(),
    rq_unpaid_recent=((pl.col("amount_paid") < pl.col("amount_due")) & (pl.col("age_d") <= 60)).sum(),
    rq_n_inst=pl.len(),
).with_columns(rq_late_trend=pl.col("rq_late_last6") - pl.col("rq_late_prior"))
R = R.join(streak, on="customer_id", how="left").join(LL, on="customer_id", how="left")
# Bayesian-shrunk late rate: (late + a)/(n + a + b) with population prior (unsupervised: prior from all customers)
pr = r["late"].mean()
R = R.join(r.group_by("customer_id").agg(nl=pl.col("late").sum(), n=pl.len()), on="customer_id").with_columns(
    rq_late_shrunk=(pl.col("nl") + 4 * pr) / (pl.col("n") + 4)).drop("nl", "n")

# (c) account cash-flow vs the new installment
a = acc.join(cut, on="customer_id").filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff"))
a = a.with_columns(mago=pl.col("month").rank("dense", descending=True).over("customer_id"),
                   free=pl.col("inflows") - pl.col("outflows"))
C = a.group_by("customer_id").agg(
    cf_free_mean=pl.col("free").mean(), cf_free_l3=pl.col("free").filter(pl.col("mago") <= 3).mean(),
    cf_free_min=pl.col("free").min(),
    cf_close_l1=pl.col("closing_balance").filter(pl.col("mago") == 1).first(),
    cf_minbal_l3=pl.col("minimum_balance").filter(pl.col("mago") <= 3).min(),
    cf_out_l3=pl.col("outflows").filter(pl.col("mago") <= 3).mean(),
    cf_inc_l3=pl.col("income_like_credits").filter(pl.col("mago") <= 3).mean(),
    cf_nonincome_share=1 - pl.col("income_like_credits").sum() / pl.col("inflows").sum(),
    cf_nonincome_l3=(1 - pl.col("income_like_credits").filter(pl.col("mago") <= 3).sum()
                     / pl.col("inflows").filter(pl.col("mago") <= 3).sum()),
    cf_low_months_l6=(pl.col("low_balance_days").filter(pl.col("mago") <= 6) > 0).sum(),
    cf_lowdays_l1=pl.col("low_balance_days").filter(pl.col("mago") == 1).first(),
    cf_n_months=pl.len(),
    inst_new=pl.col("inst_new").first(),
)
C = C.with_columns(
    cf_inst_to_free=pl.col("inst_new") / (pl.col("cf_free_l3").clip(lower_bound=0) + 1e5),
    cf_inst_to_inc_l3=pl.col("inst_new") / (pl.col("cf_inc_l3") + 1e5),
    cf_inst_to_close=pl.col("inst_new") / (pl.col("cf_close_l1") + 1e5),
    cf_inst_to_minbal=pl.col("inst_new") / (pl.col("cf_minbal_l3") + 1e5),
    cf_out_to_inc_l3=pl.col("cf_out_l3") / (pl.col("cf_inc_l3") + 1e5),
).drop("inst_new")

# combine + debt-burden ratios (needs income: declared, and verified from account)
X = A.join(R, on="customer_id", how="full", coalesce=True).join(C, on="customer_id", how="full", coalesce=True)
X = X.join(cut, on="customer_id", how="right").join(prof, on="customer_id", how="left", suffix="_p")
X = X.with_columns(
    ob_burden_decl=(pl.col("ob_inst_active").fill_null(0) + pl.col("inst_new")) / pl.col("declared_monthly_income_p"),
    ob_burden_verif=(pl.col("ob_inst_active").fill_null(0) + pl.col("inst_new")) / (pl.col("cf_inc_l3") + 1e5),
    ob_amt_vs_last=pl.col("amt_new") / (pl.col("ob_amt_90d") + 1e6),
)
X = X.select("customer_id", pl.col("^(ob|rq|cf)_.*$"))
screen(X, "24_constructs", show=60)
