"""Leak-safe, customer-level feature builders (one selected application per customer).

All auxiliary tables are already truncated at decision_datetime (verified in 01_keys_target_time.py),
but every builder still filters on the cutoff so the code stays correct if that ever changes.
"""
import polars as pl

DAY = 86400.0


def _days(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds() / DAY


def app_features(apps: pl.DataFrame, prof: pl.DataFrame) -> pl.DataFrame:
    d = apps.join(prof, on="customer_id", how="left")
    return d.select(
        "customer_id",
        "loan_amount", "annual_interest_rate", "proposed_monthly_installment", "loan_purpose",
        "age", "region", "employment_type", "employment_tenure_months", "declared_monthly_income",
        "dependents", "housing_status",
        decision_lag_h=(pl.col("decision_datetime") - pl.col("application_datetime")).dt.total_seconds() / 3600,
        app_hour=pl.col("application_datetime").dt.hour(),
        app_dow=pl.col("application_datetime").dt.weekday(),
        dti_declared=pl.col("proposed_monthly_installment") / pl.col("declared_monthly_income"),
        lti_declared=pl.col("loan_amount") / pl.col("declared_monthly_income"),
        tenure_over_age=pl.col("employment_tenure_months") / (pl.col("age") * 12),
        income_per_head=pl.col("declared_monthly_income") / (1 + pl.col("dependents")),
    )


def bureau_features(apps: pl.DataFrame, bureau: pl.DataFrame) -> pl.DataFrame:
    cut = apps.select("customer_id", sel_app=pl.col("application_id"), cutoff=pl.col("decision_datetime"))
    b = bureau.join(cut, on="customer_id").filter(pl.col("available_datetime") <= pl.col("cutoff"))
    b = b.with_columns(
        stale_d=_days("cutoff", "report_datetime"),
        delay_d=_days("available_datetime", "report_datetime"),
        is_sel=pl.col("application_id") == pl.col("sel_app"),
        prov=pl.col("bureau_provider").str.replace("Bureau_", "").str.to_lowercase(),
    )
    cur = b.filter("is_sel")
    num = ["score", "active_accounts", "outstanding_balance", "revolving_credit_limit",
           "revolving_outstanding_balance", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"]
    agg = cur.group_by("customer_id").agg(
        bc_n_reports=pl.len(),
        bc_n_scored=pl.col("score").is_not_null().sum(),
        bc_score_mean=pl.col("score").mean(), bc_score_min=pl.col("score").min(),
        bc_score_max=pl.col("score").max(), bc_score_std=pl.col("score").std(),
        bc_active_max=pl.col("active_accounts").max(),
        bc_outstanding_max=pl.col("outstanding_balance").max(),
        bc_revlimit_max=pl.col("revolving_credit_limit").max(),
        bc_revbal_max=pl.col("revolving_outstanding_balance").max(),
        bc_util_max=pl.col("utilization").max(), bc_util_mean=pl.col("utilization").mean(),
        bc_inq_max=pl.col("inquiries_last_6m").max(), bc_inq_mean=pl.col("inquiries_last_6m").mean(),
        bc_delinq_max=pl.col("delinquent_accounts_last_12m").max(),
        bc_delinq_sum=pl.col("delinquent_accounts_last_12m").sum(),
        bc_stale_mean=pl.col("stale_d").mean(), bc_stale_max=pl.col("stale_d").max(),
        bc_delay_mean=pl.col("delay_d").mean(),
    )
    piv = cur.pivot(on="prov", index="customer_id", values=["score", "utilization", "inquiries_last_6m",
                                                            "delinquent_accounts_last_12m", "outstanding_balance"],
                    aggregate_function="first")
    piv = piv.rename({c: "bp_" + c.replace("inquiries_last_6m", "inq").replace("delinquent_accounts_last_12m", "delinq")
                      .replace("outstanding_balance", "out").replace("utilization", "util")
                      for c in piv.columns if c != "customer_id"})
    # history: bureau pulled at earlier (previous) applications
    hist = b.filter(~pl.col("is_sel")).group_by("customer_id").agg(
        bh_n_pulls=pl.col("application_id").n_unique(),
        bh_score_mean=pl.col("score").mean(), bh_score_min=pl.col("score").min(),
        bh_delinq_max=pl.col("delinquent_accounts_last_12m").max(),
        bh_inq_mean=pl.col("inquiries_last_6m").mean(),
        bh_last_score=pl.col("score").sort_by("report_datetime", "bureau_observation_id").drop_nulls().last(),
        bh_last_outstanding=pl.col("outstanding_balance").sort_by("report_datetime", "bureau_observation_id").last(),
        bh_pulls_180d=pl.col("application_id").filter(pl.col("stale_d") <= 180).n_unique(),
    )
    out = agg.join(piv, on="customer_id", how="left").join(hist, on="customer_id", how="left")
    return out.with_columns(
        bh_score_delta=pl.col("bc_score_mean") - pl.col("bh_last_score"),
        bh_outstanding_delta=pl.col("bc_outstanding_max") - pl.col("bh_last_outstanding"),
    )


def repayment_features(apps: pl.DataFrame, rep: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"))
    r = rep.join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
    r = r.with_columns(
        age_d=_days("cutoff", "due_date"),
        unpaid=pl.col("amount_due") - pl.col("amount_paid"),
        late_d=pl.when(pl.col("amount_paid") >= pl.col("amount_due")).then(_days("last_payment_datetime", "due_date")),
    )
    w = lambda days: pl.col("age_d") <= days  # noqa: E731
    agg = r.group_by("customer_id").agg(
        rp_n_loans=pl.col("loan_id").n_unique(), rp_n_inst=pl.len(),
        rp_maxdpd_max=pl.col("max_days_past_due").max(), rp_maxdpd_mean=pl.col("max_days_past_due").mean(),
        rp_n_dpd1=(pl.col("max_days_past_due") > 0).sum(), rp_n_dpd30=(pl.col("max_days_past_due") >= 30).sum(),
        rp_n_dpd60=(pl.col("max_days_past_due") >= 60).sum(), rp_n_dpd90=(pl.col("max_days_past_due") >= 90).sum(),
        rp_frac_dpd1=(pl.col("max_days_past_due") > 0).mean(), rp_frac_dpd30=(pl.col("max_days_past_due") >= 30).mean(),
        rp_cur_dpd_max=pl.col("days_past_due").max(), rp_n_unpaid=(pl.col("unpaid") > 0).sum(),
        rp_unpaid_amt=pl.col("unpaid").sum(),
        rp_paid_ratio=pl.col("amount_paid").sum() / pl.col("amount_due").sum(),
        rp_late_mean=pl.col("late_d").mean(), rp_late_max=pl.col("late_d").max(),
        rp_maxdpd_6m=pl.col("max_days_past_due").filter(w(183)).max(),
        rp_maxdpd_12m=pl.col("max_days_past_due").filter(w(365)).max(),
        rp_frac_dpd1_6m=(pl.col("max_days_past_due") > 0).filter(w(183)).mean(),
        rp_late_mean_6m=pl.col("late_d").filter(w(183)).mean(),
        rp_maxdpd_last3=pl.col("max_days_past_due").sort_by("due_date", "loan_id").tail(3).max(),
        rp_days_since_delinq=pl.col("age_d").filter(pl.col("max_days_past_due") > 0).min(),
        rp_days_since_due=pl.col("age_d").min(),
    )
    # active loans: fewer installments observed than contracted term -> remaining obligation
    loans = r.group_by("loan_id").agg(customer_id=pl.col("customer_id").first(), n_seen=pl.len(),
                                      inst=pl.col("amount_due").first())
    loans = loans.join(prev.select("loan_id", "term_months"), on="loan_id", how="left").with_columns(
        remaining=(pl.col("term_months") - pl.col("n_seen")).clip(lower_bound=0))
    act = loans.group_by("customer_id").agg(
        rp_n_active=(pl.col("remaining") > 0).sum(),
        rp_active_inst=pl.col("inst").filter(pl.col("remaining") > 0).sum(),
        rp_remaining_oblig=(pl.col("remaining") * pl.col("inst")).sum(),
    )
    return agg.join(act, on="customer_id", how="left")


def account_features(apps: pl.DataFrame, acc: pl.DataFrame) -> pl.DataFrame:
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"))
    a = acc.join(cut, on="customer_id").filter(
        pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff")).sort("customer_id", "month")
    a = a.with_columns(mago=pl.col("month").rank("dense", descending=True).over("customer_id"))
    l3 = pl.col("mago") <= 3
    out = a.group_by("customer_id").agg(
        ac_n_months=pl.len(),
        ac_inflow_mean=pl.col("inflows").mean(), ac_outflow_mean=pl.col("outflows").mean(),
        ac_income_mean=pl.col("income_like_credits").mean(), ac_income_std=pl.col("income_like_credits").std(),
        ac_income_min=pl.col("income_like_credits").min(),
        ac_avgbal_mean=pl.col("average_balance").mean(), ac_minbal_min=pl.col("minimum_balance").min(),
        ac_minbal_mean=pl.col("minimum_balance").mean(),
        ac_lowdays_mean=pl.col("low_balance_days").mean(), ac_lowdays_max=pl.col("low_balance_days").max(),
        ac_frac_zero_min=(pl.col("minimum_balance") == 0).mean(),
        ac_net_flow_mean=(pl.col("inflows") - pl.col("outflows")).mean(),
        ac_out_in_ratio=pl.col("outflows").sum() / pl.col("inflows").sum(),
        ac_income_l3=pl.col("income_like_credits").filter(l3).mean(),
        ac_avgbal_l3=pl.col("average_balance").filter(l3).mean(),
        ac_lowdays_l3=pl.col("low_balance_days").filter(l3).mean(),
        ac_out_in_l3=pl.col("outflows").filter(l3).sum() / pl.col("inflows").filter(l3).sum(),
        ac_last_close=pl.col("closing_balance").last(),
        ac_out_in_b=pl.col("outflows").filter(pl.col("mago").is_between(7, 12)).sum()
        / pl.col("inflows").filter(pl.col("mago").is_between(7, 12)).sum(),
        ac_lowdays_b=pl.col("low_balance_days").filter(pl.col("mago").is_between(7, 12)).mean(),
        ac_avgbal_b=pl.col("average_balance").filter(pl.col("mago").is_between(7, 12)).mean(),
        ac_out_in_l1=(pl.col("outflows") / pl.col("inflows")).filter(pl.col("mago") == 1).first(),
        ac_close_slope=pl.cov(pl.col("mago").cast(pl.Float64), pl.col("closing_balance").cast(pl.Float64))
        / pl.col("mago").cast(pl.Float64).var() * -1,
    )
    return out.with_columns(
        ac_income_cv=pl.col("ac_income_std") / pl.col("ac_income_mean"),
        ac_income_trend=pl.col("ac_income_l3") / pl.col("ac_income_mean"),
        ac_bal_trend=pl.col("ac_avgbal_l3") / (pl.col("ac_avgbal_mean") + 1),
        ac_out_in_delta=pl.col("ac_out_in_l3") - pl.col("ac_out_in_b"),
        ac_lowdays_delta=pl.col("ac_lowdays_l3") - pl.col("ac_lowdays_b"),
        ac_bal_delta=(pl.col("ac_avgbal_l3") + 1) / (pl.col("ac_avgbal_b") + 1),
    )


def ecommerce_features(apps: pl.DataFrame, ec: pl.DataFrame) -> pl.DataFrame:
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"))
    e = ec.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    e = e.with_columns(age_d=_days("cutoff", "transaction_datetime"), done=pl.col("order_status") == "completed")
    # order level: order_total repeats on each item -> dedupe before aggregating
    o = e.group_by("order_id").agg(pl.col("customer_id", "age_d", "order_total", "payment_method", "done").first())
    agg = o.group_by("customer_id").agg(
        ec_n_orders=pl.len(), ec_cancel_rate=1 - pl.col("done").mean(),
        ec_spend=pl.col("order_total").filter("done").sum(),
        ec_aov=pl.col("order_total").filter("done").mean(),
        ec_recency_d=pl.col("age_d").min(), ec_tenure_d=pl.col("age_d").max(),
        ec_orders_7d=(pl.col("age_d") <= 7).sum(), ec_orders_14d=(pl.col("age_d") <= 14).sum(),
        ec_orders_30d=(pl.col("age_d") <= 30).sum(), ec_orders_60d=(pl.col("age_d") <= 60).sum(),
        ec_orders_90d=(pl.col("age_d") <= 90).sum(),
        ec_base_wk=((pl.col("age_d") > 120) & (pl.col("age_d") <= 365)).sum()
        / ((pl.col("age_d").max().clip(upper_bound=365) - 120).clip(lower_bound=7) / 7),
        ec_spend_30d=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") <= 30)).sum(),
        ec_spend_90d=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") <= 90)).sum(),
        ec_cancel_90d=(~pl.col("done")).filter(pl.col("age_d") <= 90).sum(),
        **{f"ec_pm_{m}": (pl.col("payment_method") == m).mean()
           for m in ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]},
    )
    cat = (e.filter("done").group_by("customer_id", "product_category").agg(s=pl.col("item_total").sum())
           .with_columns(s=pl.col("s") / pl.col("s").sum().over("customer_id"))
           .pivot(on="product_category", index="customer_id", values="s"))
    cat = cat.rename({c: f"ec_cat_{c}" for c in cat.columns if c != "customer_id"})
    out = agg.join(cat, on="customer_id", how="left").with_columns(
        ec_spend_per_month=pl.col("ec_spend") / (pl.col("ec_tenure_d") / 30.4 + 1),
        ec_spend_90d_ratio=pl.col("ec_spend_90d") / 3 / (pl.col("ec_spend") / (pl.col("ec_tenure_d") / 30.4 + 1) + 1),
    )
    return out.with_columns(
        pl.col("^ec_cat_.*$").fill_null(0),
        ec_accel_30d=(pl.col("ec_orders_30d") / (30 / 7)) / (pl.col("ec_base_wk") + 0.1),
        ec_accel_90d=(pl.col("ec_orders_90d") / (90 / 7)) / (pl.col("ec_base_wk") + 0.1),
    )


def prev_features(apps: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"), cur_rate=pl.col("annual_interest_rate"),
                      cur_amt=pl.col("loan_amount"))
    p = prev.join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff"))
    p = p.with_columns(age_d=_days("cutoff", "application_datetime"))
    out = p.group_by("customer_id").agg(
        pv_n=pl.len(), pv_n_appr=(pl.col("decision") == "approved").sum(),
        pv_n_rej=(pl.col("decision") == "rejected").sum(), pv_n_canc=(pl.col("decision") == "cancelled").sum(),
        pv_rej_rate=(pl.col("decision") == "rejected").mean(),
        pv_days_since_last=pl.col("age_d").min(), pv_days_since_first=pl.col("age_d").max(),
        pv_n_90d=(pl.col("age_d") <= 90).sum(), pv_n_180d=(pl.col("age_d") <= 180).sum(),
        pv_rej_365d=((pl.col("decision") == "rejected") & (pl.col("age_d") <= 365)).sum(),
        pv_rate_mean=pl.col("annual_interest_rate").mean(), pv_rate_max=pl.col("annual_interest_rate").max(),
        pv_rate_last=pl.col("annual_interest_rate").sort_by("application_datetime", "application_id").drop_nulls().last(),
        pv_appr_ratio=(pl.col("approved_amount") / pl.col("requested_amount")).mean(),
        pv_req_mean=pl.col("requested_amount").mean(), pv_appr_sum=pl.col("approved_amount").sum(),
        pv_appr_max=pl.col("approved_amount").max(),
        pv_last_decision_rej=(pl.col("decision").sort_by("application_datetime", "application_id").last() == "rejected"),
        cur_rate=pl.col("cur_rate").first(), cur_amt=pl.col("cur_amt").first(),
    )
    return out.with_columns(
        pv_rate_delta=pl.col("cur_rate") - pl.col("pv_rate_last"),
        pv_amt_vs_max=pl.col("cur_amt") / pl.col("pv_appr_max"),
        pv_last_decision_rej=pl.col("pv_last_decision_rej").cast(pl.Int8),
    ).drop("cur_rate", "cur_amt")


def v2_features(apps: pl.DataFrame, tables: dict) -> pl.DataFrame:
    """v2 additions (eda/21-26): signals left in the v1 residual. Same cutoff rules as the v1 builders."""
    cut = apps.select("customer_id", "application_id", cutoff=pl.col("decision_datetime"),
                      inst_new=pl.col("proposed_monthly_installment"), amt_new=pl.col("loan_amount"))
    # bureau: score trend/range over every pull, newest report, delinquency over all pulls, external inquiries
    b = tables["bureau_observations"].join(cut.select("customer_id", "cutoff", sel=pl.col("application_id")), on="customer_id")
    b = b.filter(pl.col("available_datetime") <= pl.col("cutoff")).with_columns(
        age_d=_days("cutoff", "report_datetime"), is_sel=pl.col("application_id") == pl.col("sel"))
    bz = b.group_by("customer_id").agg(
        bz_trend=pl.cov(pl.col("age_d"), pl.col("score")) / pl.col("age_d").var() * -365,
        bz_range=pl.col("score").max() - pl.col("score").min(),
        bz_newest=pl.col("score").sort_by("report_datetime", "bureau_observation_id").drop_nulls().last(),
        bz_delinq_all_mean=pl.col("delinquent_accounts_last_12m").mean(),
        bz_active_trend=pl.col("active_accounts").filter("is_sel").mean()
        - pl.col("active_accounts").filter(~pl.col("is_sel")).mean(),
        bz_inq_cur=pl.col("inquiries_last_6m").filter("is_sel").mean(),
    )
    p = tables["previous_applications"].join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff"))
    own6 = p.filter(_days("cutoff", "application_datetime") <= 183).group_by("customer_id").agg(own6=pl.len())
    bz = bz.join(own6, on="customer_id", how="left").with_columns(
        bz_inq_external=pl.col("bz_inq_cur") - pl.col("own6").fill_null(0)).drop("bz_inq_cur", "own6")
    # previous approvals: tenor mix, recency, recent approved volume, remaining scheduled obligation
    ap = p.filter(pl.col("decision") == "approved").with_columns(
        age_d=_days("cutoff", "decision_datetime"),
        rem=(pl.col("term_months") - (_days("cutoff", "decision_datetime") / 30.44).floor()).clip(lower_bound=0))
    ob = ap.group_by("customer_id").agg(
        ob_term_mean=pl.col("term_months").mean(), ob_days_since_appr=pl.col("age_d").min(),
        ob_amt_90d=pl.col("approved_amount").filter(pl.col("age_d") <= 90).sum(),
        ob_remaining=(pl.col("scheduled_monthly_installment") * pl.col("rem")).sum(),
        amt_new=pl.col("amt_new").first(),
    ).with_columns(ob_amt_vs_last=pl.col("amt_new") / (pl.col("ob_amt_90d") + 1e6)).drop("amt_new")
    # repayment: first installment, latest loan, 1-29 dpd share, recent trend, streaks, severity when late
    r = tables["repayment_history"].join(cut.select("customer_id", "cutoff"), on="customer_id")
    # loan_id breaks due_date ties between concurrent loans -> deterministic order
    r = r.filter(pl.col("due_date") <= pl.col("cutoff")).sort("customer_id", "due_date", "loan_id").with_columns(
        late=(pl.col("max_days_past_due") > 0).cast(pl.Int32),
        paid=pl.col("amount_paid") >= pl.col("amount_due"),
        late_d=_days("last_payment_datetime", "due_date"),
        k=pl.col("due_date").rank("ordinal").over("loan_id"),
        loan_start=pl.col("due_date").min().over("loan_id"))
    # streak of consecutive late installments within one loan (concurrent loans interleave in time)
    r = r.with_columns(run=(pl.col("late") != pl.col("late").shift(1)).fill_null(True).cum_sum().over("loan_id"))
    streak = (r.filter(pl.col("late") == 1).group_by("customer_id", "loan_id", "run").agg(n=pl.len())
              .group_by("customer_id").agg(rq_streak_max=pl.col("n").max()))
    last = (r.filter(pl.col("loan_start") == pl.col("loan_start").max().over("customer_id"))
            .group_by("customer_id").agg(rq_lastloan_late=pl.col("late").mean(), rq_lastloan_maxdpd=pl.col("max_days_past_due").max()))
    dpd, late = pl.col("max_days_past_due"), pl.col("max_days_past_due") > 0
    rq = r.group_by("customer_id").agg(
        rq_first_inst_late=pl.col("late").filter(pl.col("k") == 1).mean(),
        rq_frac_1_29=(late & (dpd < 30)).mean(),
        rq_late_trend=pl.col("late").tail(6).mean() - pl.col("late").head(pl.len() - 6).mean(),
        sv_maxdpd_mean_late=dpd.filter(late).mean(),
        sv_late_d_mean=pl.col("late_d").filter(pl.col("paid") & (pl.col("late_d") > 0)).mean(),
        sv_share30_of_late=(dpd >= 30).sum() / late.sum(),
        sv_unpaid_of_late=(~pl.col("paid") & late).sum() / late.sum(),
    ).join(streak, on="customer_id", how="left").join(last, on="customer_id", how="left")
    # account: non-income inflows, recent outflows, worst free cash flow, new installment vs free cash / min balance
    a = tables["account_behavior"].join(cut.select("customer_id", "cutoff", "inst_new"), on="customer_id")
    a = a.filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff")).with_columns(
        l3=pl.col("month").rank("dense", descending=True).over("customer_id") <= 3,
        free=pl.col("inflows") - pl.col("outflows"))
    cf = a.group_by("customer_id").agg(
        cf_nonincome_share=1 - pl.col("income_like_credits").sum() / pl.col("inflows").sum(),
        cf_nonincome_l3=1 - pl.col("income_like_credits").filter("l3").sum() / pl.col("inflows").filter("l3").sum(),
        cf_out_l3=pl.col("outflows").filter("l3").mean(), cf_free_min=pl.col("free").min(),
        cf_free_l3=pl.col("free").filter("l3").mean(), cf_inc_l3=pl.col("income_like_credits").filter("l3").mean(),
        cf_minbal_l3=pl.col("minimum_balance").filter("l3").min(), inst_new=pl.col("inst_new").first(),
    ).with_columns(
        cf_inst_to_free=pl.col("inst_new") / (pl.col("cf_free_l3").clip(lower_bound=0) + 1e5),
        cf_out_to_inc_l3=pl.col("cf_out_l3") / (pl.col("cf_inc_l3") + 1e5),
        cf_inst_to_minbal=pl.col("inst_new") / (pl.col("cf_minbal_l3") + 1e5),
    ).drop("inst_new", "cf_free_l3", "cf_inc_l3")
    # e-commerce: cancellations, night share, basket size, AOV shift, tightest gap, smooth recency-weighted counts
    e = tables["ecommerce_transactions"].join(cut.select("customer_id", "cutoff"), on="customer_id")
    e = e.filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    o = e.group_by("order_id").agg(pl.col("customer_id", "cutoff", "transaction_datetime", "order_total", "order_status").first(),
                                   n_items=pl.len())
    o = o.with_columns(age_d=_days("cutoff", "transaction_datetime"), done=pl.col("order_status") == "completed",
                       night=pl.col("transaction_datetime").dt.hour() < 5)
    r28, old = pl.col("age_d") <= 28, pl.col("age_d") > 182
    ec = o.group_by("customer_id").agg(
        ec2_base=(old & (pl.col("age_d") <= 730)).sum()
        / ((pl.col("age_d").max().clip(upper_bound=730) - 182).clip(lower_bound=14) / 7),
        ec2_cancel_28=(~pl.col("done")).filter(r28).sum(),
        ec2_night_share_90=pl.col("night").filter(pl.col("age_d") <= 90).mean(),
        ec2_items_per_order_28=pl.col("n_items").filter(r28).mean(),
        ec2_aov_28=pl.col("order_total").filter(pl.col("done") & r28).mean(),
        ec2_aov_old=pl.col("order_total").filter(pl.col("done") & old).mean(),
        ec2_gap_min_28=pl.col("age_d").filter(r28).sort().diff().min(),
        **{f"ec2_ewm{t}": (-pl.col("age_d") / t).exp().sum() for t in (7, 28, 56)},
    ).with_columns(
        ec2_aov_shift=pl.col("ec2_aov_28") / pl.col("ec2_aov_old"),
        ec2_ewm7_rel=pl.col("ec2_ewm7") / (pl.col("ec2_base") + 0.05),
        ec2_ewm28_exc=pl.col("ec2_ewm28") - pl.col("ec2_base") * 4,
        ec2_ewm56_rel=pl.col("ec2_ewm56") / (pl.col("ec2_base") * 8 + 0.05),
    ).drop("ec2_aov_28", "ec2_aov_old", "ec2_ewm7", "ec2_ewm28", "ec2_ewm56")
    out = cut.select("customer_id")
    for part in (bz, ob, rq, cf, ec):
        out = out.join(part, on="customer_id", how="left")
    return out


PP_CATS = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]


def v5_features(apps: pl.DataFrame, tables: dict) -> pl.DataFrame:
    """v5 additions (eda/40-45): unit-price premium within category, basket quantity, decision month.
    Category reference (mean and sd of log unit price, eda/50): an expanding window over item rows of all customers
    (no labels) dated before the first day of the application's decision month, so every application is normalised
    only with prices that were already observed when it was decided."""
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"), dm=pl.col("decision_datetime").dt.truncate("1mo"))
    raw = tables["ecommerce_transactions"].with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(),
                                                        ym=pl.col("transaction_datetime").dt.truncate("1mo"))
    ms = raw.group_by("product_category", "ym").agg(n=pl.len(), s=pl.col("lu").sum(), ss=(pl.col("lu") ** 2).sum())
    ref = (cut.select("dm").unique().join(ms, how="cross").filter(pl.col("ym") < pl.col("dm"))
           .group_by("product_category", "dm").agg(pl.col("n", "s", "ss").sum())
           .with_columns(mu=pl.col("s") / pl.col("n"),
                         sd=((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt()))
    e = raw.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    e = e.join(ref.select("product_category", "dm", "mu", "sd"), on=["product_category", "dm"], how="left")
    e = e.with_columns(age_d=_days("cutoff", "transaction_datetime"), done=pl.col("order_status") == "completed",
                       z=(pl.col("lu") - pl.col("mu")) / pl.col("sd"))
    lo, hi = pl.col("z").quantile(0.1), pl.col("z").quantile(0.9)
    pp = e.group_by("customer_id").agg(
        pp_mean=pl.col("z").mean(),
        pp_trim=pl.col("z").filter((pl.col("z") > lo) & (pl.col("z") < hi)).mean(),
        pp_done=pl.col("z").filter("done").mean(), pp_min=pl.col("z").min(), pp_max=pl.col("z").max(),
        pp_30=pl.col("z").filter(pl.col("age_d") <= 30).mean(), pp_90=pl.col("z").filter(pl.col("age_d") <= 90).mean(),
        pp_old=pl.col("z").filter(pl.col("age_d") > 182).mean(),
        pp_qty_item=pl.col("quantity").mean(), pp_qty_item_90=pl.col("quantity").filter(pl.col("age_d") <= 90).mean(),
        **{f"pp_cat_{c}": pl.col("z").filter(pl.col("product_category") == c).mean() for c in PP_CATS},
    )
    month = apps.select("customer_id", dec_month=pl.col("decision_datetime").dt.month())
    return month.join(pp, on="customer_id", how="left")


def build_all(tables: dict) -> pl.DataFrame:
    """tables: dict of polars frames keyed by table name. Returns one row per selected application."""
    tr = tables["train_applications"].with_columns(is_train=pl.lit(1))
    te = tables["test_applications"].with_columns(target=pl.lit(None, pl.Int64), is_train=pl.lit(0))
    apps = pl.concat([tr, te], how="diagonal_relaxed")
    f = apps.select("application_id", "customer_id", "decision_datetime", "target", "is_train")
    for part in [
        app_features(apps, tables["customer_profiles"]),
        bureau_features(apps, tables["bureau_observations"]),
        repayment_features(apps, tables["repayment_history"], tables["previous_applications"]),
        account_features(apps, tables["account_behavior"]),
        ecommerce_features(apps, tables["ecommerce_transactions"]),
        prev_features(apps, tables["previous_applications"]),
        v2_features(apps, tables),
        v5_features(apps, tables),
    ]:
        f = f.join(part, on="customer_id", how="left")
    # cross-table ratios: the real affordability picture
    f = f.with_columns(
        has_history=pl.col("pv_n").is_not_null().cast(pl.Int8),
        dti_verified=pl.col("proposed_monthly_installment") / pl.col("ac_income_mean"),
        dti_total=(pl.col("proposed_monthly_installment") + pl.col("rp_active_inst").fill_null(0))
        / pl.col("declared_monthly_income"),
        income_verif_ratio=pl.col("ac_income_mean") / pl.col("declared_monthly_income"),
        debt_to_income=pl.col("bc_outstanding_max") / pl.col("declared_monthly_income"),
        inst_to_avgbal=pl.col("proposed_monthly_installment") / (pl.col("ac_avgbal_mean") + 1),
        ecspend_to_income=pl.col("ec_spend_per_month") / pl.col("declared_monthly_income"),
    )
    return f
