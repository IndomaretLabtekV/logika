"""Cutoff-safe hypotheses: disjoint windows, burst excess, DPD maturity, cashflow and bureau reliability."""
from deep_common import *
import gc

def build_candidates(T):
    A=pl.concat([T["train_applications"],T["test_applications"]],how="diagonal_relaxed")
    cut=A.select("customer_id",cutoff="decision_datetime",app_time="application_datetime")
    parts=[]
    e=T["ecommerce_transactions"].join(cut,on="customer_id",validate="m:1")
    e=e.filter(pl.col("transaction_datetime")<=pl.col("cutoff"))
    o=e.group_by("order_id").agg(pl.col("customer_id","cutoff","app_time","transaction_datetime","order_total","order_status","payment_method").first(),
                                 item_sum=pl.col("item_total").sum(),item_count=pl.len(),quantity=pl.col("quantity").sum())
    assert o.filter(pl.col("item_sum")!=pl.col("order_total")).height==0
    o=o.with_columns(age=(pl.col("cutoff")-pl.col("transaction_datetime")).dt.total_seconds()/86400,
                     done=pl.col("order_status")=="completed")
    windows=[(0,7),(7,28),(28,56),(56,112),(112,224),(224,365)]
    expr=[]
    for lo,hi in windows:
        w=(pl.col("age")>=lo)&(pl.col("age")<hi)
        expr += [(w&pl.col("done")).sum().alias(f"dx_ec_count_{lo}_{hi}"),
                 pl.col("order_total").filter(w&pl.col("done")).sum().alias(f"dx_ec_spend_{lo}_{hi}"),
                 (w&~pl.col("done")).sum().alias(f"dx_ec_cancel_{lo}_{hi}")]
    for tau in (7,28,56):
        # Fixed-support integral gives the scale of a constant-rate process, not observed exposure.
        expr.append((-pl.col("age")/tau).exp().filter(pl.col("age")<365).sum().alias(f"dx_ec_kernel_{tau}"))
    ec=o.group_by("customer_id").agg(expr+[
        pl.col("age").max().alias("dx_ec_observed_span"),
        pl.col("order_total").filter(pl.col("done")&(pl.col("age")<28)).max().alias("dx_ec_max_basket28"),
        pl.col("order_total").filter(pl.col("done")&(pl.col("age")<28)).sum().alias("_sp28"),
        (pl.col("age")<28).sum().alias("_n28"),
        ((pl.col("age")>=112)&(pl.col("age")<365)).sum().alias("_base"),
        (pl.col("transaction_datetime")>=pl.col("app_time")).sum().alias("dx_ec_after_application"),
        pl.col("quantity").filter(pl.col("age")<28).sum().alias("dx_ec_quantity28"),
    ])
    ec=ec.with_columns(
        dx_ec_top_basket_share28=pl.col("dx_ec_max_basket28")/pl.when(pl.col("_sp28")>0).then(pl.col("_sp28")),
        dx_ec_excess28=pl.col("_n28")-pl.col("_base")*28/253,
        dx_ec_excess_z28=(pl.col("_n28")-pl.col("_base")*28/253)/(pl.col("_base")*28/253+1).sqrt(),
        dx_ec_log_rate_ratio28=((pl.col("_n28")+.5)/28/((pl.col("_base")+.5)/253)).log(),
        **{f"dx_ec_kernel_excess_{t}":pl.col(f"dx_ec_kernel_{t}")-pl.col("_base")/253*t*(1-np.exp(-365/t)) for t in (7,28,56)}
    ).drop("_sp28","_n28","_base")
    parts.append(ec)

    r=T["repayment_history"].join(cut.select("customer_id","cutoff"),on="customer_id",validate="m:1")
    r=r.filter(pl.col("due_date")<=pl.col("cutoff")).with_columns(
        age=(pl.col("cutoff")-pl.col("due_date")).dt.total_seconds()/86400,
        late=pl.col("max_days_past_due")>0,partial=(pl.col("amount_paid")>0)&(pl.col("amount_paid")<pl.col("amount_due")),
        unpaid=(pl.col("amount_due")-pl.col("amount_paid")).clip(lower_bound=0))
    expr=[pl.len().alias("dx_rp_n"),(pl.col("age")<90).mean().alias("dx_rp_immature90_share"),
          pl.col("partial").mean().alias("dx_rp_partial_share"),
          (pl.col("unpaid").sum()/pl.col("amount_due").sum()).alias("dx_rp_unpaid_fraction")]
    for d in (1,30,60,90):
        mature=pl.col("age")>=d
        expr += [mature.sum().alias(f"dx_rp_eligible{d}"),
                 (pl.col("max_days_past_due")>=d).filter(mature).mean().alias(f"dx_rp_mature{d}_rate")]
    for lo,hi in [(0,90),(90,180),(180,365),(365,2000)]:
        expr.append(pl.col("late").filter(pl.col("age").is_between(lo,hi,closed="left")).mean().alias(f"dx_rp_late_{lo}_{hi}"))
    rp=r.group_by("customer_id").agg(expr)
    loan=r.group_by("customer_id","loan_id").agg(
        p_late=pl.col("late").mean(),n=pl.len(),worst=pl.col("max_days_past_due").max(),
        max_age=pl.col("age").max(),open_unpaid=pl.col("unpaid").sum())
    lp=loan.group_by("customer_id").agg(
        dx_rp_loan_equal_late=pl.col("p_late").mean(),dx_rp_loan_late_std=pl.col("p_late").std(),
        dx_rp_loan_worst_mean=pl.col("worst").mean(),dx_rp_loan_count=pl.len(),
        dx_rp_unpaid_loan_share=(pl.col("open_unpaid")>0).mean())
    parts.append(rp.join(lp,on="customer_id",validate="1:1"))

    ac=T["account_behavior"].join(cut.select("customer_id","cutoff"),on="customer_id",validate="m:1")
    ac=ac.filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime)<=pl.col("cutoff"))
    ac=ac.with_columns(m=(pl.col("cutoff").dt.year()-pl.col("month").dt.year())*12+pl.col("cutoff").dt.month()-pl.col("month").dt.month(),
        flow_ratio=pl.col("outflows")/pl.when(pl.col("inflows")>0).then(pl.col("inflows")),
        low_fraction=pl.col("low_balance_days")/pl.col("month").dt.month_end().dt.day(),
        surplus=pl.col("inflows")-pl.col("outflows"),
        log_income=pl.col("income_like_credits").log1p())
    expr=[]
    for lo,hi in [(1,3),(4,6),(7,12)]:
        w=pl.col("m").is_between(lo,hi)
        for col in ["flow_ratio","low_fraction","surplus","log_income"]:
            expr.append(pl.col(col).filter(w).mean().alias(f"dx_ac_{col}_{lo}_{hi}"))
    expr += [pl.col("surplus").filter(pl.col("m")<=3).min().alias("dx_ac_surplus_min3"),
             pl.col("log_income").std().alias("dx_ac_log_income_sd"),
             (pl.col("surplus")<0).filter(pl.col("m")<=3).mean().alias("dx_ac_negative_month_share3")]
    aa=ac.group_by("customer_id").agg(expr).with_columns(
        dx_ac_flow_shock=pl.col("dx_ac_flow_ratio_1_3")-pl.col("dx_ac_flow_ratio_7_12"),
        dx_ac_lowday_shock=pl.col("dx_ac_low_fraction_1_3")-pl.col("dx_ac_low_fraction_7_12"),
        dx_ac_income_shock=pl.col("dx_ac_log_income_1_3")-pl.col("dx_ac_log_income_7_12"))
    parts.append(aa)

    b=T["bureau_observations"].join(cut.select("customer_id","cutoff"),on="customer_id",validate="m:1")
    b=b.filter(pl.col("available_datetime")<=pl.col("cutoff")).with_columns(
        age=(pl.col("cutoff")-pl.col("report_datetime")).dt.total_seconds()/86400)
    bp=b.sort("customer_id","bureau_provider","report_datetime","available_datetime","bureau_observation_id").group_by("customer_id","bureau_provider").agg(
        score_latest=pl.col("score").drop_nulls().last(),
        score_oldest=pl.col("score").drop_nulls().first(),
        newest_age=pl.col("age").filter(pl.col("score").is_not_null()).last())
    bb=bp.group_by("customer_id").agg(
        dx_bu_latest_provider_mean=pl.col("score_latest").mean(),
        dx_bu_latest_provider_std=pl.col("score_latest").std(),
        dx_bu_provider_trend_mean=(pl.col("score_latest")-pl.col("score_oldest")).mean(),
        dx_bu_newest_age_mean=pl.col("newest_age").mean())
    for tau in (30,180):
        w=(-pl.col("age")/tau).exp()
        z=b.group_by("customer_id").agg(((pl.col("score")*w).sum()/w.filter(pl.col("score").is_not_null()).sum()).alias(f"dx_bu_recency_score{tau}"))
        bb=bb.join(z,on="customer_id",validate="1:1")
    parts.append(bb)
    X=A.select("application_id","customer_id")
    for part in parts:
        X=X.join(part,on="customer_id",how="left",validate="1:1",maintain_order="left")
    X=X.with_columns(
        dx_cross_spend_cashflow=pl.col("dx_ec_log_rate_ratio28")*pl.col("dx_ac_flow_shock"),
        dx_cross_mature_late_burst=pl.col("dx_rp_mature30_rate")*pl.col("dx_ec_log_rate_ratio28"))
    X=X.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in X.columns if c.startswith("dx_")])
    assert X.height==A.height and X["application_id"].n_unique()==A.height
    return X,o,r,ac

def main():
    T={n:table(n) for n in TABLES}
    X,o,r,ac=build_candidates(T)
    X.write_parquet(OUT/"candidate_features.parquet")
    print("candidate matrix",X.shape,"unvalidated hypotheses, not selected model features")
    A=apps()
    tr=T["train_applications"].select("customer_id","target")
    # Physical count conservation is an independent preprocessing check.
    raw_total=T["ecommerce_transactions"]["item_total"].sum()
    assert raw_total==o["order_total"].sum()
    n365=o.filter(pl.col("age")<365).height
    cols=[c for c in X.columns if c.startswith("dx_ec_count_") or c.startswith("dx_ec_cancel_")]
    assert sum(X[c].sum() for c in cols)==n365
    audit=[]
    for c in X.columns[2:]:
        audit.append(dict(feature=c,null_share=X[c].null_count()/X.height,min=X[c].min(),max=X[c].max()))
    save_table(audit,"02_candidate_quality")
    # Future-row injection: every candidate must remain unchanged.
    selected=A.head(40).select("customer_id",cutoff="decision_datetime")
    T2=dict(T)
    for name,timecol in [("ecommerce_transactions","transaction_datetime"),("repayment_history","due_date"),
                          ("account_behavior","month"),("bureau_observations","available_datetime")]:
        extra=T[name].join(selected,on="customer_id").group_by("customer_id").first()
        when=pl.col("cutoff").dt.truncate("1mo").cast(pl.Date) if name=="account_behavior" else pl.col("cutoff")+pl.duration(days=10)
        extra=extra.with_columns(when.alias(timecol))
        T2[name]=pl.concat([T[name],extra.select(T[name].columns)],how="vertical_relaxed")
    check,*_=build_candidates(T2)
    z=X.sort("application_id"); zz=check.sort("application_id")
    moved=[c for c in X.columns[2:] if not np.allclose(z[c].to_numpy(),zz[c].to_numpy(),equal_nan=True)]
    assert not moved,moved
    print("PASS future injection: no candidate changes; order totals and window counts conserved")
    # Plot class-conditioned profiles with customer-level denominators, not row means.
    nclass=dict(tr.group_by("target").len().iter_rows())
    weekly=o.join(tr,on="customer_id").filter(pl.col("age")<365).with_columns(wk=(pl.col("age")//7).cast(pl.Int32))
    g=weekly.group_by("target","wk").agg(orders=pl.len(),spend=pl.col("order_total").filter(pl.col("done")).sum()).sort("target","wk")
    save_table(g,"02_weekly_ecommerce")
    rg=r.join(tr,on="customer_id").with_columns(age_bin=(pl.col("age")//30).clip(upper_bound=24).cast(pl.Int32))
    rg=rg.group_by("customer_id","target","age_bin").agg(late90=(pl.col("max_days_past_due")>=90).mean())
    rg=rg.group_by("target","age_bin").agg(late90=pl.col("late90").mean(),n_customer=pl.len()).sort("target","age_bin")
    save_table(rg,"02_dpd_maturity")
    ag=ac.join(tr,on="customer_id").group_by("target","m").agg(flow=pl.col("flow_ratio").mean(),low=pl.col("low_fraction").mean()).sort("target","m")
    save_table(ag,"02_account_trajectory")
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for cls in (0,1):
        d=g.filter((pl.col("target")==cls)&(pl.col("wk")<52))
        ax[0].plot(d["wk"],d["orders"]/nclass[cls],label=f"target {cls}")
        d=rg.filter(pl.col("target")==cls)
        ax[1].plot(d["age_bin"]*30,d["late90"],label=f"target {cls}")
        d=ag.filter(pl.col("target")==cls)
        ax[2].plot(d["m"],d["flow"],label=f"target {cls}")
    ax[0].set_title("Orders/customer/week before cutoff");ax[0].set_xlabel("Weeks before decision")
    ax[1].axvline(90,ls="--",color="grey");ax[1].set_title("90-DPD observation opportunity");ax[1].set_xlabel("Installment age at cutoff, days")
    ax[2].set_title("Mean outflow / inflow");ax[2].set_xlabel("Calendar months before decision")
    for a in ax:a.legend()
    figsave(fig,"02_temporal_profiles")
    print("Observed first-event age is NOT known observation-start time; avoid treating it as exact exposure.")
    print("Maturity-adjusted rates remain hypotheses: changing denominator can reduce sample size or remove useful signal.")

if __name__=="__main__":
    self_check()
    main()
