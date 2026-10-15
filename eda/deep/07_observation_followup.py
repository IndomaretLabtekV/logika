"""Follow up the sole discovery-FDR candidate and audit observation-window composition."""
from deep_common import *

def main():
    f,y,raw,R,s=v3()
    X=pl.read_parquet(OUT/"candidate_features.parquet")
    f=f.join(X.drop("customer_id"),on="application_id",how="left",validate="1:1",maintain_order="left")
    disc=folds(y,20261013,2)==0
    candidate="dx_ec_spend_112_224"
    x=f[candidate].to_numpy()
    bins=qbins(x,x[disc],6)
    hist=f["has_history"].to_numpy()
    risk=np.minimum((rankdata(s)/len(y)*30).astype(int),29)
    span=f["dx_ec_observed_span"].to_numpy()
    sc=qbins(span,span[disc],5)
    sensitivity=[]
    for label,st in [("risk30+history",risk*2+hist),
                     ("risk30+history+observed_span_quintile",(risk*2+hist)*6+sc)]:
        for part,mask in [("discovery",disc),("confirmation",~disc)]:
            stat,df,p=conditional_test(y[mask],bins[mask],st[mask])
            sensitivity.append(dict(candidate=candidate,conditioning=label,part=part,chi2=stat,df=df,p=p,n=int(mask.sum())))
    save_table(sensitivity,"07_spend_followup")
    # Honest two-stage family: only discovery-FDR hits are forwarded, no confirmation-based re-selection.
    screening=pl.read_csv(OUT/"03_all_feature_screen.csv")
    nominated=screening.filter(pl.col("q_discovery")<.05)
    if nominated.height:
        nominated=nominated.with_columns(q_confirmation_selected_family=pl.Series(bh(nominated["p_confirmation"].to_numpy())))
    save_table(nominated,"07_discovery_nominated_confirmation")
    # Descriptive conditional deviations on the confirmation half, not probabilities for prediction.
    st=(risk*2+hist)[~disc];yy=y[~disc];bb=bins[~disc]
    pstr=np.bincount(st,weights=yy)/np.maximum(np.bincount(st),1)
    er=pstr[st]
    profile=[]
    for b in np.unique(bb):
        m=bb==b
        profile.append(dict(bin=int(b),n=int(m.sum()),observed_rate=float(yy[m].mean()),
                            risk_stratum_rate=float(er[m].mean()),
                            excess_rate=float((yy[m]-er[m]).mean())))
    save_table(profile,"07_confirmation_spend_profile")
    # The same population at every month avoids a changing-composition explanation of account trends.
    A=apps()
    ac=table("account_behavior").join(A.select("customer_id","decision_datetime","target"),on="customer_id",validate="m:1")
    ac=ac.filter(pl.col("target").is_not_null()).with_columns(
        m=(pl.col("decision_datetime").dt.year()-pl.col("month").dt.year())*12+pl.col("decision_datetime").dt.month()-pl.col("month").dt.month(),
        ratio=pl.col("outflows")/pl.col("inflows"))
    lengths=ac.group_by("customer_id").agg(max_m=pl.col("m").max(),n=pl.len(),n_m=pl.col("m").n_unique(),min_m=pl.col("m").min())
    assert lengths.filter((pl.col("n")!=pl.col("n_m"))|(pl.col("min_m")!=1)|(pl.col("max_m")!=pl.col("n"))).height==0
    cohort=lengths.filter(pl.col("max_m")>=12).select("customer_id")
    fixed=ac.join(cohort,on="customer_id",how="inner").filter(pl.col("m")<=12)
    summaries=[]
    for label,frame in [("all_available",ac.filter(pl.col("m")<=12)),("same_customers_12m",fixed)]:
        g=frame.group_by("target","m").agg(ratio=pl.col("ratio").mean(),n=pl.len()).with_columns(cohort=pl.lit(label))
        summaries.append(g)
    panel=pl.concat(summaries).sort("cohort","target","m")
    save_table(panel,"07_matched_history_panel")
    availability=dict(account_customers=lengths.height,customers_with_12_complete_months=cohort.height,
                      median_ecommerce_oldest_event_days=float(np.median(span)),
                      fraction_oldest_event_at_least_365=float(np.mean(span>=365)),
                      oldest_event_is_observation_start=False)
    save_json(availability,"07_observation_support")
    fig,ax=plt.subplots(1,3,figsize=(16,4.5))
    for label,style in [("all_available","--"),("same_customers_12m","-")]:
        for cls,col in [(0,"C0"),(1,"C1")]:
            d=panel.filter((pl.col("cohort")==label)&(pl.col("target")==cls))
            ax[0].plot(d["m"],d["ratio"],style,color=col,label=f"{label}, y={cls}")
    ax[0].set_title("Cashflow trend: same customers vs changing cohort");ax[0].legend(fontsize=7)
    ax[1].bar([r["bin"] for r in profile],[r["excess_rate"] for r in profile])
    ax[1].axhline(0,color="grey");ax[1].set_title("Confirmation: spend112-224 residual\nDescriptive rate difference, not model gain")
    ax[1].set_xlabel("Discovery-defined spending bin")
    ax[2].hist(span,bins=40);ax[2].axvline(365,color="black",ls="--")
    ax[2].set_title("Observed oldest e-commerce event\nNot a known observation-start timestamp")
    figsave(fig,"07_followup")
    print("One discovery-selected candidate may pass confirmation while none pass correction over all 289 confirmation tests.")
    print("This is preliminary: prior OOF training/early-stopping and historical EDA reused labels; no fresh generalization proof.")

if __name__=="__main__":
    self_check()
    main()
