"""Discovery/confirmation conditional association, FDR and validation KS of individual features.
No predictive model is fitted. Existing adaptive OOF is a diagnostic anchor, not a fresh holdout.
"""
from deep_common import *

def main():
    f,y,raw,R,s=v3()
    new=pl.read_parquet(OUT/"candidate_features.parquet")
    f=f.join(new.drop("customer_id"),on="application_id",how="left",validate="1:1",maintain_order="left")
    te=features().filter(pl.col("is_train")==0).join(new.drop("customer_id"),on="application_id",how="left",validate="1:1",maintain_order="left")
    discovery=folds(y,20261013,2)==0
    hist=f["has_history"].to_numpy()
    risk=np.minimum((rankdata(s)/len(y)*30).astype(int),29)
    strata=risk*2+hist
    cols=sorted(c for c in f.columns if c not in IDS and c not in CATS)
    rows=[]
    for c in cols:
        x=f[c].cast(pl.Float64).to_numpy()
        bins=qbins(x,x[discovery],6)
        da=conditional_test(y[discovery],bins[discovery],strata[discovery])
        db=conditional_test(y[~discovery],bins[~discovery],strata[~discovery])
        finite=np.isfinite(x)
        # Fixed direction and missing-side chosen only on discovery; applied to confirmation.
        obs=finite&discovery
        med=np.median(x[obs]) if obs.any() else 0.
        candidates=[]
        low=float(np.min(x[obs])-1) if obs.any() else -1.
        high=float(np.max(x[obs])+1) if obs.any() else 1.
        for direction in (1,-1):
            for fill in (low,high,med):
                p=direction*np.where(finite,x,fill)
                candidates.append((signed_ks(y[discovery],p[discovery]),direction,fill))
        _,direction,fill=max(candidates,key=lambda z:z[0])
        p=direction*np.where(finite,x,fill)
        rows.append(dict(feature=c,new=c.startswith("dx_"),p_discovery=da[2],p_confirmation=db[2],
                         conditional_chi2_confirmation=db[0],df=db[1],
                         validation_signed_KS=signed_ks(y[~discovery],p[~discovery]),
                         KS_observed_only=ks(y[finite],x[finite]) if finite.sum() and len(np.unique(y[finite]))==2 else 0.,
                         missing_share=float((~finite).mean()),PSI=psi(x,te[c].cast(pl.Float64).to_numpy())))
    res=pl.DataFrame(rows).with_columns(q_discovery=pl.Series(bh([r["p_discovery"] for r in rows])),
                                        q_confirmation=pl.Series(bh([r["p_confirmation"] for r in rows])))
    res=res.with_columns(replicates=(pl.col("q_discovery")<.05)&(pl.col("q_confirmation")<.05)).sort("p_discovery")
    save_table(res,"03_all_feature_screen")
    save_table(res.filter("new").sort("p_discovery"),"03_new_feature_screen")
    # Increasing risk stratification resolution tests whether apparent residual is just under-conditioning.
    shortlist=res.filter("new").head(12)["feature"].to_list()
    robust=[]
    for c in shortlist:
        x=f[c].cast(pl.Float64).to_numpy(); b=qbins(x,x[discovery],6)
        for nb in (10,30,60):
            st=np.minimum((rankdata(s)/len(y)*nb).astype(int),nb-1)*2+hist
            stat,df,p=conditional_test(y[~discovery],b[~discovery],st[~discovery])
            robust.append(dict(feature=c,risk_bins=nb,chi2=stat,df=df,p_confirmation=p))
    save_table(robust,"03_conditioning_sensitivity")
    # Restricted permutations are a diagnostic null: keep risk and history composition fixed.
    rng=np.random.default_rng(7)
    yy=y[~discovery]; ss=strata[~discovery]
    groups=[np.flatnonzero(ss==i) for i in np.unique(ss)]
    perm=[]
    for c in shortlist[:6]:
        x=f[c].cast(pl.Float64).to_numpy(); b=qbins(x,x[discovery],6)[~discovery]
        observed=conditional_test(yy,b,ss)[0]
        null=[]
        for _ in range(99):
            yp=yy.copy()
            for idx in groups:yp[idx]=rng.permutation(yp[idx])
            null.append(conditional_test(yp,b,ss)[0])
        perm.append(dict(feature=c,observed=observed,permutation_p=(1+sum(t>=observed for t in null))/100))
    save_table(perm,"03_permutation_confirmation")
    # Correlation conditioned by existing tree scores is NOT causal or proof of ensemble gain.
    chosen=["rp_frac_dpd1","bc_score_mean","ec_accel_30d","ac_out_in_delta"]+shortlist[:8]
    mat=np.column_stack([rankdata(np.nan_to_num(f[c].cast(pl.Float64).to_numpy(),nan=-1e30)) for c in chosen])
    corr=np.corrcoef(mat.T)
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    d=res.filter("new").head(15)
    x=-np.log10(np.maximum(d["p_confirmation"].to_numpy(),1e-15))
    ax[0].barh(d["feature"].to_list()[::-1],x[::-1]);ax[0].axvline(-np.log10(.05),ls="--",color="grey")
    ax[0].set_title("Discovery top-15: confirmation association\nDashed line is nominal p=.05, NOT FDR")
    ax[0].set_xlabel("-log10 confirmation p")
    im=ax[1].imshow(corr,vmin=-1,vmax=1,cmap="coolwarm")
    ax[1].set_xticks(range(len(chosen)),chosen,rotation=90,fontsize=7)
    ax[1].set_yticks(range(len(chosen)),chosen,fontsize=7)
    ax[1].set_title("Spearman; missing mapped below observed values");fig.colorbar(im,ax=ax[1],shrink=.65)
    figsave(fig,"03_residual_and_correlation")
    # Do not automatically bless candidates based on one test; provide a shortlist with its evidence.
    save_json(dict(candidate_count=len(cols),new_count=sum(c.startswith("dx_") for c in cols),
        new_FDR_replicated=res.filter(pl.col("new")&pl.col("replicates"))["feature"].to_list(),
        discovery_shortlist=shortlist,
        limitation="Association conditional on coarse v3 risk bins is not incremental model KS; all labels already influenced historical EDA and model selection.",
        decision="Keep candidates separate. A fold-safe remote model ablation is required before promotion."),"03_decision")
    assert np.all(res["q_confirmation"].to_numpy()>=res["p_confirmation"].to_numpy()-1e-12)

if __name__=="__main__":
    self_check()
    main()
