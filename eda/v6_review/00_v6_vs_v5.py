"""Compare saved predictions, fixed component removals and paired sampling; no model/weight fitting."""
from review_common import *

def main():
    checks(); a,b=bundle(5),bundle(6);y=b["y"];n=len(y)
    assert np.array_equal(a["tr"]["application_id"],b["tr"]["application_id"])
    rows=[]
    for v,d in [(5,a),(6,b)]:
        for name,p in {**d["raw"],"selected_blend":d["score"],"equal_trees":d["R"][:,:6].mean(1)}.items():
            auc=(rankdata(p)[y==1].sum()-(y==1).sum()*((y==1).sum()+1)/2)/((y==1).sum()*(y==0).sum())
            rows.append(dict(version=v,model=name,KS=ks(y,p),AUC=auc))
    scores=csv(rows,"00_scores")
    removed=[]
    for j,name in enumerate(b["names"]):
        if b["w"][j]==0:continue
        w=b["w"].copy();w[j]=0;w/=w.sum()
        p=b["R"]@w
        removed.append(dict(removed=name,old_weight=b["w"][j],KS=ks(y,p),delta_vs_selected=ks(y,p)-ks(y,b["score"])))
    csv(removed,"00_fixed_component_removal")
    month=b["tr"]["decision_datetime"].dt.month().to_numpy();hist=b["tr"]["has_history"].to_numpy()
    groups={"all":np.ones(n,bool),**{f"history_{h}":hist==h for h in [0,1]},**{f"month_{m:02}":month==m for m in range(1,13)}}
    segments=csv([dict(group=g,n=int(ix.sum()),v5=ks(y[ix],a["score"][ix]),v6=ks(y[ix],b["score"][ix])) for g,ix in groups.items()],"00_segments")
    rng=np.random.default_rng(61014);plans=[ks_plan(y,d["score"]) for d in [a,b]]
    draws=[];indices=[np.flatnonzero(y==k) for k in [0,1]]
    for _ in range(400):
        w=np.bincount(np.concatenate([rng.choice(c,len(c),replace=True) for c in indices]),minlength=n)
        draws.append(planned_ks(plans[1],w)-planned_ks(plans[0],w))
    samples=[];gaps=[]
    for share in [.3,.5]:
        for it in range(400):
            ix=rng.choice(n,30000,replace=False);k=int(30000*share)
            wp=np.bincount(ix[:k],minlength=n);wv=np.bincount(ix[k:],minlength=n)
            p5,p6=[planned_ks(p,wp) for p in plans]
            q5,q6=[planned_ks(p,wv) for p in plans]
            samples.append(dict(share=share,iteration=it,public_delta_v6_v5=p6-p5,private_delta_v6_v5=q6-q5,v6_public_private_gap=p6-q6))
        rr=pd.DataFrame(samples);r=rr[rr.share==share]
        gaps.append(dict(public_share=share,prob_v6_lower_than_v5=float((r.public_delta_v6_v5<0).mean()),
            paired_delta_p025=r.public_delta_v6_v5.quantile(.025),paired_delta_p975=r.public_delta_v6_v5.quantile(.975),
            abs_gap_v6_p95=r.v6_public_private_gap.abs().quantile(.95)))
    csv(samples,"00_sampling_draws");csv(gaps,"00_sampling_summary")
    js(dict(v5_KS=ks(y,a["score"]),v6_KS=ks(y,b["score"]),delta=ks(y,b["score"])-ks(y,a["score"]),
        paired_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
        v6_public_reported=.56123,v5_public_reported=None,
        warning="Selected blends, repeated label use and fixed saved models: resampling does not include model-selection/retraining uncertainty."),
        "00_decision")
    # Threshold from discovery only; held-half examples are explanatory, not labeling test rows.
    half=folds(y,seed=61014,k=2)==0;s=b["score"]
    order=np.argsort(s[half]);sy=y[half][order];ss=s[half][order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    delta=np.cumsum(1-sy)/(sy==0).sum()-np.cumsum(sy)/(sy==1).sum()
    threshold=ss[ends[np.argmax(delta[ends])]]
    pred=s>threshold
    confusion=csv([dict(actual=c,predicted=k,n=int(((y==c)&(pred==k)&~half).sum())) for c in [0,1] for k in [0,1]],"00_held_confusion")
    idx=np.r_[np.flatnonzero((y==1)&~pred&~half)[:20],np.flatnonzero((y==0)&pred&~half)[:20]]
    examples=b["tr"][idx].select("application_id","target","has_history","pp_mean","rp_frac_dpd1","annual_interest_rate","ec2_ewm28_exc").with_columns(
        v5_score=pl.Series(a["score"][idx]),v6_score=pl.Series(s[idx]),cutoff=pl.lit(threshold))
    csv(examples.to_pandas(),"00_error_examples")
    b["tr"].select("application_id","customer_id","target").with_columns(v5_score=pl.Series(a["score"]),v6_score=pl.Series(s)).write_parquet(OUT/"00_bound_oof.parquet")
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    m=segments[segments.group.str.startswith("month")]
    ax[0].plot(range(1,13),m.v5,"o-",label="v5");ax[0].plot(range(1,13),m.v6,"s--",label="v6")
    ax[0].set(title="Saved random OOF by month",ylabel="KS");ax[0].legend()
    ax[1].hist(draws,bins=30);ax[1].axvline(0,color="red");ax[1].set(title="Paired bootstrap v6 - v5",xlabel="KS difference")
    corr=pd.DataFrame(b["R"],columns=b["names"]).corr()
    im=ax[2].imshow(corr,vmin=.95,vmax=1);ax[2].set_xticks(range(9),b["names"],rotation=90);ax[2].set_yticks(range(9),b["names"])
    ax[2].set_title("v6 rank correlations");fig.colorbar(im,ax=ax[2]);figsave(fig,"00_comparison")
    csv(corr.reset_index(),"00_prediction_correlation")
    paths=[ROOT/"notebook/v6.ipynb",ROOT/"data_pq/features.parquet",ROOT/"results/v6/submission.csv"]
    paths+=sorted((ROOT/"results/v6").glob("oof_*.npy"))
    js({str(p.relative_to(ROOT)):digest(p) for p in paths},"00_input_sha256")

if __name__=="__main__":main()
