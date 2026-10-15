"""Audit FM feature-budget bias; export fold-local feature views, not trained models."""
from deep_common import *
import pandas as pd

def family(c):
    if c.startswith(("bc_","bp_","bh_","bz_")):return "bureau"
    if c.startswith(("rp_","rq_","sv_")):return "repayment"
    if c.startswith(("ac_","cf_")):return "account"
    if c.startswith("ec"):return "ecommerce"
    if c.startswith(("pv_","ob_")):return "previous"
    return "application"

def main():
    f,y,raw,R,s=v3()
    cols=sorted(c for c in f.columns if c not in IDS and c not in CATS)
    F=f.select(cols).to_pandas()
    ff=folds(y)
    coverage=[];views={};selected_sets=[]
    for fold in range(5):
        ti=np.flatnonzero(ff!=fold);vi=np.flatnonzero(ff==fold)
        train=F.iloc[ti]
        order=sorted(cols,key=lambda c:-ks_nan(y[ti],train[c].to_numpy(dtype=float)))
        corr=train.rank().corr().abs()
        def accept(c,chosen):
            # Undefined correlation is not evidence of redundancy.
            return all(not np.isfinite(corr.loc[c,k]) or corr.loc[c,k]<.95 for k in chosen)
        uni=[]
        for c in order[:200]:
            if accept(c,uni):uni.append(c)
            if len(uni)==100:break
        balanced=[]
        for fam in sorted(set(map(family,cols))):
            for c in [n for n in order if family(n)==fam]:
                if accept(c,balanced):balanced.append(c)
                if sum(family(k)==fam for k in balanced)>=12:break
        for c in order:
            if len(balanced)>=100:break
            if c not in balanced and accept(c,balanced):balanced.append(c)
        views[str(fold)]={"top100":uni+CATS,"family_balanced100":balanced+CATS,
                           "all_features":cols+CATS,"train_rows":len(ti),"validation_rows":len(vi)}
        selected_sets.append(set(uni))
        for view,names in [("all",cols),("top100",uni),("balanced100",balanced)]:
            for fam in sorted(set(map(family,cols))):
                chosen=[c for c in names if family(c)==fam]
                thin=f[vi]["has_history"].to_numpy()==0
                populated=sum(np.isfinite(f[vi][c].cast(pl.Float64).to_numpy()[thin]).mean()>.5 for c in chosen)
                coverage.append(dict(fold=fold,view=view,family=fam,n_features=len(chosen),thin_file_populated=populated))
    save_table(coverage,"06_feature_family_coverage")
    stability=[]
    for i in range(5):
        for j in range(i+1,5):
            stability.append(dict(fold_a=i,fold_b=j,jaccard=len(selected_sets[i]&selected_sets[j])/len(selected_sets[i]|selected_sets[j])))
    save_table(stability,"06_feature_selection_stability")
    (OUT/"fm_feature_views.json").write_text(json.dumps(views,indent=2)+"\n")
    print("100-feature views are controlled remote ablations, not demonstrated improvements. Preserve native categoricals/NaN for FMs.")
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    d=pl.DataFrame(coverage).filter(pl.col("fold")==0)
    fams=sorted(set(map(family,cols)));x=np.arange(len(fams))
    for k,v in enumerate(["all","top100","balanced100"]):
        dd=d.filter(pl.col("view")==v).sort("family")
        ax[0].bar(x+(k-1)*.25,dd["n_features"],width=.25,label=v)
        ax[1].bar(x+(k-1)*.25,dd["thin_file_populated"],width=.25,label=v)
    for a in ax:a.set_xticks(x,fams,rotation=35);a.legend()
    ax[0].set_title("Feature budget by source family")
    ax[1].set_title("Features populated in >50% of thin-file rows")
    figsave(fig,"06_fm_feature_budget")
    save_json({"models_run":0,"notebook_runs":0,
        "TabPFN_Rel":"DFS + TabPFN-3.5; benchmark temporal task API does not directly match random customer split with provided labels. See relational/schema.json before remote adapter work.",
        "TabPFN_3_5":"V3 full 56k context still behind trees; compare all/family-balanced/top100 features before adding context tricks.",
        "Causilo":"Native mixed-type/NaN handling and optional KV cache. Preserve context and log actual context rows; old capped run is not a full-context benchmark.",
        "EXAONE":"21M parameters do not imply tiny context memory. CUDA memory planner supports an explicit byte budget; benchmark same fold/context remotely.",
        "Mitra_v2":"Published fine-tuning recipe differs from stock AutoGluon defaults; record recipe and folds. No local model execution.",
        "RT_Plurel":"Relational system benchmark result is not plug-and-play proof for this snapshot dataset; defer until cutoff adapter passes.",
        "rule_status":"External datasets are prohibited; supplied rules do not explicitly settle every pretrained-weight/API policy. Do not upload competition rows to hosted APIs without authorization.",
        "sources":{"TabPFN_Rel":"https://github.com/PriorLabs/tabpfn-rel",
                   "RPI":"https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md",
                   "TabPFN":"https://github.com/PriorLabs/TabPFN","Causilo":"https://github.com/nums-ai/causilo",
                   "EXAONE":"https://github.com/LGAI-Research/EXAONE-Tabular",
                   "Mitra_v2":"https://huggingface.co/autogluon/mitra-classifier-2",
                   "RT_Plurel":"https://github.com/PriorLabs/relarena"},
        "verified_date":"2026-10-13"},"06_model_research")

if __name__=="__main__":
    self_check()
    main()
