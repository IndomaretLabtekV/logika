"""Verify fold-local preprocessing against destructive alternatives; export raw and fold-0 NN inputs."""
from deep_common import *
from scipy.special import ndtri

class NumericPreprocessor:
    """Train-only numeric transforms, with separate missing flags and no supervised selection."""
    def __init__(self, mode="interpolated"):
        if mode not in ("interpolated","step","robust"):
            raise ValueError(mode)
        self.mode=mode

    def fit(self,X):
        X=np.asarray(X,float)
        self.width=X.shape[1]
        self.refs=[np.sort(X[np.isfinite(X[:,j]),j]) for j in range(self.width)]
        self.indicators=np.flatnonzero((~np.isfinite(X)).any(0))
        # Keep per-column masks: equal missing patterns on fit can diverge on new customers.
        return self

    def transform(self,X):
        X=np.asarray(X,float)
        assert X.ndim==2 and X.shape[1]==self.width
        new_missing=np.setdiff1d(np.flatnonzero((~np.isfinite(X)).any(0)),self.indicators)
        if len(new_missing):
            raise ValueError(f"Missing values appeared in fit-complete columns: {new_missing.tolist()}")
        out=np.zeros_like(X,dtype=np.float32)
        for j,s in enumerate(self.refs):
            if not len(s):continue
            m=np.isfinite(X[:,j]); x=X[m,j]
            if self.mode=="robust":
                q1,med,q3=np.quantile(s,[.25,.5,.75])
                scale=q3-q1
                if scale<=0:scale=max(float(np.std(s)),1.)
                out[m,j]=np.arcsinh((x-med)/scale)
            else:
                if self.mode=="step":
                    u=(np.searchsorted(s,x,"left")+np.searchsorted(s,x,"right"))/(2*len(s))
                else:
                    vals,first,count=np.unique(s,return_index=True,return_counts=True)
                    probs=(first+.5*count)/len(s)
                    u=np.interp(x,vals,probs)
                out[m,j]=ndtri(np.clip(u,.5/len(s),1-.5/len(s)))
        mask=(~np.isfinite(X[:,self.indicators])).astype(np.float32)
        return np.column_stack([out,mask]).astype(np.float32)

def categorical_fit_transform(train,other,cols):
    vocabs={c:{v:i+1 for i,v in enumerate(sorted(train[c].drop_nulls().unique().to_list()))} for c in cols}
    def encode(frame):
        return np.column_stack([[vocabs[c].get(v,0) for v in frame[c].to_list()] for c in cols]).astype(np.int32)
    return encode(train),encode(other),vocabs

def checks():
    x=np.array([[0.,np.nan,1],[0,2,1],[2,np.nan,1],[4,3,1]])
    p=NumericPreprocessor().fit(x)
    out=p.transform(x)
    assert np.isfinite(out).all() and out[0,0]==out[1,0]
    assert np.array_equal(out[:,-1],[1,0,1,0])
    assert out[2,0]<out[3,0]
    before=p.transform([[1,2,1]])
    p.transform([[1e30,-1e30,1]])
    try:
        p.transform([[1,2,np.nan]])
    except ValueError:
        pass
    else:
        raise AssertionError("Unrepresented new missingness must fail explicitly")
    assert np.array_equal(before,p.transform([[1,2,1]]))
    # Missingness patterns coincident in fit must still be distinguishable later.
    q=NumericPreprocessor().fit([[np.nan,np.nan],[1,2],[3,4]])
    z=q.transform([[np.nan,2]])
    assert np.array_equal(z[0,-2:],[1,0])

def main():
    checks()
    f=features();tr=f.filter(pl.col("is_train")==1);te=f.filter(pl.col("is_train")==0)
    y=tr["target"].to_numpy()
    cols=sorted(c for c in f.columns if c not in IDS and c not in CATS)
    X=tr.select(cols).to_numpy().astype(float);XT=te.select(cols).to_numpy().astype(float)
    fid=folds(y);rows=[];snap=None;compact=[]
    for fold in range(5):
        ti=np.flatnonzero(fid!=fold);vi=np.flatnonzero(fid==fold)
        T,V=X[ti],X[vi]
        for mode in ("step","interpolated","robust"):
            prep=NumericPreprocessor(mode).fit(T)
            VT=prep.transform(V)
            assert np.isfinite(VT).all()
            for j,c in enumerate(cols):
                m=np.isfinite(V[:,j])
                if m.sum()<100 or len(np.unique(y[vi][m]))!=2:continue
                rawks=ks(y[vi][m],V[m,j]);newks=ks(y[vi][m],VT[m,j])
                unseen=len(np.unique(V[m,j]))-len(np.unique(VT[m,j]))
                rows.append(dict(fold=fold,mode=mode,feature=c,validation_KS_raw_observed=rawks,
                    validation_KS_after_observed=newks,delta=newks-rawks,distinct_values_lost=unseen))
            if fold==0 and mode=="interpolated":
                snap=(prep,ti,vi,VT)
        # Empirical missing-pattern compression is safe on this split only if checked.
        mt=~np.isfinite(T);mv=~np.isfinite(V);mx=~np.isfinite(XT)
        groups={}
        for j in range(len(cols)):
            if mt[:,j].any():groups.setdefault(np.packbits(mt[:,j]).tobytes(),[]).append(j)
        divergence=sum(not np.all(mv[:,js]==mv[:,js[0],None]) or not np.all(mx[:,js]==mx[:,js[0],None]) for js in groups.values())
        compact.append(dict(fold=fold,missing_columns=int(mt.any(0).sum()),fit_mask_groups=len(groups),
                            divergent_groups_validation_or_test=divergence,
                            new_missing_columns_validation=int((mv.any(0)&~mt.any(0)).sum()),
                            new_missing_columns_test=int((mx.any(0)&~mt.any(0)).sum())))
    metrics=save_table(rows,"04_transform_validation")
    save_table(compact,"04_missing_mask_audit")
    summary=metrics.group_by("mode").agg(max_abs_KS_change=pl.col("delta").abs().max(),
        median_abs_KS_change=pl.col("delta").abs().median(),max_values_lost=pl.col("distinct_values_lost").max()).sort("mode")
    save_table(summary,"04_transform_summary")
    prep,ti,vi,VT=snap
    NTR,NTE=prep.transform(X[ti]),prep.transform(XT)
    CT,CO,vocab=categorical_fit_transform(tr[ti],pl.concat([tr[vi],te],how="diagonal_relaxed"),CATS)
    assert np.isfinite(NTR).all() and np.isfinite(NTE).all()
    assert np.array_equal(VT,prep.transform(X[vi]))
    outdir=OUT/"preprocessed";outdir.mkdir(exist_ok=True)
    # Raw/NaN data are the correct neutral starting point for trees and model-native FM preprocessing.
    f.drop("target","is_train").write_parquet(outdir/"base_features_raw.parquet")
    tr.select("application_id","target").write_parquet(outdir/"train_labels.parquet")
    pl.DataFrame({"application_id":tr["application_id"],"audit_fold":fid}).write_parquet(outdir/"folds.parquet")
    np.savez_compressed(outdir/"nn_fold0.npz",X_train=NTR,X_valid=VT,X_test=NTE,
        cat_train=CT,cat_valid=CO[:len(vi)],cat_test=CO[len(vi):],y_train=y[ti],y_valid=y[vi],
        train_ids=tr[ti]["application_id"].to_numpy().astype(str),valid_ids=tr[vi]["application_id"].to_numpy().astype(str),
        test_ids=te["application_id"].to_numpy().astype(str))
    save_json(dict(numeric_columns=cols,missing_indicator_columns=[cols[j] for j in prep.indicators],
        categorical_vocab=vocab,fit_rows=len(ti),validation_rows=len(vi),
        output_numeric_width=NTR.shape[1],mode="interpolated",
        note="Only fold 0 arrays exported to limit laptop disk/RAM. Refit NumericPreprocessor on each training fold; never reuse across folds.",
        model_KS_after_preprocessing=None,
        reason="No model was trained or run. KS checks above test transform preservation on observed validation values only."),"04_preprocessing_manifest")
    examples=[]
    for cls in (0,1):
        for history in (0,1):
            ix=vi[(y[vi]==cls)&(tr[vi]["has_history"].to_numpy()==history)][:3]
            for i in ix:
                z=prep.transform(X[i:i+1])[0]
                row={"application_id":tr["application_id"][int(i)],"target":cls,"has_history":history}
                for c in ["rp_frac_dpd1","ec_accel_30d","bc_score_mean","ac_out_in_delta"]:
                    j=cols.index(c);row[c+"_raw"]=float(X[i,j]);row[c+"_transformed"]=float(z[j]);row[c+"_missing"]=not np.isfinite(X[i,j])
                examples.append(row)
    save_table(examples,"04_class_samples_before_after")
    # A naive fill=0 erases structural absence; indicators separate missing from a genuine zero.
    fig,ax=plt.subplots(2,4,figsize=(16,7))
    selected=["rp_frac_dpd1","ec_accel_30d","bc_score_mean","ac_out_in_delta"]
    for j,c in enumerate(selected):
        k=cols.index(c);m=np.isfinite(X[vi,k]);lo,hi=np.quantile(X[vi[m],k],[.01,.99])
        for cls in (0,1):
            sel=m&(y[vi]==cls)
            ax[0,j].hist(X[vi[sel],k],bins=35,range=(lo,hi),density=True,histtype="step",label=f"target {cls}")
            ax[1,j].hist(VT[sel,k],bins=35,density=True,histtype="step",label=f"target {cls}")
        ax[0,j].set_title(c+"\nObserved values; plot range p1-p99",fontsize=9)
        ax[1,j].set_title("Fold-local interpolation; observed only",fontsize=9)
        ax[0,j].legend();ax[1,j].legend()
    figsave(fig,"04_before_after_by_class")
    fig,ax=plt.subplots(figsize=(11,4))
    sample=np.array([[r[c+"_transformed"] for c in selected] for r in examples])
    im=ax.imshow(sample.T,aspect="auto",cmap="coolwarm",vmin=-3,vmax=3)
    ax.set_xticks(range(len(examples)),[f"y={r['target']}, hist={r['has_history']}" for r in examples],rotation=50)
    ax.set_yticks(range(4),selected);ax.set_title("Actual class samples after preprocessing; M = missing, not true zero")
    for i,r in enumerate(examples):
        for j,c in enumerate(selected):
            if r[c+"_missing"]:ax.text(i,j,"M",ha="center",va="center",color="black",fontweight="bold")
    fig.colorbar(im,ax=ax,shrink=.6);figsave(fig,"04_class_sample_heatmap")
    print("Preprocessing verified structurally; improvement in trained-model KS is UNKNOWN.")

if __name__=="__main__":
    self_check()
    main()
