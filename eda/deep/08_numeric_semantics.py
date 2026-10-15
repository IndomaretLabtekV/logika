"""Numerical representation audit: monotone transforms can lose float fingerprints, not necessarily economic information."""
from deep_common import *
import importlib.util

def canonicalize(frame):
    cols=["pv_rate_delta","bz_inq_external","bc_score_std"]
    out=frame.with_columns([pl.col(c).round(8).alias(c) for c in cols])
    # Make pricing context explicit instead of relying on floating-point subtraction noise.
    return out.with_columns(rate_transition=pl.when(pl.col("pv_rate_last").is_not_null()).then(
        (pl.col("pv_rate_last")*100).round(0).cast(pl.Int32).cast(pl.String)+"->"+
        (pl.col("annual_interest_rate")*100).round(0).cast(pl.Int32).cast(pl.String)
    ).otherwise(pl.lit("no_previous_approved_rate")))

def main():
    f=features();tr=f.filter(pl.col("is_train")==1);y=tr["target"].to_numpy()
    sem=canonicalize(f);s=sem.filter(pl.col("is_train")==1)
    spec=importlib.util.spec_from_file_location("preprocessing",HERE/"04_preprocessing.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for c in ["pv_rate_delta","bz_inq_external","bc_score_std"]:
        x=tr[c].to_numpy();z=s[c].to_numpy();m=np.isfinite(x)
        assert np.max(np.abs(x[m]-z[m]))<1e-8
        rows.append(dict(feature=c,raw_unique=len(np.unique(x[m])),semantic_unique=len(np.unique(z[m])),
                         KS_raw_observed=ks(y[m],x[m]),KS_semantic_observed=ks(y[m],z[m]),
                         max_absolute_rounding_change=float(np.max(np.abs(x[m]-z[m])))))
    save_table(rows,"08_precision_summary")
    detail=tr.select("pv_rate_delta","annual_interest_rate","pv_rate_last","target").with_columns(
        canonical_delta=pl.col("pv_rate_delta").round(8),
        raw_repr=pl.col("pv_rate_delta").map_elements(repr,return_dtype=pl.String))
    groups=detail.group_by("canonical_delta","raw_repr","annual_interest_rate","pv_rate_last").agg(
        n=pl.len(),bad_rate=pl.col("target").mean()).sort("canonical_delta","raw_repr")
    save_table(groups,"08_price_subtraction_groups")
    sample=groups.filter(pl.col("canonical_delta")==.04)
    save_table(sample,"08_same_delta_different_prices")
    # Check the economic delta exactly matches the rate-tier difference in this dataset.
    m=tr["pv_rate_last"].is_not_null().to_numpy()
    expected=((tr["annual_interest_rate"].to_numpy()[m]*100).round()-(tr["pv_rate_last"].to_numpy()[m]*100).round())/100
    assert np.allclose(expected,s["pv_rate_delta"].to_numpy()[m],atol=1e-12)
    sem.drop("target","is_train").write_parquet(OUT/"preprocessed/semantic_feature_view.parquet")
    # Export a second explicitly marked ablation, using exactly the same train/validation row IDs.
    num=sorted(c for c in f.columns if c not in IDS and c not in CATS)
    cate=CATS+["rate_transition"]
    fid=folds(y);ti=np.flatnonzero(fid!=0);vi=np.flatnonzero(fid==0)
    train=sem.filter(pl.col("is_train")==1);test=sem.filter(pl.col("is_train")==0)
    X=train.select(num).to_numpy().astype(float);XT=test.select(num).to_numpy().astype(float)
    pp=module.NumericPreprocessor("interpolated").fit(X[ti])
    cv,co,vocab=module.categorical_fit_transform(train[ti],pl.concat([train[vi],test],how="diagonal_relaxed"),cate)
    np.savez_compressed(OUT/"preprocessed/nn_fold0_semantic.npz",
        X_train=pp.transform(X[ti]),X_valid=pp.transform(X[vi]),X_test=pp.transform(XT),
        cat_train=cv,cat_valid=co[:len(vi)],cat_test=co[len(vi):],y_train=y[ti],y_valid=y[vi],
        train_ids=train[ti]["application_id"].to_numpy().astype(str),
        valid_ids=train[vi]["application_id"].to_numpy().astype(str),test_ids=test["application_id"].to_numpy().astype(str))
    # The pipeline must be invariant to row ordering and input batches.
    j=np.random.default_rng(0).permutation(len(vi))[:100]
    assert np.array_equal(pp.transform(X[vi[j]]),pp.transform(X[vi])[j])
    validation=[]
    for c in ["pv_rate_delta","bz_inq_external","bc_score_std"]:
        jj=num.index(c);v=X[vi,jj];ok=np.isfinite(v)
        z=pp.transform(X[vi])[:,jj]
        validation.append(dict(feature=c,raw_semantic_KS=ks(y[vi][ok],v[ok]),
                               transformed_semantic_KS=ks(y[vi][ok],z[ok])))
    save_table(validation,"08_semantic_transform_check")
    save_json(dict(numeric_columns=num,categorical_columns=cate,categorical_vocab=vocab,
        missing_indicator_columns=[num[j] for j in pp.indicators],
        status="Separate untrained ablation. Do not claim model KS improvement.",
        conclusion="Different binary floats for one economic delta encode original price tiers. Existing parent rate columns already contain that information; explicit transition is interpretable.",
        comparison="04 raw/interpolated preserves numerical ordering; 08 canonicalized/interpolated removes insignificant decimal distinctions and preserves pricing context explicitly.",
        model_validation_KS=None),"08_semantic_manifest")
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    d=sample.sort("pv_rate_last")
    ax[0].bar(range(d.height),d["bad_rate"])
    ax[0].set_xticks(range(d.height),[f"{a:.0%}->{b:.0%}" for a,b in d.select("pv_rate_last","annual_interest_rate").iter_rows()])
    ax[0].set_title("Same +4pp interest change, different price tiers")
    ax[0].set_ylabel("Observed default rate")
    for i,(rate,n) in enumerate(d.select("bad_rate","n").iter_rows()):ax[0].text(i,rate,f"n={n}",ha="center",va="bottom")
    ax[1].bar(np.arange(3)-.18,[r["KS_raw_observed"] for r in rows],width=.36,label="binary float")
    ax[1].bar(np.arange(3)+.18,[r["KS_semantic_observed"] for r in rows],width=.36,label="canonical decimal")
    ax[1].set_xticks(range(3),[r["feature"] for r in rows],rotation=20)
    ax[1].set_title("Feature KS can change without new financial information");ax[1].legend()
    figsave(fig,"08_numeric_semantics")

if __name__=="__main__":
    self_check()
    main()
