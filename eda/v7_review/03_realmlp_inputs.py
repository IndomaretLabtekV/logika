"""Fold-local median + per-feature missing flags, ready for native model scaling."""
from shared7 import *

def main():
    f,tr,te,_,_=saved7();y=tr['target'].to_numpy();fd=folds(y,seed=71015)
    cols=[c for c in f.columns if c not in IDS and c not in CATS]
    X=f.select(cols).to_numpy().astype(float)
    train_ids=tr.filter(pl.Series(fd!=0)).select('application_id')
    fit=f['application_id'].is_in(train_ids['application_id'].implode()).to_numpy()
    med=np.array([np.median(X[fit & np.isfinite(X[:,j]),j]) if np.isfinite(X[fit,j]).any() else 0. for j in range(len(cols))])
    # Retain one flag per input column, including previously unseen missingness.
    masks=~np.isfinite(X);Z=np.where(masks,med[None,:],X)
    assert np.isfinite(Z).all() and np.array_equal(Z[~masks],X[~masks])
    out=f.select('application_id').hstack(pl.DataFrame(Z,schema=cols)).hstack(pl.DataFrame(masks.astype(np.int8),schema=[c+'__missing' for c in cols]))
    vocab={};cats=[]
    for c in CATS:
        vocab[c]=sorted(f.filter(pl.Series(fit))[c].drop_nulls().unique().to_list())
        assert '__UNKNOWN__' not in vocab[c]
        cats.append(pl.when(pl.col(c).is_in(vocab[c])).then(pl.col(c)).otherwise(pl.lit('__UNKNOWN__')).alias(c))
    out=out.hstack(f.select(cats));assert out.height==100000
    out.write_parquet(OUT/'03_median_inputs_fold0.parquet')
    assignment=tr.select('application_id').with_columns(audit_fold=pl.Series(fd),target=tr['target']);assignment.write_parquet(OUT/'03_training_manifest.parquet')
    js(dict(seed=71015,fit_excludes_fold=0,fit_rows=int(fit.sum()),numeric_columns=cols,medians=med.tolist(),categorical_vocab=vocab,output_predictors=out.width-1,model_KS=None,warning='Only a fold-0 preprocessing demonstration. Recompute medians/vocabulary inside every actual model fold. Keep native RealMLP scaling; do not additionally apply the TabM quantile transform by default.'),'03_preprocessor')
    rows=[]
    for j,c in enumerate(cols):
        obs=X[np.isfinite(X[:,j]),j]
        rows.append(dict(feature=c,missing_fraction=float(masks[:,j].mean()),unique_float64=len(np.unique(obs)),unique_float32=len(np.unique(obs.astype(np.float32)))))
    audit=csv(rows,'03_precision_and_missingness')
    sample_ids=tr.group_by('target').head(6).select('application_id','target')
    chosen=['pp_mean','bc_score_mean','rp_frac_dpd1','ac_out_in_delta']
    samples=sample_ids.join(out.select('application_id',*chosen,*[c+'__missing' for c in chosen]),on='application_id',validate='1:1')
    csv(samples.to_pandas(),'03_class_samples')
    fig,ax=plt.subplots(1,3,figsize=(15,4));m=audit.sort_values('missing_fraction',ascending=False).head(12);ax[0].barh(m.feature,m.missing_fraction);ax[0].set_title('Missingness remains explicitly represented')
    for label in [0,1]:
        ids=tr.filter(pl.col('target')==label).select('application_id');v=out.join(ids,on='application_id',how='semi')['rp_frac_dpd1'].to_numpy();ax[1].hist(v,bins=40,density=True,histtype='step',label=f'y={label}')
    ax[1].legend();ax[1].set_title('Median-imputed DPD; flag also required')
    s=samples.to_pandas();im=ax[2].imshow(s[[c+'__missing' for c in chosen]].T,aspect='auto',vmin=0,vmax=1);ax[2].set_yticks(range(4),chosen);ax[2].set_xticks(range(len(s)),s.target);ax[2].set(xlabel='Sample target',title='Missing flags, actual class samples');fig.colorbar(im,ax=ax[2]);figsave(fig,'03_preprocessing')
    js(dict(all_observed_values_unchanged=True,nonfinite_outputs=0,fit_customers=int(fit.sum()),validation_or_test_new_missing_columns=int((masks[~fit].any(0)&~masks[fit].any(0)).sum()),duplicate_raw_predictor_rows=f.select([c for c in f.columns if c not in IDS]).is_duplicated().sum(),train_test_customer_overlap=tr.select('customer_id').join(te.select('customer_id'),on='customer_id').height),'03_checks')

if __name__=='__main__':main()
