"""Point-in-time, label-free candidate preprocessing; no estimator fitting."""
from review_common import *

def main():
    f,tr,te=data()
    cut=f.select('customer_id','application_id',cutoff='decision_datetime')
    r=table('repayment_history').join(cut,on='customer_id',validate='m:1').with_columns(age=(pl.col('cutoff')-pl.col('due_date')).dt.total_days())
    r=r.filter(pl.col('age')>=0)
    rows=[]; expr=[]
    for d in [30,60,90]:
        eligible=pl.col('age')>=d; late=pl.col('max_days_past_due')>=d
        rows.append(dict(threshold=d,rows=r.height,eligible=r.filter(eligible).height,late=r.filter(late).height,impossible_late=r.filter(late & ~eligible).height))
        # Snapshot DPD is never assigned to an earlier observation date.
        expr.extend([eligible.sum().alias(f'exposure_{d}'),(late & eligible).sum().alias(f'late_{d}'),late.mean().alias(f'raw_rate_{d}')])
    a=r.group_by('customer_id').agg(expr)
    for d in [30,60,90]:
        a=a.with_columns(pl.when(pl.col(f'exposure_{d}')>0).then(pl.col(f'late_{d}')/pl.col(f'exposure_{d}')).alias(f'mature_rate_{d}'))
        assert a.filter(pl.col(f'late_{d}')>pl.col(f'exposure_{d}')).height==0
    b=table('bureau_observations').join(cut,on='customer_id',suffix='_selected',validate='m:1').filter(pl.col('available_datetime')<=pl.col('cutoff'))
    keys=['customer_id','bureau_provider']
    current=b.filter(pl.col('application_id')==pl.col('application_id_selected')).filter(pl.col('score').is_not_null()).sort('report_datetime','bureau_observation_id').group_by(keys).agg(pl.col('score').last().alias('current_score'),pl.col('report_datetime').last().alias('current_date'))
    hist=b.filter(pl.col('application_id')!=pl.col('application_id_selected')).filter(pl.col('score').is_not_null()).sort('report_datetime','bureau_observation_id').group_by(keys).agg(pl.col('score').last().alias('past_score'),pl.col('report_datetime').last().alias('past_date'))
    pairs=current.join(hist,on=keys,validate='1:1').filter(pl.col('past_date')<pl.col('current_date')).with_columns(delta=pl.col('current_score')-pl.col('past_score'))
    c=pairs.group_by('customer_id').agg(pl.col('delta').mean().alias('matched_bureau_delta'),pl.col('delta').std().alias('matched_bureau_delta_sd'),pl.len().alias('matched_providers'))
    out=cut.select('application_id','customer_id').join(a,on='customer_id',how='left',validate='1:1').join(c,on='customer_id',how='left',validate='1:1')
    assert out.height==f.height and out['application_id'].n_unique()==f.height
    out.write_parquet(OUT/'01_candidates.parquet')
    csv(rows,'01_opportunity_constraints')
    csv(pairs.group_by('bureau_provider').agg(pl.len(),pl.col('delta').mean(),pl.col('delta').std().alias('sd')).to_pandas(),'01_provider_pairs')
    x=tr.select('application_id','target','bh_score_delta').join(out.drop('customer_id'),on='application_id',validate='1:1',maintain_order='left').to_pandas()
    csv(x.groupby('target').head(10),'01_class_samples')
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for y in [0,1]:
        z=x[x.target==y]
        for name,style in [('raw_rate_90','--'),('mature_rate_90','-')]:
            ax[0].hist(z[name].dropna(),bins=30,density=True,histtype='step',linestyle=style,label=f'{name}, y={y}')
    ax[0].legend(fontsize=7);ax[0].set_title('Opportunity correction: before / after')
    ax[1].scatter(x.bh_score_delta[::20],x.matched_bureau_delta[::20],s=3,alpha=.3);ax[1].set(xlabel='Current mean minus last single score',ylabel='Mean within-provider change')
    cols=['raw_rate_90','mature_rate_90','exposure_90','bh_score_delta','matched_bureau_delta']
    corr=x[cols].corr(method='spearman');csv(corr.reset_index(),'01_correlation')
    ax[2].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax[2].set_xticks(range(5),cols,rotation=90);ax[2].set_yticks(range(5),cols)
    figsave(fig,'01_preprocessing')
    csv(x[cols].describe().T.reset_index(),'01_feature_summary')
    screen(out.drop('customer_id'),'01_conditional_screen')
    js(dict(rows=out.height,features=out.width-2,duplicate_installment_ids=r.height-r['installment_id'].n_unique(),train_test_customer_overlap=tr.select('customer_id').join(te.select('customer_id'),on='customer_id').height,missing_exposure_kept_null=True,warning='Candidates only. No new-model validation KS; conditional tests reuse selected OOF and are exploratory.'),'01_manifest')

if __name__=='__main__':main()
