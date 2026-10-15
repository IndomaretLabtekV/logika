"""Use mature historical outcomes as selected-cutoff features, not reconstructed training snapshots."""
from shared8 import *

def main():
    f,tr,te,_,_,_,_,_,_=saved8();h=hist()
    assert h.filter(pl.col('decision_datetime').dt.offset_by('12mo')>pl.col('sel_cutoff')).height==0
    # Preserve the existing due+90 proxy definition; do not claim it reconstructs a full payment ledger.
    h=h.with_columns(outcome_age=(pl.col('sel_cutoff')-pl.col('decision_datetime').dt.offset_by('12mo')).dt.total_days())
    a=h.group_by('customer_id').agg(mature_n=pl.len(),mature_bad_n=pl.col('hist_target').sum(),mature_bad_fraction=pl.col('hist_target').mean(),mature_last_bad=pl.col('hist_target').sort_by('decision_datetime','application_id').last(),mature_outcome_age=pl.col('outcome_age').min(),mature_bad_age=pl.col('outcome_age').filter(pl.col('hist_target')==1).min(),mature_12m_n=(pl.col('term_months')==12).sum(),mature_12m_bad_fraction=pl.col('hist_target').filter(pl.col('term_months')==12).mean())
    out=f.select('application_id','customer_id').join(a,on='customer_id',how='left',validate='1:1',maintain_order='left').with_columns(pl.col('mature_n','mature_bad_n','mature_12m_n').fill_null(0),has_mature_history=pl.col('mature_n').is_not_null().cast(pl.Int8))
    assert out.height==f.height and out['application_id'].n_unique()==f.height
    assert out.filter((pl.col('mature_n')==0)&pl.col('mature_bad_fraction').is_not_null()).height==0
    out.write_parquet(OUT/'03_selected_history_features.parquet')
    z=screen(out.drop('customer_id'),'03_conditional_screen')
    q=tr.select('application_id','target','has_history','rp_frac_dpd1').join(out.drop('customer_id'),on='application_id',validate='1:1',maintain_order='left').to_pandas()
    csv(q.groupby(['target','has_mature_history']).head(5),'03_class_samples')
    rates=q.groupby('mature_bad_n').agg(n=('target','size'),bad_rate=('target','mean')).reset_index();csv(rates,'03_selected_bad_rates')
    counts=q.groupby('has_mature_history').size();print('History coverage',counts.to_dict(),flush=True)
    corr=q[['rp_frac_dpd1','mature_bad_fraction','mature_bad_n','mature_n','mature_last_bad']].corr(method='spearman');csv(corr.reset_index(),'03_correlation')
    fig,ax=plt.subplots(1,3,figsize=(16,4));ax[0].bar(counts.index.astype(str),counts);ax[0].set(title='Observed mature history',xlabel='has_mature_history')
    for label in [0,1]:ax[1].hist(q.loc[q.target==label,'mature_bad_fraction'].dropna(),bins=20,density=True,histtype='step',label=f'y={label}')
    ax[1].legend();ax[1].set_title('Observed-only historical bad fraction')
    im=ax[2].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax[2].set_xticks(range(len(corr)),corr.columns,rotation=90);ax[2].set_yticks(range(len(corr)),corr.columns);fig.colorbar(im,ax=ax[2]);figsave(fig,'03_history_features')
    js(dict(rows=out.height,predictors=out.width-2,replicating_candidates=int(z.replicates.sum()),missing_rates_kept_null=True,warning='Features known at selected cutoff using provided past-loan proxy; no historical income/state reconstruction and no additional training rows. New-model validation KS remains unknown. Screening is exploratory after repeated label use.'),'03_manifest')

if __name__=='__main__':main()
