"""Compare historical and selected samples on shared bins, with customer-aware eligibility."""
from shared8 import *

def main():
    _,tr,te,_,_,_,_,_,_=saved8();H=pl.read_parquet(ROOT/'data_pq/hist_feats.parquet');lab=hist();y=tr['target'].to_numpy()
    cols=['annual_interest_rate','bc_score_mean','pp_mean','ec_accel_30d','dti_declared','age','rp_frac_dpd1'];rows=[];coverage=[]
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for ax,c in zip(axes.ravel(),cols):
        x=tr[c].cast(pl.Float64).to_numpy();xx=H[c].cast(pl.Float64).to_numpy();cuts=np.unique(np.nanquantile(x,[.2,.4,.6,.8]))
        for tag,vals,yy in [('selected',x,y),('historical',xx,H['target'].to_numpy())]:
            bins=np.searchsorted(cuts,vals);bins[~np.isfinite(vals)]=len(cuts)+1
            rr=[]
            for j in range(len(cuts)+2):
                m=bins==j
                row=dict(feature=c,population=tag,bin=j,n=int(m.sum()),bad_rate=float(yy[m].mean()) if m.any() else np.nan)
                rows.append(row);rr.append(row['bad_rate'])
            line,=ax.plot(range(len(rr)-1),rr[:-1],'o-',label=tag)
            ax.scatter([len(rr)-1],[rr[-1]],marker='x',color=line.get_color())
            ax.set_xticks(range(len(rr)),[str(j) for j in range(len(rr)-1)]+['missing'])
            coverage.append(dict(feature=c,population=tag,coverage=float(np.isfinite(vals).mean())))
        ax.set_title(c+'; shared selected cutpoints');ax.legend(fontsize=7)
    term=lab.group_by('term_months').agg(rows=pl.len(),bad_rate=pl.col('hist_target').mean()).sort('term_months');axes.ravel()[-1].bar(term['term_months'].cast(pl.String),term['rows']);axes.ravel()[-1].set_title('Historical terms; selected are all 12m');figsave(fig,'02_domains')
    csv(rows,'02_shared_bin_rates');csv(coverage,'02_coverage');csv(term.to_pandas(),'02_terms')
    counts=lab.group_by('customer_id').len();csv(counts.group_by('len').len(name='customers').sort('len').to_pandas(),'02_customer_multiplicity')
    fd=folds(y,seed=81015);valid=tr.filter(pl.Series(fd==0)).select('customer_id');train=tr.filter(pl.Series(fd!=0)).select('customer_id')
    eligible=lab.with_columns(strict_fold0_train=pl.col('customer_id').is_in(train['customer_id'].implode()),validation_customer=pl.col('customer_id').is_in(valid['customer_id'].implode()),test_customer=pl.col('customer_id').is_in(te['customer_id'].implode()))
    eligible.select('application_id','customer_id','hist_target','strict_fold0_train','validation_customer','test_customer').write_parquet(OUT/'02_group_manifest.parquet')
    assert eligible.filter(pl.col('strict_fold0_train') & (pl.col('validation_customer')|pl.col('test_customer'))).height==0
    shares={c:int(eligible[c].sum()) for c in ['strict_fold0_train','validation_customer','test_customer']}
    js(dict(historical_rows=lab.height,unique_historical_customers=lab['customer_id'].n_unique(),main_customers=tr.height,test_customers=te.height,group_manifest_seed=81015,**shares,unknown_historical_purpose=int(H['loan_purpose'].null_count()),historical_duplicate_application_ids=lab.height-lab['application_id'].n_unique(),warning='V8 adds all history rows in each fold, including histories of validation and test customers. These are not selected-loan target labels, but the evaluation is transductive rather than unseen-customer training. Compare protocols explicitly.'),'02_summary')

if __name__=='__main__':main()
