"""Measure finite-history reliability and test label-free price summaries."""
from shared7 import *
from shared import items
import importlib.util

def main():
    f,tr,te,p,_=saved7();y=p['target'].to_numpy();fold=folds(y,seed=71015)
    spec=importlib.util.spec_from_file_location('price',ROOT/'eda/v5_review/01_price_preprocessing.py');price=importlib.util.module_from_spec(spec);spec.loader.exec_module(price)
    e=items(f);fit_ids=tr.filter(pl.Series(fold!=0)).select('customer_id');ref=price.reference(e,fit_ids);z=price.apply_reference(e,ref)
    # Hash only assigns orders to disjoint diagnostic halves; opaque ID never becomes a predictor.
    z=z.with_columns(part=(pl.col('order_id').hash(seed=71015)%2).cast(pl.Int8))
    o=z.group_by('customer_id','order_id').agg(pl.col('part').first(),pl.col('z').mean(),pl.col('age').first())
    a=o.group_by('customer_id').agg(pl.len().alias('price_orders'),pl.col('z').mean().alias('order_price_mean'),pl.col('z').std().alias('order_price_sd'),pl.col('z').median().alias('order_price_median'),pl.col('z').filter(pl.col('part')==0).mean().alias('half0'),pl.col('z').filter(pl.col('part')==1).mean().alias('half1'))
    a=a.with_columns(price_se=pl.col('order_price_sd')/pl.col('price_orders').sqrt(),shrunk_price_10=pl.col('order_price_mean')*pl.col('price_orders')/(pl.col('price_orders')+10))
    x=f.select('application_id','customer_id').join(a,on='customer_id',validate='1:1',maintain_order='left')
    assert x.height==f.height and x['application_id'].n_unique()==f.height
    x.drop('half0','half1').write_parquet(OUT/'02_price_candidates.parquet');ref.write_parquet(OUT/'02_reference_fold0.parquet')
    q=tr.select('application_id','target','pp_mean').join(x.drop('customer_id'),on='application_id',validate='1:1',maintain_order='left').to_pandas()
    rows=[]
    for lo,hi in [(0,15),(15,30),(30,60),(60,10000)]:
        m=(q.price_orders>lo)&(q.price_orders<=hi);u=q[m]
        rows.append(dict(orders_from_exclusive=lo,orders_to=hi,n=len(u),split_half_rho=u.half0.corr(u.half1,method='spearman'),median_half_disagreement=(u.half0-u.half1).abs().median(),median_se=u.price_se.median()))
    reliability=csv(rows,'02_measurement_reliability');csv(q.groupby('target').head(12),'02_class_samples')
    cols=['order_price_mean','order_price_sd','order_price_median','price_orders','price_se','shrunk_price_10']
    screen(x.select('application_id',*cols),'02_conditional_screen')
    val=fold==0
    csv([dict(feature=c,validation_rows=int(val.sum()),feature_KS=ks(y[val],q[c].to_numpy()[val])) for c in ['pp_mean',*cols]],'02_feature_KS')
    fig,ax=plt.subplots(1,3,figsize=(15,4));sample=q.iloc[::30];ax[0].scatter(sample.half0,sample.half1,s=4,alpha=.3);ax[0].set(xlabel='Order-half 0 mean',ylabel='Order-half 1 mean')
    ax[1].plot(reliability.orders_from_exclusive,reliability.median_half_disagreement,'o-');ax[1].set(xlabel='Order-count group lower bound',ylabel='Median split-half difference')
    for label in [0,1]:
        for c,style in [('order_price_mean','--'),('shrunk_price_10','-')]:ax[2].hist(q.loc[(y==label)&val,c],bins=40,histtype='step',density=True,linestyle=style,label=f'{c}, y={label}')
    ax[2].legend(fontsize=7);ax[2].set_title('Actual before/after fixed shrinkage');figsave(fig,'02_price_measurement')
    js(dict(reference_fold_seed=71015,reference_excludes_fold=0,reference_fit_customers=fit_ids.height,rows=x.height,warning='Split-half disagreement includes real within-customer heterogeneity, not just noise. Fixed shrinkage strength 10 is a hypothesis, not an optimized posterior. Refit reference in each actual model fold.'),'02_manifest')

if __name__=='__main__':main()
