"""Audit v6 expanding price reference against a training-customer-only reference."""
from review_common import *
from shared import items
import importlib.util

def main():
    spec=importlib.util.spec_from_file_location('price',ROOT/'eda/v5_review/01_price_preprocessing.py')
    price=importlib.util.module_from_spec(spec);spec.loader.exec_module(price)
    f,tr,te=data(); y=tr['target'].to_numpy(); fd=folds(y,seed=20261014)
    e=items(f);fit_ids=tr.filter(pl.Series(fd!=0)).select('customer_id')
    ref=price.reference(e,fit_ids);z=price.apply_reference(e,ref)
    a=z.group_by('customer_id').agg(pl.col('z').mean().alias('fold_price_mean'))
    out=f.select('application_id','customer_id','is_train','pp_mean','target').join(a,on='customer_id',validate='1:1',maintain_order='left')
    out.write_parquet(OUT/'02_price_fold0.parquet');ref.write_parquet(OUT/'02_reference_fold0.parquet')
    val=tr.filter(pl.Series(fd==0)).select('customer_id')
    injected=e.join(val,on='customer_id',how='semi').head(10).with_columns(lu=pl.lit(50.))
    assert ref.equals(price.reference(pl.concat([e,injected]),fit_ids))
    sample=z.row(0,named=True)
    eligible=e.join(fit_ids,on='customer_id',how='semi').filter((pl.col('product_category')==sample['product_category']) & (pl.col('event_month')<sample['query_month']))
    assert np.isclose(sample['mu'],eligible['lu'].mean()) and np.isclose(sample['sd'],eligible['lu'].std())
    x=out.filter(pl.col('is_train')==1).to_pandas();valid=fd==0
    old=x.pp_mean.to_numpy();new=x.fold_price_mean.to_numpy()
    js(dict(diagnostic_seed=20261014,validation_rows=int(valid.sum()),v6_univariate_KS=ks(y[valid],old[valid]),fold_local_univariate_KS=ks(y[valid],new[valid]),rank_correlation=float(pd.Series(old[valid]).corr(pd.Series(new[valid]),method='spearman')),median_abs_change=float(np.median(abs(old[valid]-new[valid]))),p99_abs_change=float(np.quantile(abs(old[valid]-new[valid]),.99)),held_customer_perturbation_passed=True,independent_reference_check_passed=True,warning='This is feature KS, not model validation. Diagnostic folds differ from notebook sklearn folds; recompute inside actual model folds.'),'02_summary')
    csv(x.groupby('target').head(12),'02_class_samples')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].scatter(old[valid][::5],new[valid][::5],s=3,alpha=.3);ax[0].set(xlabel='v6 all-customer past reference',ylabel='Training-customer past reference')
    for label in [0,1]:
        for values,style,name in [(old,'--','v6'),(new,'-','fold local')]:
            ax[1].hist(values[valid & (y==label)],bins=50,density=True,histtype='step',linestyle=style,label=f'{name}, y={label}')
    ax[1].legend();ax[1].set_title('Actual transformed validation distributions');figsave(fig,'02_price_before_after')

if __name__=='__main__':main()
