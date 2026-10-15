"""Audit synthetic constraints, duplicate identities and quantization without label reconstruction."""
from review_common import *
from shared import items

def main():
    f,tr,te=data();p=table('customer_profiles');a=table('account_behavior')
    residual=a['closing_balance']-a['opening_balance']-a['inflows']+a['outflows']
    e=items(f)
    o=e.group_by('order_id').agg(pl.col('order_total').first(),pl.col('item_total').sum(),pl.col('customer_id').n_unique().alias('customers'),pl.col('order_total').n_unique().alias('totals'))
    sig=[c for c in f.columns if c not in IDS]
    audit=dict(account_rows=a.height,account_key_duplicate_rows=a.select('customer_id','month').is_duplicated().sum(),account_balance_identity_failures=int((residual!=0).sum()),minimum_balance_zero_share=float((a['minimum_balance']==0).mean()),minimum_balance_negative_rows=int((a['minimum_balance']<0).sum()),profile_duplicate_content_rows=p.drop('customer_id').is_duplicated().sum(),feature_duplicate_content_rows=f.select(sig).is_duplicated().sum(),item_duplicate_key_rows=e.height-e['order_item_id'].n_unique(),repeated_item_content_rows=e.select('order_id','product_category','quantity','item_total').is_duplicated().sum(),order_total_mismatch_rows=o.filter(pl.col('order_total')!=pl.col('item_total')).height,order_inconsistent_owner_rows=o.filter(pl.col('customers')!=1).height,order_inconsistent_total_rows=o.filter(pl.col('totals')!=1).height,naive_spending_inflation=float(e['order_total'].sum()/o['order_total'].sum()))
    js(audit,'03_integrity')
    rows=[]
    for name,t,cols in [('profile',p,['declared_monthly_income']),('account',a,['opening_balance','closing_balance','inflows','outflows']),('commerce',e,['item_total','order_total'])]:
        for c in cols:
            x=t[c].to_numpy()
            rows.append(dict(table=name,column=c,unique=int(t[c].n_unique()),multiple_100_share=float((x%100==0).mean()),multiple_1000_share=float((x%1000==0).mean())))
    q=csv(rows,'03_quantization')
    # Aggregate order_total once per order, but preserve distinct legitimate item IDs.
    spend=o.join(e.select('order_id','customer_id').unique(),on='order_id',validate='1:1').group_by('customer_id').agg(pl.col('order_total').sum().alias('deduplicated_order_spend'),pl.len().alias('orders'))
    naive=e.group_by('customer_id').agg(pl.col('order_total').sum().alias('naive_item_row_spend'))
    out=f.select('application_id','customer_id','target').join(spend,on='customer_id',validate='1:1').join(naive,on='customer_id',validate='1:1')
    assert out.height==f.height
    out.drop('target').write_parquet(OUT/'03_order_spend.parquet')
    csv(out.filter(pl.col('target').is_not_null()).group_by('target').head(8).to_pandas(),'03_class_samples')
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].hist(residual.to_numpy(),bins=30);ax[0].set(title='Closing minus implied closing',xlabel='Currency residual')
    ax[1].bar(q.column,q.multiple_1000_share);ax[1].tick_params(axis='x',rotation=90);ax[1].set_title('Fraction on 1,000-unit grid')
    for label in [0,1]:
        s=out.filter(pl.col('target')==label)
        ax[2].hist((s['naive_item_row_spend']/s['deduplicated_order_spend']).to_numpy(),bins=40,density=True,histtype='step',label=f'y={label}')
    ax[2].legend();ax[2].set(title='Inflation from repeated order totals',xlabel='Naive / correct spend')
    figsave(fig,'03_integrity')

if __name__=='__main__':main()
