"""Add price composition, repayment shrinkage, and cash-flow coverage before decision."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def ratio(a,b):
    return a/b.where(b>0)


def main():
    base=pd.read_parquet(ROOT/'data/processed/application_features_extended.parquet').set_index('customer_id')
    cutoff=base[['decision_datetime']].rename(columns={'decision_datetime':'cutoff'})
    blocks=[]
    audits={}
    raw=ROOT/'data'

    repayments=pd.read_csv(raw/'repayment_history.csv',parse_dates=['due_date','last_payment_datetime'])
    repayments=repayments.merge(cutoff,left_on='customer_id',right_index=True,validate='many_to_one')
    audits['future_due_rows']=int(repayments.due_date.ge(repayments.cutoff).sum())
    audits['future_payment_rows']=int(repayments.last_payment_datetime.ge(repayments.cutoff).sum())
    assert audits['future_due_rows']==audits['future_payment_rows']==0
    repayments['age_days']=(repayments.cutoff-repayments.due_date).dt.total_seconds()/86400
    repayments['fully_paid']=repayments.amount_paid.ge(repayments.amount_due).astype(int)
    repayments['payment_delay']=(repayments.last_payment_datetime-repayments.due_date).dt.total_seconds()/86400
    repayments['unpaid_fraction']=ratio((repayments.amount_due-repayments.amount_paid).clip(lower=0),repayments.amount_due)
    for threshold in [1,7,30,60,90]:
        repayments[f'late_{threshold}']=repayments.max_days_past_due.ge(threshold).astype(int)
    features=[]
    training_ids=set(base.index[base.split.eq('train')])
    for window in [None,180,365]:
        part=repayments if window is None else repayments[repayments.age_days.le(window)]
        grouped=part.groupby('customer_id')
        data=grouped.agg(n=('loan_id','size'),fully_paid_share=('fully_paid','mean'),
                         unpaid_fraction_mean=('unpaid_fraction','mean'),unpaid_fraction_max=('unpaid_fraction','max'),
                         paid_delay_mean=('payment_delay','mean'),paid_delay_std=('payment_delay','std'))
        tag='all' if window is None else f'{window}d'
        for threshold in [1,7,30,60,90]:
            counts=grouped[f'late_{threshold}'].sum()
            observed=part[part.customer_id.isin(training_ids)][f'late_{threshold}'].mean()
            for strength in [6,24]:
                data[f'late_{threshold}_posterior_{strength}']=(counts+strength*observed)/(data.n+strength)
        data.columns=[f'v3_repay_{tag}_{col}' for col in data.columns]
        features.append(data)
    blocks.append(pd.concat(features,axis=1))
    print('Repayment shrinkage and censoring features complete.',flush=True)
    del repayments,features,data,part,grouped

    bureau=pd.read_parquet(ROOT/'data/processed/bureau_latest.parquet')
    bureau=bureau.merge(cutoff,left_on='customer_id',right_index=True,validate='many_to_one')
    bureau['age_days']=(bureau.cutoff-bureau.report_datetime).dt.total_seconds()/86400
    blocks_bureau=[]
    for half_life in [30,90]:
        weight=np.exp(-np.log(2)*bureau.age_days/half_life)
        bureau['weight']=weight.where(bureau.score.notna(),0)
        bureau['weighted_score']=bureau.score.fillna(0)*bureau.weight
        grouped=bureau.groupby('customer_id').agg(numerator=('weighted_score','sum'),denominator=('weight','sum'))
        blocks_bureau.append(ratio(grouped.numerator,grouped.denominator).rename(f'v3_bureau_score_recency_weighted_{half_life}d'))
    blocks.append(pd.concat(blocks_bureau,axis=1))
    del bureau,blocks_bureau,grouped,weight

    ecom=pd.read_csv(raw/'ecommerce_transactions.csv',parse_dates=['transaction_datetime'])
    ecom=ecom.merge(cutoff,left_on='customer_id',right_index=True,validate='many_to_one')
    audits['future_transaction_rows']=int(ecom.transaction_datetime.ge(ecom.cutoff).sum())
    assert audits['future_transaction_rows']==0
    ecom['age_days']=(ecom.cutoff-ecom.transaction_datetime).dt.total_seconds()/86400
    ecom['unit_price']=ratio(ecom.item_total,ecom.quantity)
    completed=ecom[ecom.order_status.eq('completed')]
    features=[]
    for window in [None,180]:
        part=completed if window is None else completed[completed.age_days.le(window)]
        tag='all' if window is None else '180d'
        summary=part.groupby(['customer_id','product_category']).agg(
            price_mean=('unit_price','mean'),price_median=('unit_price','median'),price_min=('unit_price','min'),
            quantity_mean=('quantity','mean'),item_count=('order_item_id','size'),
            distinct_orders=('order_id','nunique'),spend=('item_total','sum'),last_age=('age_days','min'))
        wide=summary.unstack('product_category')
        wide.columns=[f'v3_ecom_{tag}_{category}_{value}' for value,category in wide.columns]
        features.append(wide)
    shares=completed.groupby(['customer_id','product_category']).item_total.sum().unstack(fill_value=0)
    shares=shares.div(shares.sum(axis=1),axis=0)
    entropy=-(shares.replace(0,np.nan)*np.log(shares.replace(0,np.nan))).sum(axis=1)
    features.append(pd.DataFrame({'v3_ecom_category_entropy':entropy,'v3_ecom_category_concentration':(shares**2).sum(axis=1)}))
    orders=ecom.drop_duplicates('order_id').copy()
    orders['completed_spend']=orders.order_total.where(orders.order_status.eq('completed'),0)
    orders['completed']=orders.order_status.eq('completed').astype(int)
    orders['days_bucket']=(orders.age_days//30).astype(int)
    month=orders[orders.days_bucket.between(0,5)].groupby(['customer_id','days_bucket']).agg(
        spend=('completed_spend','sum'),orders=('completed','sum'))
    for value in ['spend','orders']:
        wide=month[value].unstack(fill_value=0).reindex(columns=range(6),fill_value=0)
        # No orders in a known lookback bucket are an observed zero.
        wide=wide.reindex(base.index,fill_value=0)
        wide.columns=[f'v3_ecom_month_back_{i}_{value}' for i in range(6)]
        features.append(wide)
        features.append(pd.DataFrame({f'v3_ecom_monthly_{value}_cv':ratio(wide.std(axis=1),wide.mean(axis=1)),
                                      f'v3_ecom_monthly_{value}_max_share':ratio(wide.max(axis=1),wide.sum(axis=1))}))
    blocks.append(pd.concat(features,axis=1))
    print('Category prices, purchase composition, and monthly variability complete.',flush=True)
    del ecom,completed,orders,features,part,summary,wide,shares,entropy,month

    extra=pd.DataFrame(index=base.index)
    income=base.v2_account_3m_income_like_credits_mean
    outflow=base.v2_account_3m_outflows_mean
    existing=base.v2_existing_active_installment
    installment=base.proposed_monthly_installment
    extra['v3_cash_surplus_before_new_installment']=income-outflow
    extra['v3_cash_surplus_after_new_installment']=income-outflow-installment
    extra['v3_cash_surplus_installment_cover']=ratio(income-outflow,installment)
    extra['v3_cash_surplus_installment_existing_cover']=ratio(income-outflow,installment+existing.fillna(0))
    extra['v3_min_balance_installment_cover']=ratio(base.v2_account_3m_minimum_balance_min,installment)
    extra['v3_balance_total_installment_cover']=ratio(base.v2_account_3m_average_balance_mean,installment+existing.fillna(0))
    extra['v3_income_after_installments_per_dependent']=(income-installment-existing.fillna(0))/(base.dependents+1)
    extra['v3_cash_deficit_monthly_income_share']=ratio((outflow+installment-income).clip(lower=0),income)
    blocks.append(extra)
    full=pd.concat([base]+blocks,axis=1)
    assert full.columns.is_unique and full.index.is_unique
    full.replace([np.inf,-np.inf],np.nan,inplace=True)
    output=ROOT/'data/processed/application_features_v3.parquet'
    full.reset_index().to_parquet(output,index=False)
    audits['new_columns']=len(full.columns)-len(base.columns)
    audits['training_labels_used_for_features']=False
    audits['repayment_priors']='Global historical late-event frequency among training customers only; no target labels.'
    (ROOT/'artifacts/optimization/v3_feature_audit.json').write_text(json.dumps(audits,indent=2))
    print(f'Saved {len(full):,} customers with {len(full.columns)+1} columns.',flush=True)


if __name__=='__main__':
    main()
