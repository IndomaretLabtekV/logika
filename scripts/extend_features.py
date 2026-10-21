"""Add debt burden, recent behavior, trends, and provider histories before decision."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'
PROCESSED=ROOT/'data/processed'


def ratio(a,b):
    return a/b.where(b>0)


def main():
    base=pd.read_parquet(PROCESSED/'application_features.parquet').set_index('customer_id')
    cutoff=base[['decision_datetime']].rename(columns={'decision_datetime':'cutoff'})
    parts=[]
    audit={}

    def attach(f):
        assert f.index.is_unique
        parts.append(f.reindex(base.index))
        print(f'Added {len(f.columns)} columns; total added {sum(len(p.columns) for p in parts)}',flush=True)

    def history(name,dates):
        df=pd.read_csv(RAW/f'{name}.csv',parse_dates=dates)
        df=df.merge(cutoff,left_on='customer_id',right_index=True,how='left',validate='many_to_one')
        assert df.cutoff.notna().all()
        for col in dates:
            audit[f'{name}_{col}_at_or_after_cutoff']=int(df[col].ge(df.cutoff).sum())
        return df

    def stats(df,cols,prefix,functions=('mean','std','min','max')):
        f=df.groupby('customer_id')[cols].agg(list(functions))
        f.columns=[f'{prefix}_{col}_{fn}' for col,fn in f.columns]
        return f

    # Installment-level patterns and loan-level outcomes observed before cutoff.
    rep=history('repayment_history',['due_date','last_payment_datetime'])
    assert rep.last_payment_datetime.isna().sum()+rep.last_payment_datetime.lt(rep.cutoff).sum()==len(rep)
    rep=rep[rep.due_date.lt(rep.cutoff)].copy()
    rep['age_days']=(rep.cutoff-rep.due_date).dt.total_seconds()/86400
    rep['payment_fraction']=ratio(rep.amount_paid,rep.amount_due)
    rep['unpaid']=(rep.amount_due-rep.amount_paid).clip(lower=0)
    rep['unpaid_fraction']=ratio(rep.unpaid,rep.amount_due)
    rep['partial']=rep.payment_fraction.between(1e-9,.999999).astype(int)
    rep['unpaid_flag']=rep.amount_paid.eq(0).astype(int)
    rep['late']=rep.max_days_past_due.gt(0).astype(int)
    rep['dpd30']=rep.max_days_past_due.ge(30).astype(int)
    rep['dpd60']=rep.max_days_past_due.ge(60).astype(int)
    rep['dpd90']=rep.max_days_past_due.ge(90).astype(int)
    rep['payment_delay_days']=(rep.last_payment_datetime-rep.due_date).dt.total_seconds()/86400
    rep['weighted_dpd']=rep.max_days_past_due*rep.amount_due
    feature=stats(rep,['days_past_due','max_days_past_due','payment_fraction','unpaid_fraction','payment_delay_days'],'v2_repay')
    for days in [90,180,365]:
        subset=rep[rep.age_days.le(days)]
        f=subset.groupby('customer_id').agg(
            n=('loan_id','size'),loans=('loan_id','nunique'),due=('amount_due','sum'),paid=('amount_paid','sum'),
            unpaid=('unpaid','sum'),dpd_mean=('max_days_past_due','mean'),dpd_max=('max_days_past_due','max'),
            current_dpd_max=('days_past_due','max'),late_share=('late','mean'),dpd30_share=('dpd30','mean'),
            dpd60_share=('dpd60','mean'),dpd90_share=('dpd90','mean'),partial_share=('partial','mean'),
            completely_unpaid_share=('unpaid_flag','mean'),weighted_dpd=('weighted_dpd','sum'))
        f['paid_ratio']=ratio(f.paid,f.due)
        f['weighted_dpd']=ratio(f.weighted_dpd,f.due)
        f=f.add_prefix(f'v2_repay_{days}d_')
        feature=feature.join(f)
    latest_due=rep.sort_values('due_date').groupby('customer_id').tail(1).set_index('customer_id')
    feature=feature.join(latest_due[['max_days_past_due','days_past_due','payment_fraction','unpaid','age_days']].add_prefix('v2_repay_latest_'))
    late_last=rep[rep.late.eq(1)].groupby('customer_id').age_days.min()
    feature['v2_repay_last_late_age_days']=late_last
    feature['v2_repay_late_recency_missing']=late_last.reindex(feature.index).isna().astype(int)
    loans=rep.groupby(['loan_id','customer_id']).agg(
        due_count=('due_date','size'),due_sum=('amount_due','sum'),paid_sum=('amount_paid','sum'),
        unpaid_sum=('unpaid','sum'),max_historical_dpd=('max_days_past_due','max'),
        current_dpd=('days_past_due','max'),mean_historical_dpd=('max_days_past_due','mean'),
        last_due_date=('due_date','max')).reset_index()
    loans['paid_ratio']=ratio(loans.paid_sum,loans.due_sum)
    loanfeature=stats(loans,['max_historical_dpd','current_dpd','mean_historical_dpd','paid_ratio'],'v2_loan_repay')
    attach(feature.join(loanfeature))
    del rep,feature,loanfeature,latest_due,late_last,subset,f
    gc.collect()

    prev=history('previous_applications',['application_datetime','decision_datetime'])
    prev=prev[prev.decision_datetime.lt(prev.cutoff)].copy()
    prev['age_days']=(prev.cutoff-prev.decision_datetime).dt.total_seconds()/86400
    prev['decision_delay_hours']=(prev.decision_datetime-prev.application_datetime).dt.total_seconds()/3600
    prev['approved_fraction']=ratio(prev.approved_amount,prev.requested_amount)
    feature=stats(prev,['requested_amount','approved_amount','term_months','annual_interest_rate','scheduled_monthly_installment','decision_delay_hours'],'v2_prev')
    last=prev.sort_values('decision_datetime').groupby('customer_id').tail(1).set_index('customer_id')
    feature=feature.join(last[['requested_amount','approved_amount','term_months','annual_interest_rate','scheduled_monthly_installment','age_days']].add_prefix('v2_prev_last_'))
    feature['v2_prev_last_decision']=last.decision
    approved=prev[prev.decision.eq('approved')].merge(loans,on=['loan_id','customer_id'],how='left',validate='one_to_one')
    approved['due_count']=approved.due_count.fillna(0)
    approved['unpaid_sum']=approved.unpaid_sum.fillna(0)
    approved['future_installments']=(approved.term_months-approved.due_count).clip(lower=0)
    approved['active']=approved.future_installments.gt(0).astype(int)
    approved['active_installment']=approved.scheduled_monthly_installment*approved.active
    monthly_rate=approved.annual_interest_rate/12
    factor=(1+monthly_rate)**approved.due_count
    approved['estimated_remaining_principal']=(approved.approved_amount*factor-approved.scheduled_monthly_installment*(factor-1)/monthly_rate).clip(lower=0)
    approved['remaining_contractual_payments']=approved.future_installments*approved.scheduled_monthly_installment
    f=approved.groupby('customer_id').agg(
        active_count=('active','sum'),active_installment=('active_installment','sum'),
        future_installments=('future_installments','sum'),future_scheduled_amount=('remaining_contractual_payments','sum'),
        estimated_remaining_principal=('estimated_remaining_principal','sum'),unpaid_amount=('unpaid_sum','sum'))
    f=f.add_prefix('v2_existing_')
    feature=feature.join(f)
    lastloan=approved.sort_values('decision_datetime').groupby('customer_id').tail(1).set_index('customer_id')
    feature=feature.join(lastloan[['max_historical_dpd','mean_historical_dpd','current_dpd','paid_ratio','active','age_days']].add_prefix('v2_latest_loan_'))
    for days in [180,365]:
        sub=prev[prev.age_days.le(days)]
        for status in ['approved','rejected','cancelled']:
            counts=sub[sub.decision.eq(status)].groupby('customer_id').size()
            feature[f'v2_prev_{status}_{days}d']=counts.reindex(feature.index,fill_value=0)
    attach(feature)
    del prev,approved,feature,f,sub,last,lastloan,loans
    gc.collect()

    bureau=history('bureau_observations',['report_datetime','available_datetime'])
    bureau=bureau[bureau.report_datetime.lt(bureau.cutoff)&bureau.available_datetime.lt(bureau.cutoff)].copy()
    bureau['age_days']=(bureau.cutoff-bureau.report_datetime).dt.total_seconds()/86400
    bureau['lag_days']=(bureau.available_datetime-bureau.report_datetime).dt.total_seconds()/86400
    cols=['score','active_accounts','outstanding_balance','revolving_credit_limit','revolving_outstanding_balance',
          'utilization','inquiries_last_6m','delinquent_accounts_last_12m','age_days','lag_days']
    feature=stats(bureau,cols,'v2_bureau_history')
    selected_ids=base.application_id
    current=bureau[bureau.application_id.eq(bureau.customer_id.map(selected_ids))].copy()
    feature=feature.join(stats(current,cols,'v2_bureau_current'))
    for days in [90,180,365]:
        recent=bureau[bureau.age_days.le(days)]
        feature=feature.join(stats(recent,['score','utilization','outstanding_balance','inquiries_last_6m','delinquent_accounts_last_12m'],f'v2_bureau_{days}d',('mean','min','max')))
    ordered=bureau.sort_values(['report_datetime','available_datetime','bureau_observation_id'])
    latest=ordered.groupby(['customer_id','bureau_provider']).tail(1).copy()
    oldest=ordered.groupby(['customer_id','bureau_provider']).head(1).copy()
    for col in cols:
        pivot=latest.pivot(index='customer_id',columns='bureau_provider',values=col)
        pivot.columns=[f'v2_{provider}_latest_{col}' for provider in pivot.columns]
        feature=feature.join(pivot)
    change=latest.merge(oldest[['customer_id','bureau_provider','score','utilization','outstanding_balance','report_datetime']],
                        on=['customer_id','bureau_provider'],suffixes=('','_first'),validate='one_to_one')
    for col in ['score','utilization','outstanding_balance']:
        change[f'{col}_change']=change[col]-change[f'{col}_first']
    change['elapsed_days']=(change.report_datetime-change.report_datetime_first).dt.total_seconds()/86400
    feature=feature.join(stats(change,['score_change','utilization_change','outstanding_balance_change','elapsed_days'],'v2_bureau_change'))
    attach(feature)
    del bureau,current,feature,recent,ordered,latest,oldest,change,pivot
    gc.collect()

    account=history('account_behavior',['month'])
    end=account.month+pd.offsets.MonthBegin(1)
    audit['account_incomplete_month']=int(end.gt(account.cutoff).sum())
    account=account[end.le(account.cutoff)].copy()
    account['months_back']=12*(account.cutoff.dt.year-account.month.dt.year)+account.cutoff.dt.month-account.month.dt.month
    account['low_balance_fraction']=account.low_balance_days/account.month.dt.days_in_month
    account['saving_rate']=ratio(account.inflows-account.outflows,account.inflows)
    account['spending_income_ratio']=ratio(account.outflows,account.income_like_credits)
    account['balance_income_ratio']=ratio(account.average_balance,account.income_like_credits)
    cols=['inflows','outflows','income_like_credits','average_balance','minimum_balance','low_balance_fraction','saving_rate','spending_income_ratio','balance_income_ratio']
    feature=stats(account,cols,'v2_account')
    for months in [3,6,12]:
        recent=account[account.months_back.between(1,months)]
        feature=feature.join(stats(recent,cols,f'v2_account_{months}m'))
    for col in ['income_like_credits','outflows','average_balance','low_balance_fraction']:
        recent=account[account.months_back.between(1,3)].groupby('customer_id')[col].mean()
        earlier=account[account.months_back.between(4,6)].groupby('customer_id')[col].mean()
        feature[f'v2_account_{col}_recent_prior_ratio']=ratio(recent,earlier)
        feature[f'v2_account_{col}_recent_prior_difference']=recent-earlier
    last=account.sort_values('month').groupby('customer_id').tail(1).set_index('customer_id')
    feature=feature.join(last[['minimum_balance','low_balance_fraction','saving_rate','spending_income_ratio','balance_income_ratio']].add_prefix('v2_account_last_'))
    attach(feature)
    del account,feature,recent,earlier,last
    gc.collect()

    ecom=history('ecommerce_transactions',['transaction_datetime'])
    ecom=ecom[ecom.transaction_datetime.lt(ecom.cutoff)].copy()
    ecom['age_days']=(ecom.cutoff-ecom.transaction_datetime).dt.total_seconds()/86400
    ecom['unit_price']=ratio(ecom.item_total,ecom.quantity)
    orders=ecom.drop_duplicates('order_id').copy()
    orders['completed']=orders.order_status.eq('completed').astype(int)
    orders['cancelled']=orders.order_status.eq('cancelled').astype(int)
    orders['spend']=orders.order_total*orders.completed
    completed=orders[orders.completed.eq(1)].copy()
    feature=stats(completed,['order_total','age_days'],'v2_ecom_order',('std','min','max','median'))
    for days in [30,60,90,180,365]:
        subset=orders[orders.age_days.le(days)]
        f=subset.groupby('customer_id').agg(
            orders=('order_id','size'),completed=('completed','sum'),cancelled=('cancelled','sum'),spend=('spend','sum'))
        f['cancel_rate']=ratio(f.cancelled,f.orders)
        f['average_order_value']=ratio(f.spend,f.completed)
        feature=feature.join(f.add_prefix(f'v2_ecom_{days}d_'))
    prior=completed[completed.age_days.gt(90)&completed.age_days.le(180)].groupby('customer_id').agg(
        spend=('order_total','sum'),orders=('order_id','size'))
    recent=completed[completed.age_days.le(90)].groupby('customer_id').agg(spend=('order_total','sum'),orders=('order_id','size'))
    feature['v2_ecom_spend_recent_prior_ratio']=ratio(recent.spend,prior.spend)
    feature['v2_ecom_orders_recent_prior_ratio']=ratio(recent.orders,prior.orders)
    feature['v2_ecom_spend_recent_prior_difference']=recent.spend-prior.spend
    first=orders.groupby('customer_id').age_days.max()
    feature['v2_ecom_observed_days']=first
    feature['v2_ecom_completed_per_observed_month']=ratio(base.ecom_completed_count,first/30)
    for days in [90,180]:
        items=ecom[ecom.order_status.eq('completed')&ecom.age_days.le(days)]
        cat=items.groupby(['customer_id','product_category']).item_total.sum().unstack(fill_value=0)
        cat.columns=[f'v2_ecom_{days}d_category_{c}' for c in cat.columns]
        feature=feature.join(cat)
        pay=completed[completed.age_days.le(days)].groupby(['customer_id','payment_method']).order_total.sum().unstack(fill_value=0)
        pay=pay.div(pay.sum(axis=1),axis=0)
        pay.columns=[f'v2_ecom_{days}d_payment_spend_share_{c}' for c in pay.columns]
        feature=feature.join(pay)
    latest=completed.sort_values('transaction_datetime').groupby('customer_id').tail(1).set_index('customer_id')
    feature['v2_ecom_last_order_amount']=latest.order_total
    feature['v2_ecom_last_payment_method']=latest.payment_method
    itemstats=stats(ecom[ecom.order_status.eq('completed')],['unit_price','quantity'],'v2_ecom_item')
    attach(feature.join(itemstats))
    del ecom,orders,completed,feature,subset,f,prior,recent,first,items,cat,pay,latest,itemstats
    gc.collect()

    full=pd.concat([base]+parts,axis=1).copy()
    full['v2_existing_total_burden_declared']=(full.proposed_monthly_installment+full.v2_existing_active_installment.fillna(0))/full.declared_monthly_income
    full['v2_existing_total_burden_observed']=ratio(full.proposed_monthly_installment+full.v2_existing_active_installment,full.account_income_mean)
    full['v2_existing_debt_declared_income']=ratio(full.v2_existing_estimated_remaining_principal,full.declared_monthly_income)
    full['v2_existing_unpaid_declared_income']=ratio(full.v2_existing_unpaid_amount,full.declared_monthly_income)
    full['v2_new_previous_loan_amount_ratio']=ratio(full.loan_amount,full.v2_prev_last_approved_amount)
    full['v2_new_previous_installment_ratio']=ratio(full.proposed_monthly_installment,full.v2_prev_last_scheduled_monthly_installment)
    full['v2_income_after_existing_installments']=full.declared_monthly_income-full.v2_existing_active_installment.fillna(0)
    full['v2_installment_disposable_income_ratio']=ratio(full.proposed_monthly_installment,full.v2_income_after_existing_installments)
    full['v2_repay_dpd_recent_historical_difference']=full.v2_repay_90d_dpd_mean-full.repayment_historical_dpd_mean
    full['v2_bureau_current_historical_score_difference']=full.v2_bureau_current_score_mean-full.v2_bureau_history_score_mean
    full['v2_bureau_current_latest_score_difference']=full.v2_bureau_current_score_mean-full.bureau_score_mean
    full['v2_account_income_3m_declared_ratio']=ratio(full.v2_account_3m_income_like_credits_mean,full.declared_monthly_income)
    full['v2_ecom_spend30_declared_income_ratio']=ratio(full.v2_ecom_30d_spend,full.declared_monthly_income)
    full['v2_ecom_spend90_observed_income_ratio']=ratio(full.v2_ecom_90d_spend/3,full.account_income_mean)
    full['v2_ecom_spend90_balance_ratio']=ratio(full.v2_ecom_90d_spend,full.account_last_closing_balance)
    full['v2_application_month']=full.decision_datetime.dt.month
    # Month is observed at prediction time; opaque IDs remain excluded.
    assert full.columns.is_unique and full.index.is_unique
    full.replace([np.inf,-np.inf],np.nan,inplace=True)
    assert all(v==0 for v in audit.values()),audit
    full.reset_index().to_parquet(PROCESSED/'application_features_extended.parquet',index=False)
    audit['base_columns']=len(base.columns)+1
    audit['extended_columns']=len(full.columns)+1
    (ROOT/'artifacts/extended_feature_audit.json').write_text(json.dumps(audit,indent=2))
    print(f'Saved extended features: {full.shape[0]:,} rows, {full.shape[1]+1} columns.',flush=True)


if __name__=='__main__':
    main()
