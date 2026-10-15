"""Audit what historical state is and is not identifiable from a final repayment snapshot."""
from shared8 import *

def main():
    h=hist();r=table('repayment_history')
    q=h.select('application_id','customer_id',T='decision_datetime').join(r,on='customer_id').filter(pl.col('due_date')<=pl.col('T'))
    future_payment=pl.col('last_payment_datetime')>pl.col('T')
    before=pl.col('last_payment_datetime')<=pl.col('T')
    q=q.with_columns(ambiguous_payment=future_payment.fill_null(False),age_days=(pl.col('T')-pl.col('due_date')).dt.total_days())
    summary=q.group_by('application_id').agg(n_installments=pl.len(),ambiguous_installments=pl.col('ambiguous_payment').sum(),ambiguous_final_amount=pl.col('amount_paid').filter('ambiguous_payment').sum())
    out=h.select('application_id','customer_id','hist_target','decision_datetime','sel_cutoff').join(summary,on='application_id',how='left',validate='1:1').with_columns(pl.col('n_installments','ambiguous_installments','ambiguous_final_amount').fill_null(0))
    out.write_parquet(OUT/'01_snapshot_quality.parquet')
    samples=q.filter('ambiguous_payment').join(h.select('application_id','hist_target'),on='application_id',validate='m:1').group_by('hist_target').head(12)
    csv(samples.select('application_id','hist_target','T','due_date','amount_due','amount_paid','last_payment_datetime','days_past_due','max_days_past_due').to_pandas(),'01_ambiguous_examples')
    # Two allocation histories produce identical final data, but different amount_paid at historical T.
    traces=pd.DataFrame([dict(history='A',day=10,allocation=0),dict(history='A',day=40,allocation=100),dict(history='B',day=10,allocation=40),dict(history='B',day=40,allocation=60)])
    a=traces.groupby('history').allocation.sum();assert a.A==a.B==100
    past=traces[traces.day<=20].groupby('history').allocation.sum();assert past.A!=past.B
    csv(traces,'01_nonidentifiability_fixture')
    d=out.group_by('hist_target').agg(rows=pl.len(),with_ambiguous=(pl.col('ambiguous_installments')>0).sum(),mean_ambiguous=pl.col('ambiguous_installments').mean()).sort('hist_target');csv(d.to_pandas(),'01_quality_by_class')
    prof=table('customer_profiles').join(h.select('application_id','customer_id','decision_datetime','sel_cutoff'),on='customer_id')
    elapsed=(pl.col('sel_cutoff')-pl.col('decision_datetime')).dt.total_days()/30.44
    js(dict(historical_rows=h.height,earlier_due_snapshot_pairs=q.height,ambiguous_payment_pairs=q.filter('ambiguous_payment').height,historical_rows_with_ambiguity=out.filter(pl.col('ambiguous_installments')>0).height,profile_time_travel_rows=prof.height,current_occupation_shorter_than_lookback=prof.filter(pl.col('employment_tenure_months')<elapsed).height,warning='Final allocation after T does not prove zero payments before T. Current income/employment/dependents/housing are not dated historical measurements. This is an identifiability problem, not proof that every historical row is wrong.'),'01_summary')
    fig,ax=plt.subplots(1,3,figsize=(15,4));ax[0].bar(['no ambiguous installment','ambiguous installment'],[out.filter(pl.col('ambiguous_installments')==0).height,out.filter(pl.col('ambiguous_installments')>0).height]);ax[0].tick_params(axis='x',rotation=15)
    for name,g in traces.groupby('history'):ax[1].step([0,*g.day],[0,*g.allocation.cumsum()],where='post',label=name)
    ax[1].axvline(20,color='black',linestyle='--');ax[1].legend();ax[1].set(title='Same final snapshot, different past',xlabel='Day',ylabel='Cumulative allocated payment')
    for label in [0,1]:ax[2].hist(out.filter(pl.col('hist_target')==label)['ambiguous_installments'],bins=np.arange(0,22),density=True,histtype='step',label=f'hist_target={label}')
    ax[2].legend();ax[2].set_title('Ambiguity count; display 0–20');figsave(fig,'01_snapshot_quality')

if __name__=='__main__':main()
