"""Audit cross-customer entity ownership, timestamp boundaries and maturity proxy."""
import json
import polars as pl
from common import ROOT


def main():
    out=ROOT/'results/eda_23okt/deeper_audit'
    train=pl.read_csv(ROOT/'data/train_applications.csv',try_parse_dates=True)
    test=pl.read_csv(ROOT/'data/test_applications.csv',try_parse_dates=True)
    apps=pl.concat([train.select('application_id','customer_id','decision_datetime'),test.select('application_id','customer_id','decision_datetime')])
    cutoff=apps.select('customer_id',pl.col('decision_datetime').alias('cutoff'))
    ownership=[]; boundaries=[]; loan_owners=[]; app_owners=[apps.select('application_id','customer_id')]
    schema={
        'previous_applications':(['application_id','loan_id'],['application_datetime','decision_datetime']),
        'repayment_history':(['loan_id','installment_id'],['due_date','last_payment_datetime']),
        'bureau_observations':(['bureau_observation_id','application_id'],['report_datetime','available_datetime']),
        'ecommerce_transactions':(['order_id','order_item_id'],['transaction_datetime']),
        'account_behavior':([],['month'])}
    for name,(keys,times) in schema.items():
        data=pl.read_csv(ROOT/f'data/{name}.csv',columns=['customer_id']+keys+times,try_parse_dates=True)
        unknown=data.join(cutoff,on='customer_id',how='anti').height
        for key in keys:
            grouped=data.filter(pl.col(key).is_not_null()).group_by(key).agg(pl.col('customer_id').n_unique().alias('owners'))
            ownership.append({'table':name,'key':key,'unique_keys':grouped.height,'cross_customer_keys':grouped.filter(pl.col('owners')>1).height,'unknown_customer_rows':unknown})
        if 'loan_id' in keys:
            loan_owners.append(data.select('loan_id','customer_id').drop_nulls().unique())
        if 'application_id' in keys:
            app_owners.append(data.select('application_id','customer_id').unique())
        joined=data.join(cutoff,on='customer_id',how='left',validate='m:1')
        for col in times:
            value=pl.col(col)
            if col=='month':
                value=value.dt.offset_by('1mo').cast(pl.Datetime)
            row=joined.select((value<pl.col('cutoff')).sum().alias('before'),(value==pl.col('cutoff')).sum().alias('equal'),(value>pl.col('cutoff')).sum().alias('after'),value.is_null().sum().alias('missing')).row(0,named=True)
            boundaries.append({'table':name,'timestamp':col,'comparison':'next-month start for account complete-month; raw timestamp otherwise',**row})
    for key,parts in [('loan_id',loan_owners),('application_id',app_owners)]:
        groups=pl.concat(parts).group_by(key).agg(pl.col('customer_id').n_unique().alias('owners'))
        ownership.append({'table':'cross_table_union','key':key,'unique_keys':groups.height,'cross_customer_keys':groups.filter(pl.col('owners')>1).height,'unknown_customer_rows':0})
    profiles=pl.read_csv(ROOT/'data/customer_profiles.csv')
    fingerprint=profiles.group_by([c for c in profiles.columns if c!='customer_id']).len()
    maturity=[]
    for month in range(2,13):
        boundary=f'2025-{month:02d}-01'
        observed=train.filter(pl.col('decision_datetime')<pl.lit(boundary).str.to_datetime())
        eligible=observed.filter(pl.col('decision_datetime').dt.offset_by('12mo')<=pl.lit(boundary).str.to_datetime()).height
        maturity.append({'boundary':boundary,'earlier_decisions':observed.height,'proxy_full_horizon_mature':eligible})
    result={'ownership':ownership,'cutoff_boundaries':boundaries,
            'exact_non_id_profile_duplicate_groups':fingerprint.filter(pl.col('len')>1).height,
            'label_maturity_proxy':maturity,
            'limits':['Decision+12 calendar months is a conservative full-horizon proxy, not actual origination/outcome availability.',
                      'Some positive events can be known earlier; their event/label timestamps are not supplied.',
                      'Zero mature full-horizon proxies means historical forward CV is retrospective, not proven deployment simulation.',
                      'Unique observed entity IDs do not rule out unobserved household/device/employer dependence.']}
    (out/'entity_cutoff_maturity.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
