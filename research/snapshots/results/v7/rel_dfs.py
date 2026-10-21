import sys
import time
from types import SimpleNamespace

import pandas as pd
import polars as pl
from relarena_core.featurization.cache import CacheConfig
from relarena_core.featurization.dfs import build_dfs_features
from relbench.base import Database, Table

data_dir, out_path, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])

def rd(name):
    return pl.read_csv(f'{data_dir}/{name}.csv', try_parse_dates=True, infer_schema_length=100000)

def tb(df, **kw):
    return Table(df=df.to_pandas(), **kw)

apps = pl.concat([rd('train_applications'), rd('test_applications')], how='diagonal_relaxed')
prev, e = rd('previous_applications'), rd('ecommerce_transactions')
tables = {
    'customers': tb(rd('customer_profiles'), fkey_col_to_pkey_table={}, pkey_col='customer_id'),
    'applications': tb(apps.drop('target', 'decision_datetime'), fkey_col_to_pkey_table={'customer_id': 'customers'},
                       pkey_col='application_id', time_col='application_datetime'),
    'prev_apps': tb(prev.drop('loan_id', 'approved_amount', 'scheduled_monthly_installment'),
                    fkey_col_to_pkey_table={'customer_id': 'customers'}, pkey_col='application_id', time_col='decision_datetime'),
    'loans': tb(prev.filter(pl.col('loan_id').is_not_null()).select(
        'loan_id', 'customer_id', 'decision_datetime', 'approved_amount', 'annual_interest_rate', 'term_months',
        'scheduled_monthly_installment'), fkey_col_to_pkey_table={'customer_id': 'customers'}, pkey_col='loan_id',
        time_col='decision_datetime'),
    'installments': tb(rd('repayment_history'), fkey_col_to_pkey_table={'customer_id': 'customers', 'loan_id': 'loans'},
                       pkey_col='installment_id', time_col='due_date'),
    'account_months': tb(rd('account_behavior').with_columns(available=pl.col('month').dt.offset_by('1mo').cast(pl.Datetime('us'))),
                         fkey_col_to_pkey_table={'customer_id': 'customers'}, time_col='available'),
    'bureau': tb(rd('bureau_observations').drop('application_id'), fkey_col_to_pkey_table={'customer_id': 'customers'},
                 pkey_col='bureau_observation_id', time_col='available_datetime'),
    'orders': tb(e.group_by('order_id').agg(pl.col('customer_id', 'transaction_datetime', 'order_total', 'payment_method',
                                                   'order_status').first()),
                 fkey_col_to_pkey_table={'customer_id': 'customers'}, pkey_col='order_id', time_col='transaction_datetime'),
    'items': tb(e.select('order_item_id', 'order_id', 'transaction_datetime', 'product_category', 'quantity', 'item_total'),
                fkey_col_to_pkey_table={'order_id': 'orders'}, pkey_col='order_item_id', time_col='transaction_datetime'),
}
task = SimpleNamespace(entity_table='customers', entity_col='customer_id', time_col='decision_datetime', target_col='target')
label = tb(apps.select('customer_id', 'decision_datetime', 'target'), fkey_col_to_pkey_table={'customer_id': 'customers'},
           time_col='decision_datetime')
t0 = time.time()
X, cats = build_dfs_features(task, Database(tables), label, depth=depth, max_depth=depth, keep_anchor_columns=False,
                             cache=CacheConfig(directory=None, on_miss='compute'))
X = X.reset_index(drop=True).drop(columns=[c for c in X.columns if 'MODE(' in str(c)])
for c in [c for c in cats if c in X.columns]:
    X[c] = X[c].astype('category').cat.codes.astype('float64').where(X[c].notna())
X.columns = [f'rel_{i:03d}' for i in range(X.shape[1])]
X.insert(0, 'application_id', apps['application_id'].to_list())
X.to_parquet(out_path)
print(f'depth {depth}: {X.shape[1] - 1} DFS columns ({len(cats)} categorical) for {len(X)} applications in {time.time() - t0:.0f}s')
