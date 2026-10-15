"""Native TabPFN-Rel featurization (relarena_core.build_dfs_features = fastdfs, the exact code TabPFN-Rel calls
before its TabPFN step) on the LOGIKA relational schema. Preprocessing only: no TabPFN / torch is imported.
Needs the Python 3.12 environment (TabPFN-Rel requires Python < 3.13):
    uv venv --python 3.12 .venv-rel && VIRTUAL_ENV=.venv-rel uv pip install tabpfn-rel "relarena-core[dfs]" polars pyarrow
    .venv-rel/bin/python eda/66a_tabpfn_rel_dfs.py 3
Schema: customers (entity) <- applications, prev_apps, loans <- installments, account months, bureau reports,
orders <- items. Every child row is filtered strictly before the application's decision_datetime by fastdfs.
Writes data_pq/rel_dfs_d{depth}.parquet (application_id + DFS columns) for 66b to screen."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
from relarena_core.featurization.cache import CacheConfig
from relarena_core.featurization.dfs import build_dfs_features
from relbench.base import Database, Table

PQ = Path(__file__).resolve().parent.parent / "data_pq"
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rd = lambda n: pl.read_parquet(PQ / f"{n}.parquet")  # noqa: E731
apps = pl.concat([rd("train_applications"), rd("test_applications")], how="diagonal_relaxed")
prev = rd("previous_applications")
T = lambda df, **kw: Table(df=df.to_pandas(), **kw)  # noqa: E731
tables = {
    "customers": T(rd("customer_profiles"), fkey_col_to_pkey_table={}, pkey_col="customer_id"),
    "applications": T(apps.drop("target", "decision_datetime"), fkey_col_to_pkey_table={"customer_id": "customers"},
                      pkey_col="application_id", time_col="application_datetime"),
    "prev_apps": T(prev.drop("loan_id", "approved_amount", "scheduled_monthly_installment"),
                   fkey_col_to_pkey_table={"customer_id": "customers"}, pkey_col="application_id", time_col="decision_datetime"),
    "loans": T(prev.filter(pl.col("loan_id").is_not_null()).select(
        "loan_id", "customer_id", "decision_datetime", "approved_amount", "annual_interest_rate", "term_months", "scheduled_monthly_installment"),
        fkey_col_to_pkey_table={"customer_id": "customers"}, pkey_col="loan_id", time_col="decision_datetime"),
    "installments": T(rd("repayment_history"), fkey_col_to_pkey_table={"customer_id": "customers", "loan_id": "loans"},
                      pkey_col="installment_id", time_col="due_date"),
    "account_months": T(rd("account_behavior").with_columns(available=pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us"))),
                        fkey_col_to_pkey_table={"customer_id": "customers"}, time_col="available"),
    "bureau": T(rd("bureau_observations").drop("application_id"), fkey_col_to_pkey_table={"customer_id": "customers"},
                pkey_col="bureau_observation_id", time_col="available_datetime"),
}
e = rd("ecommerce_transactions")
tables["orders"] = T(e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "order_total", "payment_method", "order_status").first()),
                     fkey_col_to_pkey_table={"customer_id": "customers"}, pkey_col="order_id", time_col="transaction_datetime")
tables["items"] = T(e.select("order_item_id", "order_id", "transaction_datetime", "product_category", "quantity", "item_total"),
                    fkey_col_to_pkey_table={"order_id": "orders"}, pkey_col="order_item_id", time_col="transaction_datetime")
db = Database(tables)
task = SimpleNamespace(entity_table="customers", entity_col="customer_id", time_col="decision_datetime", target_col="target")
label = Table(df=apps.select("customer_id", "decision_datetime", "target").to_pandas(), fkey_col_to_pkey_table={"customer_id": "customers"},
              time_col="decision_datetime")
t0 = time.time()
X, cats = build_dfs_features(task, db, label, depth=depth, max_depth=depth, keep_anchor_columns=False,
                             cache=CacheConfig(directory=None, on_miss="compute"))
print(f"depth {depth}: {X.shape[1]} DFS columns ({len(cats)} categorical) for {len(X)} applications in {time.time() - t0:.0f}s")
X = X.copy()
for c in cats:
    X[c] = X[c].astype("category").cat.codes.astype("float64").where(X[c].notna())
names = pd.DataFrame({"col": [f"rel_{i:03d}" for i in range(X.shape[1])], "dfs_name": list(map(str, X.columns))})
X.columns = names["col"]
X.insert(0, "application_id", apps["application_id"].to_list())
X.to_parquet(PQ / f"rel_dfs_d{depth}.parquet")
names.to_csv(PQ / f"rel_dfs_d{depth}_names.csv", index=False)
print("saved", PQ / f"rel_dfs_d{depth}.parquet")
