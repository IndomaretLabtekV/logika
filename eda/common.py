from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = (
    "train_applications", "test_applications", "customer_profiles",
    "previous_applications", "repayment_history", "account_behavior",
    "bureau_observations", "ecommerce_transactions",
)


def connect():
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    for table in TABLES:
        path = str(DATA / f"{table}.csv").replace("'", "''")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{path}')")
    con.execute("""
        CREATE TEMP TABLE applications AS
        SELECT *, 'train' AS split FROM train_applications
        UNION ALL BY NAME
        SELECT *, 'test' AS split FROM test_applications
    """)
    return con
