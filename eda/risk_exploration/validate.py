#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.5.6"]
# ///
# How to run: install uv from https://docs.astral.sh/uv/ then
# uv run eda/risk_exploration/validate.py
"""Exercise actual feature SQL with cutoff, censoring, missingness and label fixtures."""
from __future__ import annotations

import json
import sys

import duckdb

from build import DEST, ROOT, TABLES, construct


def fixture(con: duckdb.DuckDBPyConnection, label: int) -> None:
    """Use raw source schemas with hand-calculable historical and future events."""
    for name in TABLES:
        source = str(ROOT / "data" / f"{name}.csv").replace("'", "''")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{source}') LIMIT 0")
    con.execute("""INSERT INTO train_applications VALUES
      ('a','c','2025-03-31','2025-04-01',1000,12,0.2,100,'household',?),
      ('b','d','2025-03-31','2025-04-01',1000,12,0.2,100,'household',?)""", [label,label])
    con.execute("""INSERT INTO customer_profiles VALUES
      ('c',30,'Region_A','salaried',24,1000,0,'rented'),
      ('d',30,'Region_A','salaried',24,1000,0,'rented')""")
    con.execute("""INSERT INTO previous_applications VALUES
      ('p','c','2025-01-01','2025-01-01',500,12,'approved','l',500,0.2,50),
      ('future','c','2025-04-01','2025-04-01',999999,12,'rejected',NULL,NULL,NULL,NULL)""")
    con.execute("""INSERT INTO repayment_history VALUES
      ('l','r1','c','2025-03-01',100,50,'2025-03-02',5,10),
      ('l','r2','c','2025-04-01',100,0,NULL,0,0),
      ('l','r3','c','2025-03-02',100,100,'2025-04-02',0,10)""")
    con.execute("""INSERT INTO account_behavior VALUES
      ('c','2025-01-01',100,200,300,200,300,150,50,4),
      ('c','2025-03-01',200,400,500,300,500,250,50,5),
      ('c','2025-04-01',400,900000,900000,400,900000,50000,400,0)""")
    con.execute("""INSERT INTO bureau_observations VALUES
      ('bu1','a','c','Bureau_A','2025-03-30','2025-04-01',600,1,100,100,50,0.5,1,0),
      ('bu2','a','c','Bureau_A','2025-03-31','2025-04-02',850,1,0,100,0,0,0,0)""")
    con.execute("""INSERT INTO ecommerce_transactions VALUES
      ('o1','i1','c','2025-03-31','groceries',1,40,100,'COD','completed'),
      ('o1','i2','c','2025-03-31','health',1,60,100,'COD','completed'),
      ('o2','i3','c','2025-03-31','electronics',1,900,900,'COD','cancelled'),
      ('o3','i4','c','2025-04-01','electronics',1,9999,9999,'COD','completed'),
      ('o4','i5','c','2025-04-02','electronics',1,9999,9999,'COD','completed')""")
    construct(con)


def main() -> None:
    """Assert observable feature values and save a concise evidence record."""
    if "--help" in sys.argv:
        print(__doc__ + "\nUsage: python eda/risk_exploration/validate.py (no arguments)")
        return
    if len(sys.argv) != 1:
        sys.exit("Unexpected argument. Use --help.")
    checks = []
    with duckdb.connect() as con:
        fixture(con,0)
        row=con.sql("""SELECT prev_count,rep_dues,rep_shortfall,ac_months,ac_missing_months,
          ac_adjacent_pairs,ac_latest_close,bu_score,ec_orders,ec_completed,ec_spend90,
          graph_category_degree,income_change3 FROM features WHERE application_id='a'""").fetchone()
        assert row == (1,1,50,2,1,0,400,600.0,2,1,100,2,None), row
        checks.extend(["strict event cutoff", "bureau availability at cutoff", "complete month at cutoff",
                       "future payment snapshot quarantined", "missing month breaks lag adjacency",
                       "short windows do not fabricate income changes", "order totals deduplicated",
                       "cancelled spending excluded", "category weights from item totals"])
        row=con.sql("SELECT ac_months,rep_dues,ec_spend90 FROM features WHERE application_id='b'").fetchone()
        assert row == (None,None,None), row
        checks.append("absent source remains distinct from observed zero")
        without_labels=con.sql("SELECT * EXCLUDE(target) FROM features ORDER BY application_id").fetchall()
    with duckdb.connect() as con:
        fixture(con,1)
        assert con.sql("SELECT * EXCLUDE(target) FROM features ORDER BY application_id").fetchall() == without_labels
        checks.append("all candidate features invariant to target mutation")
    with duckdb.connect() as con:
        con.read_parquet(str(DEST / "features.parquet")).create_view("features")
        assert con.sql("SELECT count(*),count(DISTINCT application_id),count(DISTINCT customer_id) FROM features").fetchone() == (100000,100000,100000)
        assert con.sql("SELECT count(*) FROM features WHERE split='train'").fetchone() == (70000,)
        assert con.sql("SELECT count(*) FROM features WHERE split='test' AND target IS NULL").fetchone() == (30000,)
        checks.append("full data grain, split sizes and unlabeled test preserved")
    result={"status":"passed","checks":checks,"count":len(checks)}
    (DEST / "validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
