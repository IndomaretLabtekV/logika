#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.5.6"]
# ///
# How to run: install uv from https://docs.astral.sh/uv/ then
# uv run eda/risk_exploration/build.py
"""Build cutoff-safe candidate features and aligned monthly panels from raw CSVs."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Final

import duckdb

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parents[1]
DEST: Final = ROOT / "outputs" / "04_risk_exploration"
TABLES: Final = (
    "train_applications", "test_applications", "customer_profiles", "previous_applications",
    "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions",
)


def construct(con: duckdb.DuckDBPyConnection) -> None:
    """Transform registered source tables; targets are carried, never used in features."""
    con.execute("""CREATE TEMP TABLE applications AS
        SELECT *, 'train' AS split FROM train_applications UNION ALL BY NAME
        SELECT *, 'test' AS split FROM test_applications""")
    for source in sorted((HERE / "sql").glob("*.sql")):
        for statement in source.read_text().split(";"):
            if statement.strip():
                con.execute(statement)


def export(con: duckdb.DuckDBPyConnection, query: str, name: str) -> None:
    """Persist ordered aggregate evidence or a compressed application-level artifact."""
    path = str(DEST / name).replace("'", "''")
    fmt = "PARQUET, COMPRESSION ZSTD" if name.endswith(".parquet") else "CSV, HEADER TRUE"
    con.execute(f"COPY ({query}) TO '{path}' (FORMAT {fmt})")


def main() -> None:
    """Run the complete source-to-feature pipeline with reproducibility metadata."""
    if "--help" in sys.argv:
        print(__doc__ + "\nUsage: python eda/risk_exploration/build.py (no arguments)")
        return
    if len(sys.argv) != 1:
        sys.exit("Unexpected argument. Use --help.")
    DEST.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute("SET threads=4; SET memory_limit='6GB'")
        sources = []
        for name in TABLES:
            path = ROOT / "data" / f"{name}.csv"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources.append({"file": str(path.relative_to(ROOT)), "sha256": digest, "bytes": path.stat().st_size})
            con.read_csv(str(path)).create_view(name)
        construct(con)
        counts = con.sql("SELECT count(*),count(DISTINCT application_id),count(DISTINCT customer_id) FROM features").fetchone()
        expected = con.sql("SELECT count(*) FROM applications").fetchone()
        assert counts is not None and expected is not None
        assert counts[0] == counts[1] == counts[2] == expected[0], "Application/customer grain changed"
        assert con.sql("SELECT count(*) FROM features WHERE target NOT IN (0,1) OR (split='train' AND target IS NULL) OR (split='test' AND target IS NOT NULL)").fetchone() == (0,)
        for table in ("source_contracts", "eligibility_audit", "relationship_inventory"):
            export(con, f"SELECT * FROM {table} ORDER BY 1", f"{table}.csv")
        assert con.sql("SELECT sum(flagged) FROM source_contracts").fetchone() == (0,), "Source contract failure; inspect source_contracts.csv"
        for table in ("features", "joint_months", "category_edges"):
            export(con, f"SELECT * FROM {table} ORDER BY application_id", f"{table}.parquet")
        export(con, """SELECT application_id,customer_id,split,decision_datetime,
               decision_datetime+INTERVAL 12 MONTH AS label_maturity_proxy,
               CASE WHEN month(decision_datetime)<=9 THEN 'early' ELSE 'late' END AS retrospective_period
               FROM features ORDER BY application_id""", "split_manifest.parquet")
        inventory = []
        metadata = {"application_id", "customer_id", "target", "split", "application_datetime", "decision_datetime"}
        for name, dtype, *_ in con.sql("DESCRIBE features").fetchall():
            missing, distinct = con.sql(f'SELECT avg(("{name}" IS NULL)::INTEGER),count(DISTINCT "{name}") FROM features').fetchone()
            inventory.append((name, dtype, missing, distinct, "metadata" if name in metadata else "candidate"))
        con.execute("CREATE TABLE inventory(name VARCHAR,dtype VARCHAR,missing_fraction DOUBLE,distinct_values BIGINT,role VARCHAR)")
        con.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", inventory)
        export(con, "SELECT * FROM inventory ORDER BY role,name", "feature_inventory.csv")
        export(con, """SELECT split,count(*) AS applications,avg(target) AS target_rate,
            avg((ac_months IS NOT NULL)::INTEGER) AS account_coverage,
            avg((rep_dues IS NOT NULL)::INTEGER) AS repayment_coverage,
            avg((income_change3 IS NOT NULL)::INTEGER) AS six_month_change_coverage,
            avg((bu_score_change_per30d IS NOT NULL)::INTEGER) AS bureau_change_coverage,
            avg((ac_income_lag1_corr IS NOT NULL)::INTEGER) AS lag1_coverage
            FROM features GROUP BY split ORDER BY split""", "coverage.csv")
        code = []
        for path in sorted(HERE.rglob("*")):
            if path.suffix in {".py", ".sql"}:
                code.append({"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {"python": platform.python_version(), "duckdb": duckdb.__version__,
                    "rows": counts[0], "candidate_features": len(inventory)-len(metadata),
                    "sources": sources, "code": code,
                    "timezone": "Asia/Jakarta naive local timestamps", "labels_used_for_features": False}
        (DEST / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"rows": counts[0], "candidate_features": manifest["candidate_features"], "destination": str(DEST)}))


if __name__ == "__main__":
    main()
