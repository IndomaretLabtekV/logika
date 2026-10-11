#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.4", "scipy==1.18.1"]
# ///
# How to run: install uv from https://docs.astral.sh/uv/ then
# uv run eda/risk_exploration/diagnose.py
"""Write descriptive evidence; none of these statistics are validation scores."""
from __future__ import annotations

import sys

import duckdb
import numpy as np
from scipy.stats import ks_2samp

from build import DEST, HERE, export


def main() -> None:
    """Measure associations, drift, chronology and supported graph structures."""
    if "--help" in sys.argv:
        print(__doc__ + "\nRun build.py first. Usage: python eda/risk_exploration/diagnose.py")
        return
    if len(sys.argv) != 1:
        sys.exit("Unexpected argument. Use --help.")
    with duckdb.connect() as con:
        for name in ("features", "joint_months", "category_edges"):
            con.read_parquet(str(DEST / f"{name}.parquet")).create_view(name)
        con.execute((HERE / "diagnostics.sql").read_text())
        for name in ("quality_checks", "lag_profiles", "seasonality", "account_transitions",
                     "category_network", "region_category_network", "geographic_cohorts",
                     "region_standardized", "coverage_patterns", "stress_scenarios"):
            export(con, f"SELECT * FROM {name} ORDER BY ALL", f"{name}.csv")
        con.execute("""CREATE TABLE numeric_associations(feature VARCHAR, period VARCHAR,
          observed_n BIGINT, missing_fraction DOUBLE, raw_feature_ks DOUBLE,
          train_test_ks DOUBLE, p01 DOUBLE,p50 DOUBLE,p99 DOUBLE)
        """)
        con.execute("CREATE TABLE bin_rates(feature VARCHAR,bin VARCHAR,n BIGINT,events BIGINT,rate DOUBLE)")
        for name, dtype, *_ in con.sql("DESCRIBE features").fetchall():
            if dtype not in {"BIGINT", "DOUBLE", "HUGEINT", "INTEGER", "FLOAT"} or name == "target":
                continue
            rows = con.sql(f'SELECT "{name}"::DOUBLE AS x,target,split,period FROM segmented').fetchnumpy()
            x = np.asarray(rows["x"].filled(np.nan) if np.ma.isMaskedArray(rows["x"]) else rows["x"])
            y = np.asarray(rows["target"].filled(-1) if np.ma.isMaskedArray(rows["target"]) else rows["target"])
            finite = np.isfinite(x)
            assert not np.isinf(x).any(), f"Infinite feature: {name}"
            train = rows["split"] == "train"
            observed = x[train & finite]
            if len(observed) == 0:
                continue
            quantiles = np.quantile(observed, [0.01, 0.25, 0.5, 0.75, 0.99])
            test = x[(~train) & finite]
            drift = float(ks_2samp(observed, test).statistic) if len(test) else None
            for period in ("all", "early", "late"):
                mask = train if period == "all" else train & (rows["period"] == period)
                good, bad = x[mask & finite & (y == 0)], x[mask & finite & (y == 1)]
                ks = float(ks_2samp(good, bad).statistic) if len(good) and len(bad) else None
                con.execute("INSERT INTO numeric_associations VALUES (?,?,?,?,?,?,?,?,?)", [
                    name, period, int((mask & finite).sum()), float(1-finite[mask].mean()), ks, drift,
                    float(quantiles[0]), float(quantiles[2]), float(quantiles[4])])
            edges = np.unique(quantiles[1:4])
            for low, high in zip(np.r_[-np.inf, edges], np.r_[edges, np.inf], strict=True):
                mask = train & finite & (x>low) & (x<=high)
                if mask.any():
                    con.execute("INSERT INTO bin_rates VALUES (?,?,?,?,?)", [name,f"({low:.6g}, {high:.6g}]",
                                int(mask.sum()),int(y[mask].sum()),float(y[mask].mean())])
            mask = train & (~finite)
            if mask.any():
                con.execute("INSERT INTO bin_rates VALUES (?,?,?,?,?)", [name,"missing",int(mask.sum()),
                            int(y[mask].sum()),float(y[mask].mean())])
        export(con,"SELECT * FROM numeric_associations ORDER BY period,raw_feature_ks DESC", "numeric_associations.csv")
        con.execute("""CREATE TABLE interactions(question VARCHAR,stratum VARCHAR,segment VARCHAR,
           period VARCHAR,n BIGINT,events BIGINT,rate DOUBLE)""")
        pairs = [
            ("cash beyond score", "score_band", "cash_state"),
            ("spending beyond score", "score_band", "spend_band"),
            ("income change by employment", "employment_type", "income_state"),
            ("cash versus repayment", "repayment_state", "cash_state"),
            ("spending versus history length", "history_band", "spend_band"),
            ("purpose versus liquidity", "loan_purpose", "cash_state"),
            ("region versus score", "score_band", "region"),
            ("category diversity versus score", "score_band", "graph_category_degree::VARCHAR"),
        ]
        for question, stratum, segment in pairs:
            con.execute(f"""INSERT INTO interactions SELECT ?,{stratum},coalesce({segment},'missing'),
                period,count(*),sum(target),avg(target) FROM segmented WHERE split='train' GROUP BY 2,3,4""", [question])
        for table in ("bin_rates", "interactions"):
            export(con, f"""SELECT *,
              (rate+3.8416/(2*n)-1.96*sqrt(rate*(1-rate)/n+3.8416/(4*n*n)))/(1+3.8416/n) AS wilson_low,
              (rate+3.8416/(2*n)+1.96*sqrt(rate*(1-rate)/n+3.8416/(4*n*n)))/(1+3.8416/n) AS wilson_high,
              n>=200 AND events>=20 AS adequate_support
              FROM {table} ORDER BY ALL""", f"{table}.csv")
        export(con,"""SELECT region,count(*) AS n,avg(target) AS raw_rate,
            avg((ac_months IS NOT NULL)::INTEGER) AS account_coverage,
            avg(bu_score) AS score,avg(installment_income) AS installment_income
            FROM features WHERE split='train' GROUP BY 1 ORDER BY 1""", "region_summary.csv")
        export(con,"""SELECT score_band,cash_state,spend_band,period,count(*) AS n,
            sum(target) AS events,avg(target) AS rate
            FROM segmented WHERE split='train' GROUP BY 1,2,3,4 ORDER BY ALL""", "three_way_risk.csv")
        export(con,"""SELECT employment_type,history_band,period,count(*) AS n,
            avg(target) AS rate,avg(income_change3) AS income_change,
            avg(ec_observed_span_days) AS commerce_observed_span
            FROM segmented WHERE split='train' GROUP BY 1,2,3 ORDER BY ALL""", "observation_controls.csv")
        export(con,"""SELECT date_trunc('month',decision_datetime)::DATE AS month,split,count(*) AS n,
             avg(target) AS target_rate,avg(bu_score) AS bureau_score,avg(ec_spend90) AS spend90,
             avg((ac_months IS NOT NULL)::INTEGER) AS account_coverage
             FROM features GROUP BY 1,2 ORDER BY 1,2""", "decision_cohorts.csv")
        print(con.sql("SELECT feature,raw_feature_ks,train_test_ks FROM numeric_associations WHERE period='all' ORDER BY raw_feature_ks DESC LIMIT 12").fetchall())


if __name__ == "__main__":
    main()
