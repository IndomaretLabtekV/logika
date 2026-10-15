import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUT, TABLES, connect


DEST = OUT / "01_audit"
DEST.mkdir(parents=True, exist_ok=True)
con = connect()

inventory = []
missing = []
keys = {
    "train_applications": "application_id",
    "test_applications": "application_id",
    "customer_profiles": "customer_id",
    "previous_applications": "application_id",
    "repayment_history": "installment_id",
    "account_behavior": "customer_id || ':' || CAST(month AS VARCHAR)",
    "bureau_observations": "bureau_observation_id",
    "ecommerce_transactions": "order_item_id",
}
for table in TABLES:
    columns = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    fields = [f"COUNT(*) AS rows", f"COUNT(DISTINCT {keys[table]}) AS unique_keys"]
    fields += [f'COUNT(*) - COUNT("{col}") AS "{col}"' for col in columns]
    row = con.execute(f"SELECT {', '.join(fields)} FROM {table}").fetchone()
    inventory.append({"table": table, "rows": row[0], "unique_keys": row[1],
                      "duplicate_keys": row[0] - row[1], "columns": len(columns)})
    missing.extend({"table": table, "column": col, "missing": count,
                    "missing_pct": 100 * count / row[0]}
                   for col, count in zip(columns, row[2:]))
pd.DataFrame(inventory).to_csv(DEST / "inventory.csv", index=False)
pd.DataFrame(missing).to_csv(DEST / "missingness.csv", index=False)

coverage = []
for table in TABLES[2:]:
    df = con.execute(f"""
        SELECT a.split, COUNT(DISTINCT a.customer_id) AS customers,
               COUNT(DISTINCT h.customer_id) AS covered
        FROM applications a LEFT JOIN (SELECT DISTINCT customer_id FROM {table}) h
        USING (customer_id) GROUP BY a.split ORDER BY a.split
    """).df()
    for row in df.itertuples(index=False):
        coverage.append({"table": table, "split": row.split, "customers": row.customers,
                         "covered": row.covered, "coverage_pct": 100 * row.covered / row.customers})
pd.DataFrame(coverage).to_csv(DEST / "customer_coverage.csv", index=False)

monthly = con.execute("""
    SELECT split, date_trunc('month', decision_datetime) AS month,
           COUNT(*) AS applications, SUM(COALESCE(target, 0)) AS positives,
           AVG(target) AS target_rate
    FROM applications GROUP BY 1, 2 ORDER BY 2, 1
""").df()
monthly.to_csv(DEST / "monthly_cohorts.csv", index=False)

checks = {
    "train_test_application_overlap": con.execute("""
        SELECT COUNT(*) FROM train_applications t JOIN test_applications e USING (application_id)
    """).fetchone()[0],
    "train_test_customer_overlap": con.execute("""
        SELECT COUNT(*) FROM train_applications t JOIN test_applications e USING (customer_id)
    """).fetchone()[0],
    "application_after_decision": con.execute("""
        SELECT COUNT(*) FROM applications WHERE application_datetime > decision_datetime
    """).fetchone()[0],
    "bureau_after_cutoff": con.execute("""
        SELECT COUNT(*) FROM bureau_observations b JOIN applications a USING (customer_id)
        WHERE b.available_datetime > a.decision_datetime
    """).fetchone()[0],
    "bureau_report_after_available": con.execute("""
        SELECT COUNT(*) FROM bureau_observations WHERE report_datetime > available_datetime
    """).fetchone()[0],
    "repayment_payment_after_cutoff": con.execute("""
        SELECT COUNT(*) FROM repayment_history r JOIN applications a USING (customer_id)
        WHERE r.last_payment_datetime > a.decision_datetime
    """).fetchone()[0],
    "repayment_due_at_or_after_cutoff": con.execute("""
        SELECT COUNT(*) FROM repayment_history r JOIN applications a USING (customer_id)
        WHERE r.due_date >= a.decision_datetime
    """).fetchone()[0],
    "account_incomplete_month_rows": con.execute("""
        SELECT COUNT(*) FROM account_behavior h JOIN applications a USING (customer_id)
        WHERE CAST(h.month AS TIMESTAMP) + INTERVAL 1 MONTH > a.decision_datetime
    """).fetchone()[0],
    "ecommerce_at_or_after_cutoff_rows": con.execute("""
        SELECT COUNT(*) FROM ecommerce_transactions h JOIN applications a USING (customer_id)
        WHERE h.transaction_datetime >= a.decision_datetime
    """).fetchone()[0],
    "previous_at_or_after_cutoff_rows": con.execute("""
        SELECT COUNT(*) FROM previous_applications h JOIN applications a USING (customer_id)
        WHERE h.decision_datetime >= a.decision_datetime
    """).fetchone()[0],
    "ecommerce_orders": con.execute("""
        SELECT COUNT(DISTINCT order_id) FROM ecommerce_transactions
    """).fetchone()[0],
    "ecommerce_inconsistent_order_totals": con.execute("""
        SELECT COUNT(*) FROM (
            SELECT order_id FROM ecommerce_transactions
            GROUP BY 1 HAVING MIN(order_total) <> MAX(order_total)
                OR SUM(item_total) <> MAX(order_total)
        )
    """).fetchone()[0],
    "target_distribution": con.execute("""
        SELECT target, COUNT(*) FROM train_applications GROUP BY 1 ORDER BY 1
    """).fetchall(),
    "decision_min_max": con.execute("""
        SELECT MIN(decision_datetime), MAX(decision_datetime) FROM applications
    """).fetchone(),
}
with (DEST / "checks.json").open("w") as f:
    json.dump(checks, f, indent=2, default=str)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
for split, group in monthly.groupby("split"):
    axes[0, 0].plot(group["month"], group["applications"], marker="o", label=split)
axes[0, 0].set(title="Applications by decision month", ylabel="Count")
axes[0, 0].legend()
train_month = monthly[monthly.split == "train"]
axes[0, 1].plot(train_month["month"], 100 * train_month["target_rate"], marker="o")
axes[0, 1].set(title="Observed 90-DPD rate by decision month", ylabel="Percent")
cover = pd.DataFrame(coverage).pivot(index="table", columns="split", values="coverage_pct")
cover.plot.barh(ax=axes[1, 0], color=["#e69f00", "#0072b2"])
axes[1, 0].set(title="Customer coverage by source", xlabel="Percent", xlim=(0, 105))
miss = pd.DataFrame(missing).query("missing > 0").sort_values("missing_pct", ascending=False).head(12)
axes[1, 1].barh(miss["table"] + "." + miss["column"], miss["missing_pct"])
axes[1, 1].invert_yaxis()
axes[1, 1].set(title="Largest column missingness", xlabel="Percent")
for ax in axes.flat:
    if ax in axes[0]:
        ax.tick_params(axis="x", rotation=30)
fig.savefig(DEST / "audit_overview.png", dpi=160)
plt.close(fig)

print(json.dumps({"rows": inventory, "checks": checks, "coverage": coverage}, default=str, indent=2))
