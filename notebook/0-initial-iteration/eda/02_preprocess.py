import json

import pandas as pd

from common import OUT, connect


DEST = OUT / "02_preprocess"
DEST.mkdir(parents=True, exist_ok=True)
con = connect()

con.execute("""
    CREATE TEMP TABLE base AS
    SELECT a.application_id, a.customer_id, a.split, a.target,
           a.decision_datetime, EXTRACT(MONTH FROM a.decision_datetime) AS decision_month,
           a.loan_amount, a.term_months, a.annual_interest_rate,
           a.proposed_monthly_installment, a.loan_purpose,
           p.age, p.region, p.employment_type, p.employment_tenure_months,
           p.declared_monthly_income, p.dependents, p.housing_status,
           a.proposed_monthly_installment / NULLIF(p.declared_monthly_income, 0)
               AS installment_income_ratio,
           a.loan_amount / NULLIF(12 * p.declared_monthly_income, 0)
               AS loan_annual_income_ratio
    FROM applications a LEFT JOIN customer_profiles p USING (customer_id)
""")

con.execute("""
    CREATE TEMP TABLE previous_eligible AS
    SELECT a.application_id AS selected_application_id, h.*,
           a.decision_datetime AS selected_decision_datetime
    FROM applications a JOIN previous_applications h USING (customer_id)
    WHERE h.decision_datetime < a.decision_datetime
""")
con.execute("""
    CREATE TEMP TABLE previous_features AS
    SELECT selected_application_id AS application_id,
           COUNT(*) AS previous_count,
           COUNT(*) FILTER (WHERE decision = 'approved') AS previous_approved_count,
           COUNT(*) FILTER (WHERE decision = 'rejected') AS previous_rejected_count,
           COUNT(*) FILTER (WHERE decision = 'cancelled') AS previous_cancelled_count,
           AVG(requested_amount) AS previous_requested_mean,
           AVG(approved_amount) AS previous_approved_amount_mean,
           date_diff('day', MAX(decision_datetime), MAX(selected_decision_datetime))
               AS previous_recency_days
    FROM previous_eligible GROUP BY 1
""")

con.execute("""
    CREATE TEMP TABLE repayment_features AS
    SELECT p.selected_application_id AS application_id,
           COUNT(DISTINCT r.loan_id) AS historical_loan_count,
           COUNT(*) AS due_installment_count,
           AVG(r.days_past_due) AS repayment_mean_dpd,
           MAX(r.max_days_past_due) AS repayment_max_historical_dpd,
           AVG(CAST(r.days_past_due > 0 AS DOUBLE)) AS repayment_late_share,
           AVG(CAST(r.days_past_due >= 30 AS DOUBLE)) AS repayment_30dpd_share,
           SUM(GREATEST(r.amount_due - r.amount_paid, 0)) AS repayment_due_shortfall,
           SUM(r.amount_paid) / NULLIF(SUM(r.amount_due), 0) AS repayment_paid_due_ratio
    FROM previous_eligible p JOIN repayment_history r
      ON p.loan_id = r.loan_id AND p.customer_id = r.customer_id
    WHERE p.decision = 'approved'
      AND r.due_date < p.selected_decision_datetime
      AND (r.last_payment_datetime IS NULL
           OR r.last_payment_datetime < p.selected_decision_datetime)
    GROUP BY 1
""")

con.execute("""
    CREATE TEMP TABLE account_features AS
    SELECT a.application_id, COUNT(*) AS account_complete_months,
           AVG(h.average_balance) AS account_average_balance,
           AVG(h.minimum_balance) AS account_minimum_balance_mean,
           AVG(h.low_balance_days) AS account_low_balance_days_mean,
           STDDEV_POP(h.inflows - h.outflows) AS account_net_flow_volatility,
           AVG(h.inflows - h.outflows) AS account_net_flow_mean,
           AVG(h.income_like_credits) AS account_income_like_mean,
           AVG(h.average_balance) FILTER (
               WHERE h.month >= CAST(a.decision_datetime - INTERVAL 3 MONTH AS DATE)
           ) AS account_3m_average_balance,
           AVG(h.low_balance_days) FILTER (
               WHERE h.month >= CAST(a.decision_datetime - INTERVAL 3 MONTH AS DATE)
           ) AS account_3m_low_balance_days_mean,
           AVG(h.inflows - h.outflows) FILTER (
               WHERE h.month >= CAST(a.decision_datetime - INTERVAL 6 MONTH AS DATE)
           ) AS account_6m_net_flow_mean
    FROM applications a JOIN account_behavior h USING (customer_id)
    WHERE CAST(h.month AS TIMESTAMP) + INTERVAL 1 MONTH <= a.decision_datetime
    GROUP BY 1
""")

con.execute("""
    CREATE TEMP TABLE bureau_eligible AS
    SELECT b.*, a.application_id AS selected_application_id, a.decision_datetime,
           ROW_NUMBER() OVER (
               PARTITION BY a.application_id, b.bureau_provider
               ORDER BY b.available_datetime DESC, b.report_datetime DESC,
                        b.bureau_observation_id DESC
           ) AS provider_rank
    FROM bureau_observations b JOIN applications a USING (customer_id)
    WHERE b.available_datetime <= a.decision_datetime
      AND b.report_datetime <= a.decision_datetime
""")
con.execute("""
    CREATE TEMP TABLE bureau_features AS
    SELECT selected_application_id AS application_id,
           COUNT(*) AS bureau_report_count,
           COUNT(*) FILTER (WHERE application_id <> selected_application_id)
               AS bureau_prior_report_count,
           COUNT(*) FILTER (WHERE provider_rank = 1) AS bureau_provider_count,
           AVG(score) FILTER (WHERE provider_rank = 1) AS bureau_score_mean,
           MIN(score) FILTER (WHERE provider_rank = 1) AS bureau_score_min,
           MAX(delinquent_accounts_last_12m) FILTER (WHERE provider_rank = 1)
               AS bureau_delinquent_accounts_max,
           MAX(inquiries_last_6m) FILTER (WHERE provider_rank = 1)
               AS bureau_inquiries_6m_max,
           AVG(utilization) FILTER (WHERE provider_rank = 1) AS bureau_utilization_mean,
           MAX(outstanding_balance) FILTER (WHERE provider_rank = 1)
               AS bureau_outstanding_max,
           MAX(revolving_outstanding_balance) FILTER (WHERE provider_rank = 1)
               AS bureau_revolving_balance_max,
           MIN(date_diff('day', available_datetime, decision_datetime))
               AS bureau_freshest_age_days,
           COUNT(*) FILTER (WHERE provider_rank = 1 AND score IS NULL)
               AS bureau_unscored_provider_count
    FROM bureau_eligible GROUP BY 1
""")

con.execute("""
    CREATE TEMP TABLE valid_items AS
    SELECT a.application_id, a.decision_datetime, h.order_id,
           h.transaction_datetime, h.product_category, h.item_total,
           h.order_total, h.payment_method, h.order_status
    FROM applications a JOIN ecommerce_transactions h USING (customer_id)
    WHERE h.transaction_datetime < a.decision_datetime
""")
con.execute("""
    CREATE TEMP TABLE order_events AS
    SELECT application_id, decision_datetime, order_id, transaction_datetime,
           payment_method, order_status, MAX(order_total) AS order_total
    FROM valid_items
    GROUP BY 1, 2, 3, 4, 5, 6
""")
con.execute("""
    CREATE TEMP TABLE ecommerce_orders AS
    SELECT application_id, COUNT(*) AS ecommerce_order_count,
           COUNT(*) FILTER (WHERE order_status = 'completed') AS ecommerce_completed_count,
           COUNT(*) FILTER (WHERE order_status = 'cancelled') AS ecommerce_cancelled_count,
           SUM(order_total) FILTER (WHERE order_status = 'completed') AS ecommerce_spend,
           SUM(order_total) FILTER (
               WHERE order_status = 'completed'
                 AND transaction_datetime >= decision_datetime - INTERVAL 90 DAY
           ) AS ecommerce_90d_spend,
           COUNT(*) FILTER (
               WHERE order_status = 'completed'
                 AND transaction_datetime >= decision_datetime - INTERVAL 90 DAY
           ) AS ecommerce_90d_orders,
           AVG(CAST(payment_method = 'COD' AS DOUBLE)) AS ecommerce_cod_share,
           AVG(CAST(payment_method = 'credit_card' AS DOUBLE)) AS ecommerce_credit_card_share,
           MAX(transaction_datetime) AS last_ecommerce_datetime,
           MAX(decision_datetime) AS decision_datetime
    FROM order_events GROUP BY 1
""")
con.execute("""
    CREATE TEMP TABLE ecommerce_items AS
    SELECT application_id,
           COUNT(DISTINCT product_category) FILTER (
               WHERE order_status = 'completed') AS ecommerce_unique_categories,
           SUM(item_total) FILTER (
               WHERE order_status = 'completed' AND product_category = 'electronics'
           ) AS ecommerce_electronics_spend
    FROM valid_items GROUP BY 1
""")

con.execute("""
    CREATE TEMP TABLE features AS
    SELECT b.*,
           COALESCE(p.previous_count, 0) AS previous_count,
           COALESCE(p.previous_approved_count, 0) AS previous_approved_count,
           COALESCE(p.previous_rejected_count, 0) AS previous_rejected_count,
           COALESCE(p.previous_cancelled_count, 0) AS previous_cancelled_count,
           p.previous_requested_mean, p.previous_approved_amount_mean,
           p.previous_recency_days,
           COALESCE(r.historical_loan_count, 0) AS historical_loan_count,
           COALESCE(r.due_installment_count, 0) AS due_installment_count,
           r.repayment_mean_dpd, r.repayment_max_historical_dpd,
           r.repayment_late_share, r.repayment_30dpd_share,
           r.repayment_due_shortfall, r.repayment_paid_due_ratio,
           COALESCE(ac.account_complete_months, 0) AS account_complete_months,
           ac.account_average_balance, ac.account_minimum_balance_mean,
           ac.account_low_balance_days_mean, ac.account_net_flow_volatility,
           ac.account_net_flow_mean, ac.account_income_like_mean,
           ac.account_3m_average_balance, ac.account_3m_low_balance_days_mean,
           ac.account_6m_net_flow_mean,
           COALESCE(bu.bureau_report_count, 0) AS bureau_report_count,
           COALESCE(bu.bureau_prior_report_count, 0) AS bureau_prior_report_count,
           COALESCE(bu.bureau_provider_count, 0) AS bureau_provider_count,
           bu.bureau_score_mean, bu.bureau_score_min,
           bu.bureau_delinquent_accounts_max, bu.bureau_inquiries_6m_max,
           bu.bureau_utilization_mean, bu.bureau_outstanding_max,
           bu.bureau_revolving_balance_max, bu.bureau_freshest_age_days,
           bu.bureau_unscored_provider_count,
           COALESCE(eo.ecommerce_order_count, 0) AS ecommerce_order_count,
           COALESCE(eo.ecommerce_completed_count, 0) AS ecommerce_completed_count,
           COALESCE(eo.ecommerce_cancelled_count, 0) AS ecommerce_cancelled_count,
           eo.ecommerce_spend, eo.ecommerce_90d_spend,
           COALESCE(eo.ecommerce_90d_orders, 0) AS ecommerce_90d_orders,
           eo.ecommerce_cod_share, eo.ecommerce_credit_card_share,
           date_diff('day', eo.last_ecommerce_datetime, b.decision_datetime)
               AS ecommerce_recency_days,
           ei.ecommerce_unique_categories, ei.ecommerce_electronics_spend
    FROM base b
    LEFT JOIN previous_features p USING (application_id)
    LEFT JOIN repayment_features r USING (application_id)
    LEFT JOIN account_features ac USING (application_id)
    LEFT JOIN bureau_features bu USING (application_id)
    LEFT JOIN ecommerce_orders eo USING (application_id)
    LEFT JOIN ecommerce_items ei USING (application_id)
""")

counts = con.execute("""
    SELECT split, COUNT(*), COUNT(DISTINCT application_id),
           SUM(CAST(declared_monthly_income IS NULL AS INT)),
           SUM(CAST(bureau_provider_count = 0 AS INT))
    FROM features GROUP BY 1 ORDER BY 1
""").fetchall()
assert counts == [("test", 30000, 30000, 0, 0), ("train", 70000, 70000, 0, 0)], counts

con.execute(f"""
    COPY (SELECT * EXCLUDE (split, customer_id, decision_datetime)
          FROM features WHERE split = 'train')
    TO '{DEST / 'train_features.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
con.execute(f"""
    COPY (SELECT * EXCLUDE (split, customer_id, decision_datetime, target)
          FROM features WHERE split = 'test')
    TO '{DEST / 'test_features.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
con.execute(f"""
    COPY (SELECT application_id, customer_id, split, decision_datetime, target
          FROM applications)
    TO '{DEST / 'split_manifest.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

columns = con.execute("DESCRIBE features").fetchall()
features = [r[0] for r in columns if r[0] not in {
    "application_id", "customer_id", "split", "target", "decision_datetime"}]
null_expr = ", ".join(
    f'AVG(CAST("{name}" IS NULL AS DOUBLE)) AS "{name}"' for name in features
)
nulls = con.execute(f"SELECT split, {null_expr} FROM features GROUP BY split").df()
nulls.melt(id_vars="split", var_name="feature", value_name="missing_fraction").to_csv(
    DEST / "feature_missingness.csv", index=False
)

summary = {
    "split_counts": counts,
    "feature_count": len(features),
    "excluded_from_model": ["application_id", "customer_id", "split", "decision_datetime"],
    "note": "No model fitted; missing continuous values remain null and event counts without history are zero.",
    "order_item_rows": con.execute("SELECT COUNT(*) FROM valid_items").fetchone()[0],
    "unique_order_events": con.execute("SELECT COUNT(*) FROM order_events").fetchone()[0],
}
with (DEST / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
