CREATE TEMP TABLE source_contracts AS
SELECT 'previous application ID duplicate' AS check_name,
 count(*)-count(DISTINCT application_id) AS flagged FROM previous_applications
UNION ALL SELECT 'originated loan ID duplicate',count(*)-count(DISTINCT loan_id)
 FROM previous_applications WHERE loan_id IS NOT NULL
UNION ALL SELECT 'account customer-month duplicate',count(*)-count(DISTINCT (customer_id,month)) FROM account_behavior
UNION ALL SELECT 'repayment installment ID duplicate',count(*)-count(DISTINCT installment_id) FROM repayment_history
UNION ALL SELECT 'repayment without matching owned approved loan',count(*) FROM repayment_history r
 WHERE NOT EXISTS(SELECT 1 FROM previous_applications p WHERE r.loan_id=p.loan_id
 AND r.customer_id=p.customer_id AND p.decision='approved')
UNION ALL SELECT 'inconsistent order attributes',count(*) FROM(
 SELECT order_id FROM ecommerce_transactions GROUP BY 1
 HAVING count(DISTINCT(customer_id,transaction_datetime,payment_method,order_status,order_total))>1)
UNION ALL SELECT 'order total does not reconcile to item sum',count(*) FROM(
 SELECT order_id FROM ecommerce_transactions GROUP BY 1 HAVING sum(item_total)<>max(order_total))
UNION ALL SELECT 'repayment max DPD below current DPD',count(*) FROM repayment_history WHERE max_days_past_due<days_past_due;

CREATE TEMP TABLE eligibility_audit AS
SELECT 'previous' AS source,count(*) AS source_rows,
 count(*) FILTER(WHERE p.decision_datetime<a.decision_datetime) AS eligible_rows
FROM applications a JOIN previous_applications p USING(customer_id)
UNION ALL SELECT 'repayment cutoff',count(*),count(*) FILTER(WHERE r.due_date<a.decision_datetime
 AND (r.last_payment_datetime IS NULL OR r.last_payment_datetime<a.decision_datetime))
FROM applications a JOIN repayment_history r USING(customer_id)
UNION ALL SELECT 'account',count(*),count(*) FILTER(WHERE month+INTERVAL 1 MONTH<=decision_datetime)
FROM applications JOIN account_behavior USING(customer_id)
UNION ALL SELECT 'bureau',count(*),count(*) FILTER(WHERE report_datetime<=decision_datetime AND available_datetime<=decision_datetime)
FROM applications JOIN bureau_observations USING(customer_id)
UNION ALL SELECT 'commerce items',count(*),count(*) FILTER(WHERE transaction_datetime<decision_datetime)
FROM applications JOIN ecommerce_transactions USING(customer_id);

CREATE TEMP TABLE relationship_inventory AS
SELECT 'customer -> originated loan' AS relation,count(*) AS edges,count(DISTINCT customer_id) AS source_nodes,
 count(DISTINCT loan_id) AS destination_nodes FROM previous_applications WHERE loan_id IS NOT NULL
UNION ALL SELECT 'loan -> installment',count(*),count(DISTINCT loan_id),count(DISTINCT installment_id) FROM repayment_history
UNION ALL SELECT 'customer -> category',count(*),count(DISTINCT application_id),count(DISTINCT product_category) FROM category_edges
UNION ALL SELECT 'customer -> fictional region',count(*),count(DISTINCT customer_id),count(DISTINCT region) FROM customer_profiles;
