CREATE TEMP TABLE items AS
SELECT a.application_id, a.decision_datetime AS cutoff, e.*
FROM applications a JOIN ecommerce_transactions e USING(customer_id)
WHERE transaction_datetime<a.decision_datetime;

CREATE TEMP TABLE orders AS
SELECT application_id,cutoff,customer_id,order_id,transaction_datetime,payment_method,order_status,
       max(order_total) AS amount, count(*) AS basket_items
FROM items GROUP BY 1,2,3,4,5,6,7;

CREATE TEMP TABLE order_sequence AS
SELECT *, epoch(transaction_datetime-lag(transaction_datetime) OVER w)/86400 AS gap_days
FROM orders WHERE order_status='completed'
WINDOW w AS(PARTITION BY application_id ORDER BY transaction_datetime,order_id);

CREATE TEMP TABLE commerce_features AS
SELECT application_id,count(*) AS ec_orders,
       avg((order_status='cancelled')::INTEGER) AS ec_cancel_share,
       count(*) FILTER(WHERE order_status='completed') AS ec_completed,
       count(*) FILTER(WHERE order_status='completed' AND transaction_datetime>=cutoff-INTERVAL 30 DAY) AS ec_count30,
       count(*) FILTER(WHERE order_status='completed' AND transaction_datetime>=cutoff-INTERVAL 90 DAY) AS ec_count90,
       coalesce(sum(amount) FILTER(WHERE order_status='completed' AND transaction_datetime>=cutoff-INTERVAL 30 DAY),0) AS ec_spend30,
       coalesce(sum(amount) FILTER(WHERE order_status='completed' AND transaction_datetime>=cutoff-INTERVAL 90 DAY),0) AS ec_spend90,
       coalesce(sum(amount) FILTER(WHERE order_status='completed' AND transaction_datetime>=cutoff-INTERVAL 180 DAY
         AND transaction_datetime<cutoff-INTERVAL 90 DAY),0) AS ec_spend_prior90,
       date_diff('day',max(transaction_datetime) FILTER(WHERE order_status='completed'),max(cutoff)) AS ec_recency,
       date_diff('day',min(transaction_datetime),max(cutoff)) AS ec_observed_span_days,
       avg((payment_method='credit_card')::INTEGER) FILTER(WHERE order_status='completed') AS ec_credit_share,
       avg((payment_method='COD')::INTEGER) FILTER(WHERE order_status='completed') AS ec_cod_share,
       avg((day(transaction_datetime)>=25)::INTEGER) FILTER(WHERE order_status='completed') AS ec_monthend_share,
       avg((hour(transaction_datetime)<6)::INTEGER) FILTER(WHERE order_status='completed') AS ec_night_share,
       avg(basket_items) FILTER(WHERE order_status='completed') AS ec_basket_items,
       sqrt(pow(avg(cos(2*pi()*dayofweek(transaction_datetime)/7)) FILTER(WHERE order_status='completed'),2)
           +pow(avg(sin(2*pi()*dayofweek(transaction_datetime)/7)) FILTER(WHERE order_status='completed'),2)) AS ec_weekday_concentration
FROM orders GROUP BY 1;

CREATE TEMP TABLE cadence AS
SELECT application_id,count(gap_days) AS ec_gap_support,
       avg(gap_days) AS ec_gap_mean, stddev_pop(gap_days)/nullif(avg(gap_days),0) AS ec_gap_cv,
       avg((gap_days BETWEEN 6 AND 8)::INTEGER) AS ec_weekly_gap_share,
       avg((gap_days BETWEEN 27 AND 33)::INTEGER) AS ec_monthly_gap_share
FROM order_sequence GROUP BY 1;

CREATE TEMP TABLE category_edges AS
SELECT application_id,product_category,sum(item_total) AS spend,count(DISTINCT order_id) AS orders,
       sum(item_total) FILTER(WHERE transaction_datetime>=cutoff-INTERVAL 90 DAY) AS spend90
FROM items WHERE order_status='completed' GROUP BY 1,2;

CREATE TEMP TABLE category_shares AS
SELECT *,spend/nullif(sum(spend) OVER(PARTITION BY application_id),0) AS share
FROM category_edges;

CREATE TEMP TABLE category_features AS
SELECT application_id,count(*) AS graph_category_degree,
       -sum(share*ln(nullif(share,0))) AS graph_category_entropy,
       sum(share*share) AS graph_category_hhi,
       max(share) AS graph_category_dominance,
       sum(share) FILTER(WHERE product_category IN ('groceries','health','education','personal_care'))
         AS ec_basic_category_share,
       coalesce(sum(spend90) FILTER(WHERE product_category='health'),0) AS ec_health_spend90,
       coalesce(sum(spend90) FILTER(WHERE product_category='electronics'),0) AS ec_electronics_spend90
FROM category_shares GROUP BY 1;

CREATE TEMP TABLE commerce_months AS
SELECT application_id,date_trunc('month',transaction_datetime)::DATE AS month,
       sum(amount) FILTER(WHERE order_status='completed') AS spending,
       count(*) FILTER(WHERE order_status='completed') AS completed_orders
FROM orders GROUP BY 1,2;

CREATE TEMP TABLE joint_months AS
SELECT p.*, coalesce(e.spending,0) AS ecommerce_spending,
       coalesce(e.completed_orders,0) AS ecommerce_orders,
       p.month>=o.first_month+INTERVAL 1 MONTH AS commerce_history_started
FROM account_panel p LEFT JOIN commerce_months e USING(application_id,month)
LEFT JOIN (SELECT application_id,min(month) AS first_month FROM commerce_months GROUP BY 1) o USING(application_id);

CREATE TEMP TABLE cross_features AS
SELECT application_id,corr(ecommerce_spending,net_flow) FILTER(WHERE commerce_history_started) AS cross_spend_net_corr,
       count(*) FILTER(WHERE commerce_history_started) AS cross_supported_months,
       avg(ecommerce_spending/nullif(outflows,0)) FILTER(WHERE month_lag<=3 AND commerce_history_started) AS cross_spend_outflow3,
       count(*) FILTER(WHERE ecommerce_spending>outflows) AS cross_spend_exceeds_outflow_months
FROM joint_months GROUP BY 1;

CREATE TEMP TABLE features AS
SELECT b.*, ac.* EXCLUDE(application_id),p.* EXCLUDE(application_id),
       r.* EXCLUDE(application_id),bu.* EXCLUDE(application_id),e.* EXCLUDE(application_id),
       ca.* EXCLUDE(application_id),g.* EXCLUDE(application_id),c.* EXCLUDE(application_id),
       ac_income3/nullif(declared_monthly_income,0) AS income_observed_declared,
       proposed_monthly_installment/nullif(ac_income3,0) AS installment_observed_income,
       ac_latest_close/nullif(proposed_monthly_installment,0) AS liquidity_installments,
       (ac_net3-proposed_monthly_installment)/nullif(ac_income3,0) AS residual_cash_ratio,
       (ac_net3-proposed_monthly_installment-0.2*ac_income3)/nullif(ac_income3,0) AS stress20_cash_ratio,
       CASE WHEN ac_n3=3 AND ac_nprior3=3 THEN ac_income3/nullif(ac_income_prior3,0)-1 END AS income_change3,
       CASE WHEN ac_n3=3 AND ac_nprior3=3 THEN (ac_net3-ac_net_prior3)/nullif(declared_monthly_income,0) END AS net_change3,
       rep_shortfall/nullif(ac_latest_close,0) AS arrears_to_liquidity,
       (proposed_monthly_installment+coalesce(prev_scheduled_active_proxy,0))/nullif(declared_monthly_income,0)
         AS scheduled_burden_proxy,
       bu_debt_max/nullif(12*declared_monthly_income,0) AS debt_annual_income,
       ec_spend90/nullif(3*declared_monthly_income,0) AS spend_income90,
       ec_spend30/nullif(ec_spend90/3,0) AS spend_recent_acceleration,
       ec_spend90/nullif(ec_spend_prior90,0) AS spend_ratio90,
       sin(2*pi()*(month(decision_datetime)-1)/12) AS decision_month_sin,
       cos(2*pi()*(month(decision_datetime)-1)/12) AS decision_month_cos,
       sin(2*pi()*dayofweek(decision_datetime)/7) AS decision_weekday_sin,
       cos(2*pi()*dayofweek(decision_datetime)/7) AS decision_weekday_cos
FROM base b LEFT JOIN account_features ac USING(application_id)
LEFT JOIN prior_features p USING(application_id) LEFT JOIN repay_features r USING(application_id)
LEFT JOIN bureau_features bu USING(application_id) LEFT JOIN commerce_features e USING(application_id)
LEFT JOIN cadence ca USING(application_id) LEFT JOIN category_features g USING(application_id)
LEFT JOIN cross_features c USING(application_id);
