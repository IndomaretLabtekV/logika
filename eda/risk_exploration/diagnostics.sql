CREATE TEMP VIEW segmented AS
SELECT *,
 CASE WHEN bu_score IS NULL THEN 'unscored' WHEN bu_score<550 THEN '<550'
      WHEN bu_score<650 THEN '550-649' ELSE '650+' END AS score_band,
 CASE WHEN residual_cash_ratio IS NULL THEN 'unobserved' WHEN residual_cash_ratio<0 THEN 'deficit'
      ELSE 'surplus' END AS cash_state,
 CASE WHEN rep_dues IS NULL THEN 'no history' WHEN rep_worst_dpd>=30 THEN 'past 30+'
      ELSE 'past <30' END AS repayment_state,
 CASE WHEN ac_months IS NULL THEN 'unobserved' WHEN income_change3 IS NULL THEN 'short window'
      WHEN income_change3 < -0.2 THEN 'fall >20%' WHEN income_change3>0.2 THEN 'rise >20%'
      ELSE 'within 20%' END AS income_state,
 CASE WHEN spend_income90<0.1 THEN '<10%' WHEN spend_income90<0.3 THEN '10-30%'
      ELSE '30%+' END AS spend_band,
 CASE WHEN ac_months IS NULL THEN 'unobserved' WHEN ac_months<12 THEN '<12 months'
      ELSE '12+ months' END AS history_band,
 CASE WHEN month(decision_datetime)<=9 THEN 'early' ELSE 'late' END AS period
FROM features;

CREATE TEMP TABLE quality_checks AS
SELECT 'account ledger residual != 0' AS check_name,count(*) AS rows_checked,
       count(*) FILTER(WHERE abs(ledger_residual)>1) AS flagged FROM joint_months
UNION ALL SELECT 'adjacent account continuity residual > 1',count(*),
 count(*) FILTER(WHERE abs(opening_balance-previous_close)>1)
 FROM joint_months WHERE date_diff('month',previous_month,month)=1
UNION ALL SELECT 'account low balance days outside month',count(*),
 count(*) FILTER(WHERE low_balance_days<0 OR low_balance_days>days_in_month) FROM joint_months
UNION ALL SELECT 'missing account months inside observed span',count(*),
 count(*) FILTER(WHERE ac_missing_months>0) FROM features WHERE ac_months IS NOT NULL
UNION ALL SELECT 'stale account latest month lag > 1',count(*),
 count(*) FILTER(WHERE ac_latest_lag>1) FROM features WHERE ac_months IS NOT NULL
UNION ALL SELECT 'ecommerce spend exceeds same account outflow',count(*),
 count(*) FILTER(WHERE ecommerce_spending>outflows) FROM joint_months
UNION ALL SELECT 'less than 180 days since first observed commerce event',count(*),
 count(*) FILTER(WHERE ec_observed_span_days<180) FROM features
UNION ALL SELECT 'multi-provider debt sum exceeds maximum',count(*),
 count(*) FILTER(WHERE bu_debt_sum_overlap_bound>bu_debt_max) FROM features
UNION ALL SELECT 'account cutoff violation',count(*),
 count(*) FILTER(WHERE j.month+INTERVAL 1 MONTH>f.decision_datetime)
 FROM joint_months j JOIN features f USING(application_id);

CREATE TEMP TABLE lag_profiles AS
SELECT j.month_lag,f.target,count(*) AS customers,
 avg(j.income_like_credits/nullif(f.declared_monthly_income,0)) AS income_to_declared,
 avg(j.net_flow/nullif(f.declared_monthly_income,0)) AS net_to_declared,
 avg(j.low_balance_days/j.days_in_month) AS low_balance_fraction,
 avg(j.ecommerce_spending/nullif(f.declared_monthly_income,0)) AS commerce_to_income
FROM joint_months j JOIN features f USING(application_id)
WHERE f.split='train' AND j.month_lag<=18 GROUP BY 1,2;

CREATE TEMP TABLE seasonality AS
WITH centered AS (
 SELECT *,income_like_credits/nullif(avg(income_like_credits) OVER(PARTITION BY application_id),0) AS income_relative,
 ecommerce_spending/nullif(avg(ecommerce_spending) OVER(PARTITION BY application_id),0) AS spending_naive,
 CASE WHEN commerce_history_started THEN ecommerce_spending/nullif(avg(ecommerce_spending)
 FILTER(WHERE commerce_history_started) OVER(PARTITION BY application_id),0) END AS spending_relative
 FROM joint_months)
SELECT year(month) AS year,month(month) AS calendar_month,count(*) AS observations,
 count(DISTINCT application_id) AS customers,avg(income_relative) AS within_customer_income,
 avg(spending_relative) AS within_customer_spending,avg(spending_naive) AS naive_within_customer_spending,
 count(spending_relative) AS supported_spending_observations,
 avg(low_balance_days/days_in_month) AS low_balance_fraction
FROM centered GROUP BY 1,2;

CREATE TEMP TABLE account_transitions AS
WITH states AS (
 SELECT j.*,lag(net_flow) OVER(PARTITION BY application_id ORDER BY month) AS previous_net
 FROM joint_months j)
SELECT f.target,CASE WHEN previous_net<0 THEN 'deficit' ELSE 'surplus' END AS from_state,
 CASE WHEN net_flow<0 THEN 'deficit' ELSE 'surplus' END AS to_state,
 count(*) AS month_pairs,count(DISTINCT s.application_id) AS customers
FROM states s JOIN features f USING(application_id)
WHERE f.split='train' AND date_diff('month',previous_month,month)=1 GROUP BY 1,2,3;

CREATE TEMP TABLE category_network AS
WITH pairs AS (
 SELECT a.product_category AS category_a,b.product_category AS category_b,count(*) AS shared_customers
 FROM category_edges a JOIN category_edges b USING(application_id)
 JOIN features f USING(application_id)
 WHERE a.product_category<b.product_category AND f.split='train' GROUP BY 1,2),
 degrees AS (SELECT product_category,count(*) AS customers FROM category_edges e JOIN features f USING(application_id)
 WHERE f.split='train' GROUP BY 1), total AS(SELECT count(*) AS n FROM features WHERE split='train')
SELECT p.*,a.customers AS customers_a,b.customers AS customers_b,
 shared_customers::DOUBLE/(a.customers+b.customers-shared_customers) AS jaccard,
 shared_customers*n::DOUBLE/(a.customers*b.customers) AS cooccurrence_lift
FROM pairs p JOIN degrees a ON p.category_a=a.product_category
JOIN degrees b ON p.category_b=b.product_category CROSS JOIN total;

CREATE TEMP TABLE region_category_network AS
SELECT f.region,e.product_category,count(*) AS customers,sum(e.spend) AS recorded_spending,
 avg(e.spend/nullif(12*f.declared_monthly_income,0)) AS spend_to_annual_income
FROM category_edges e JOIN features f USING(application_id) WHERE f.split='train' GROUP BY 1,2;

CREATE TEMP TABLE geographic_cohorts AS
SELECT region,employment_type,score_band,count(*) AS n,sum(target) AS events,avg(target) AS rate,
 avg(installment_income) AS installment_income,avg(residual_cash_ratio) AS residual_cash,
 avg((ac_months IS NOT NULL)::INTEGER) AS account_coverage
FROM segmented WHERE split='train' GROUP BY 1,2,3;

CREATE TEMP TABLE region_standardized AS
WITH weights AS(SELECT score_band,count(*)::DOUBLE/(SELECT count(*) FROM features WHERE split='train') AS weight
 FROM segmented WHERE split='train' GROUP BY 1), cells AS(
 SELECT region,score_band,count(*) AS n,avg(target) AS rate FROM segmented WHERE split='train' GROUP BY 1,2)
SELECT region,sum(rate*weight) AS score_standardized_rate,min(n) AS smallest_stratum_n
FROM cells JOIN weights USING(score_band) GROUP BY 1;

CREATE TEMP TABLE coverage_patterns AS
SELECT (ac_months IS NOT NULL) AS account_present,(rep_dues IS NOT NULL) AS repayment_present,
 (prev_count IS NOT NULL) AS prior_present,split,count(*) AS n,avg(target) AS rate
FROM features GROUP BY 1,2,3,4;

CREATE TEMP TABLE stress_scenarios AS
SELECT shock,employment_type,count(*) AS customers,
 avg((ac_net3-proposed_monthly_installment-shock*ac_income3<0)::INTEGER) AS deficit_share,
 avg((ac_net3-proposed_monthly_installment-shock*ac_income3<0)::INTEGER)
  -avg((ac_net3-proposed_monthly_installment<0)::INTEGER) AS newly_deficit_share
FROM features CROSS JOIN (VALUES(0.0),(0.1),(0.2),(0.3)) AS s(shock)
WHERE split='train' AND ac_n3=3 GROUP BY 1,2;
