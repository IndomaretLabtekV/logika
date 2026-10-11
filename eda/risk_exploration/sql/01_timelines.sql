CREATE TEMP TABLE base AS
SELECT a.*, p.* EXCLUDE(customer_id),
       proposed_monthly_installment / nullif(declared_monthly_income, 0) AS installment_income,
       declared_monthly_income / (dependents + 1) AS income_per_person,
       date_diff('minute', application_datetime, decision_datetime) AS decision_minutes
FROM applications a JOIN customer_profiles p USING(customer_id);

CREATE TEMP TABLE account_panel AS
SELECT a.application_id, h.*,
       date_diff('month', h.month, date_trunc('month', a.decision_datetime)) AS month_lag,
       date_diff('day', h.month, h.month + INTERVAL 1 MONTH) AS days_in_month,
       h.inflows - h.outflows AS net_flow,
       h.closing_balance - h.opening_balance - h.inflows + h.outflows AS ledger_residual,
       lag(h.month) OVER w AS previous_month,
       lag(h.closing_balance) OVER w AS previous_close,
       lag(h.income_like_credits) OVER w AS previous_income
FROM applications a JOIN account_behavior h USING(customer_id)
WHERE h.month + INTERVAL 1 MONTH <= a.decision_datetime
WINDOW w AS (PARTITION BY a.application_id ORDER BY h.month);

CREATE TEMP TABLE account_features AS
SELECT application_id, count(*) AS ac_months,
       max(month_lag) AS ac_oldest_lag, min(month_lag) AS ac_latest_lag,
       max(month_lag)-min(month_lag)+1-count(*) AS ac_missing_months,
       count(*) FILTER(WHERE month_lag <= 3) AS ac_n3,
       count(*) FILTER(WHERE month_lag BETWEEN 4 AND 6) AS ac_nprior3,
       avg(income_like_credits) FILTER(WHERE month_lag <= 3) AS ac_income3,
       avg(income_like_credits) FILTER(WHERE month_lag BETWEEN 4 AND 6) AS ac_income_prior3,
       avg(net_flow) FILTER(WHERE month_lag <= 3) AS ac_net3,
       avg(net_flow) FILTER(WHERE month_lag BETWEEN 4 AND 6) AS ac_net_prior3,
       avg(outflows) FILTER(WHERE month_lag <= 3) AS ac_outflow3,
       min(minimum_balance) FILTER(WHERE month_lag <= 3) AS ac_minimum3,
       arg_min(closing_balance, month_lag) AS ac_latest_close,
       arg_min(income_like_credits, month_lag) AS ac_latest_income,
       sum(low_balance_days) / sum(days_in_month) AS ac_low_day_fraction,
       sum(low_balance_days) FILTER(WHERE month_lag <= 3) /
         sum(days_in_month) FILTER(WHERE month_lag <= 3) AS ac_low_day_fraction3,
       avg((net_flow < 0)::INTEGER) AS ac_negative_month_share,
       stddev_pop(income_like_credits) / nullif(avg(income_like_credits),0) AS ac_income_cv,
       regr_slope(income_like_credits, -month_lag) AS ac_income_slope,
       regr_slope(closing_balance, -month_lag) AS ac_balance_slope,
       corr(income_like_credits, previous_income) FILTER(
         WHERE date_diff('month', previous_month, month)=1) AS ac_income_lag1_corr,
       count(*) FILTER(WHERE date_diff('month', previous_month, month)=1) AS ac_adjacent_pairs,
       max(abs(ledger_residual)) AS ac_ledger_residual_max,
       max(abs(opening_balance-previous_close)) FILTER(
         WHERE date_diff('month',previous_month,month)=1) AS ac_continuity_residual_max
FROM account_panel GROUP BY 1;

CREATE TEMP TABLE prior AS
SELECT a.application_id AS selected_id, a.decision_datetime AS cutoff, p.*,
       lag(p.decision) OVER w AS prior_decision,
       lag(p.decision_datetime) OVER w AS prior_decision_time
FROM applications a JOIN previous_applications p USING(customer_id)
WHERE p.decision_datetime < a.decision_datetime
WINDOW w AS (PARTITION BY a.application_id ORDER BY p.decision_datetime,p.application_id);

CREATE TEMP TABLE prior_features AS
SELECT selected_id AS application_id, count(*) AS prev_count,
       count(*) FILTER(WHERE decision_datetime >= cutoff - INTERVAL 90 DAY) AS prev_count90,
       avg((decision='rejected')::INTEGER) AS prev_rejected_share,
       count(*) FILTER(WHERE decision='approved' AND prior_decision='rejected') AS prev_reject_approve,
       min(date_diff('day',prior_decision_time,decision_datetime)) AS prev_min_gap_days,
       arg_max(requested_amount,decision_datetime) AS prev_latest_requested,
       sum(scheduled_monthly_installment) FILTER(WHERE decision='approved'
         AND decision_datetime + term_months * INTERVAL 1 MONTH > cutoff) AS prev_scheduled_active_proxy
FROM prior GROUP BY 1;

CREATE TEMP TABLE repay AS
SELECT p.selected_id AS application_id, p.cutoff, r.*,
       date_diff('day',due_date,cutoff) AS due_age,
       greatest(amount_due-amount_paid,0) AS shortfall,
       lag((days_past_due>0)::INTEGER) OVER w AS older_due_unpaid
FROM prior p JOIN repayment_history r ON p.loan_id=r.loan_id AND p.customer_id=r.customer_id
WHERE p.decision='approved' AND r.due_date < p.cutoff
  AND (last_payment_datetime IS NULL OR last_payment_datetime < p.cutoff)
WINDOW w AS (PARTITION BY p.selected_id,r.loan_id ORDER BY due_date,installment_id);

CREATE TEMP TABLE repay_features AS
SELECT application_id, count(*) AS rep_dues, count(DISTINCT loan_id) AS rep_loans,
       max(max_days_past_due) AS rep_worst_dpd,
       max(days_past_due) AS rep_current_dpd,
       sum(shortfall) AS rep_shortfall,
       sum(shortfall) FILTER(WHERE due_age>=30) AS rep_shortfall_mature30,
       count(*) FILTER(WHERE due_age>=30) AS rep_mature_dues30,
       avg((max_days_past_due>=30)::INTEGER) FILTER(WHERE due_age>=30) AS rep_ever30_mature_share,
       avg((max_days_past_due>=30)::INTEGER) FILTER(WHERE due_age BETWEEN 30 AND 180)
         AS rep_ever30_recent_mature,
       avg((max_days_past_due>=30)::INTEGER) FILTER(WHERE due_age>180) AS rep_ever30_older,
       avg((days_past_due=0)::INTEGER) FILTER(WHERE max_days_past_due>=30) AS rep_cured_share,
       avg((days_past_due>0)::INTEGER) FILTER(WHERE older_due_unpaid=1) AS rep_adjacent_unpaid_share,
       count(*) FILTER(WHERE older_due_unpaid=1) AS rep_adjacent_unpaid_support,
       sum(amount_paid)/nullif(sum(amount_due),0) AS rep_paid_fraction
FROM repay GROUP BY 1;

CREATE TEMP TABLE bureau_panel AS
SELECT a.application_id AS selected_id, a.decision_datetime AS cutoff, b.*,
       row_number() OVER(PARTITION BY a.application_id,bureau_provider
         ORDER BY available_datetime DESC,report_datetime DESC,bureau_observation_id DESC) AS rn,
       lag(score) OVER w AS previous_score,
       lag(report_datetime) OVER w AS previous_report_time,
       lag(utilization) OVER w AS previous_utilization
FROM applications a JOIN bureau_observations b USING(customer_id)
WHERE available_datetime<=a.decision_datetime AND report_datetime<=a.decision_datetime
WINDOW w AS (PARTITION BY a.application_id,bureau_provider
 ORDER BY available_datetime,report_datetime,bureau_observation_id);

CREATE TEMP TABLE bureau_features AS
SELECT selected_id AS application_id, count(*) AS bu_reports,
       count(DISTINCT bureau_provider) AS bu_providers,
       count(*) FILTER(WHERE rn=1 AND score IS NOT NULL) AS bu_scored_providers,
       avg(score) FILTER(WHERE rn=1) AS bu_score,
       max(score) FILTER(WHERE rn=1)-min(score) FILTER(WHERE rn=1) AS bu_score_range,
       max(utilization) FILTER(WHERE rn=1) AS bu_utilization,
       max(outstanding_balance) FILTER(WHERE rn=1) AS bu_debt_max,
       sum(outstanding_balance) FILTER(WHERE rn=1) AS bu_debt_sum_overlap_bound,
       max(inquiries_last_6m) FILTER(WHERE rn=1) AS bu_inquiries,
       max(delinquent_accounts_last_12m) FILTER(WHERE rn=1) AS bu_delinquent,
       avg(date_diff('day',report_datetime,available_datetime)) AS bu_delivery_lag,
       max(date_diff('day',report_datetime,cutoff)) FILTER(WHERE rn=1) AS bu_stalest_days,
       avg((score-previous_score)*30.0/nullif(date_diff('day',previous_report_time,report_datetime),0))
         FILTER(WHERE rn=1 AND report_datetime>previous_report_time) AS bu_score_change_per30d,
       avg(utilization-previous_utilization) FILTER(WHERE rn=1 AND report_datetime>previous_report_time)
         AS bu_utilization_change,
       count(*) FILTER(WHERE rn=1 AND previous_score IS NOT NULL AND report_datetime>previous_report_time)
         AS bu_score_change_support
FROM bureau_panel GROUP BY 1;
