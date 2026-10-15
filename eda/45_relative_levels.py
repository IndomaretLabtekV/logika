"""The price-premium lesson generalised: levels measured relative to their natural reference group, across the other
tables. Each candidate is a customer-level mean of log(value / group median) and is tested against the v3 residual."""
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import ROOT, calibrate, features, ks_nan, load, resid_test

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}
p = calibrate(y, sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000)
rng = np.random.default_rng(9)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(20)])
print(f"noise vs v3: chi2/df p95={np.percentile(null[:, 0], 95):.2f}  dKS p95={np.percentile(null[:, 2], 95):+.4f}")
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime")
prof = load("customer_profiles")

def rel(df, value, groups, name):
    d = df.filter(pl.col(value) > 0).with_columns(lv=pl.col(value).cast(pl.Float64).log())
    d = d.with_columns(z=pl.col("lv") - pl.col("lv").median().over(groups) if groups else pl.col("lv") - pl.col("lv").median())
    return d.group_by("customer_id").agg(pl.col("z").mean().alias(name))

prev = load("previous_applications").join(cut, on="customer_id").filter(pl.col("decision_datetime") < pl.col("cutoff"))
bur = load("bureau_observations").join(cut, on="customer_id").filter(pl.col("available_datetime") < pl.col("cutoff"))
rep = load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") < pl.col("cutoff"))
acc = load("account_behavior")
orders = (load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") < pl.col("cutoff"))
          .group_by("order_id").agg(pl.col("customer_id", "order_total", "payment_method").first(), n_items=pl.len()))
cands = [
    rel(prev, "requested_amount", ["term_months"], "rl_req_by_term"),
    rel(prev.filter(pl.col("decision") == "approved"), "scheduled_monthly_installment", ["term_months"], "rl_inst_by_term"),
    rel(bur, "outstanding_balance", ["bureau_provider"], "rl_bureau_out"),
    rel(bur, "revolving_credit_limit", ["bureau_provider"], "rl_bureau_limit"),
    rel(rep, "amount_due", [], "rl_amount_due"),
    rel(acc, "outflows", [], "rl_outflows"),
    rel(acc, "income_like_credits", [], "rl_income"),
    rel(orders, "order_total", ["n_items"], "rl_order_by_items"),
    rel(orders, "order_total", ["payment_method"], "rl_order_by_pm"),
    rel(apps.select("customer_id", "loan_amount", "loan_purpose"), "loan_amount", ["loan_purpose"], "rl_loan_by_purpose"),
    rel(apps.join(prof, on="customer_id").select("customer_id", "declared_monthly_income", "employment_type", "region"),
        "declared_monthly_income", ["employment_type", "region"], "rl_income_by_job_region"),
]
print("\ncandidate                    KS     chi2/df   dKS vs v3")
for c in cands:
    name = c.columns[1]
    x = tr.select("customer_id").join(c, on="customer_id", how="left", maintain_order="left")[name].to_numpy()
    chi, _, d = resid_test(y, p, x)
    flag = "  <-- above noise" if chi > np.percentile(null[:, 0], 95) and d > np.percentile(null[:, 2], 95) else ""
    print(f"{name:<26} {ks_nan(y, x):.4f}  {chi:7.2f}  {d:+.4f}{flag}")
