"""Duplicates (exact / near), ID leakage, and cross-split twins that could inflate CV."""
import numpy as np
import polars as pl
from common import features, ks, load

# 1. ecommerce duplicated item rows: legit repeated items or double-logging?
e = load("ecommerce_transactions")
dup = e.filter(e.drop("order_item_id").is_duplicated())
print("dup item rows:", dup.height, "orders affected:", dup["order_id"].n_unique())
chk = e.group_by("order_id").agg(s=pl.col("item_total").sum(), t=pl.col("order_total").first())
bad = chk.filter(pl.col("s") != pl.col("t"))
print("orders where sum(item_total) != order_total:", bad.height,
      "| of which contain dup rows:", bad.filter(pl.col("order_id").is_in(dup["order_id"].implode())).height)
print("order-level fields constant within order:",
      e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "payment_method", "order_status").n_unique())
      .select(pl.exclude("order_id").max()).row(0))

# 2. ID leakage: do hex ids encode anything about target?
f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
for c in ["application_id", "customer_id"]:
    v = tr[c].str.slice(2, 8).str.to_integer(base=16).to_numpy()
    print(f"KS of {c} hex prefix vs target: {ks(y, v):.4f}")
print("KS of decision_datetime vs target:", round(ks(y, tr["decision_datetime"].dt.epoch("s").to_numpy()), 4))

# 3. exact feature-twins (same profile + same loan) within and across splits
key = ["age", "region", "employment_type", "employment_tenure_months", "declared_monthly_income", "dependents",
       "housing_status"]
tw = f.group_by(key).agg(n=pl.len(), splits=pl.col("is_train").n_unique()).filter(pl.col("n") > 1)
print("profile twins:", tw.height, "cross-split:", tw.filter(pl.col("splits") > 1).height)
num = [c for c, d in f.schema.items() if d.is_numeric() and c not in ("target", "is_train")]
r = f.select(pl.col(num).cast(pl.Float64).round(6))
print("exact duplicated full feature rows:", r.is_duplicated().sum())

# 4. near-duplicates: nearest neighbour in rank space on top features, train<->test
top = ["bc_score_mean", "rp_frac_dpd1", "ec_accel_30d", "annual_interest_rate", "loan_amount", "declared_monthly_income",
       "ec_spend", "age"]
X = f.select([pl.col(c).rank() / pl.len() for c in top]).fill_null(-1).to_numpy()
idx_tr, idx_te = np.where(f["is_train"].to_numpy() == 1)[0], np.where(f["is_train"].to_numpy() == 0)[0]
rng = np.random.default_rng(0)
q = rng.choice(idx_te, 2000, replace=False)
dmin = np.array([np.abs(X[idx_tr] - X[i]).sum(1).min() for i in q])
qq = rng.choice(idx_tr, 2000, replace=False)
dmin_tr = np.array([np.sort(np.abs(X[idx_tr] - X[i]).sum(1))[1] for i in qq])
print("L1 nearest-train distance  test->train  p1/p50:", np.percentile(dmin, [1, 50]).round(4),
      "| train->train p1/p50:", np.percentile(dmin_tr, [1, 50]).round(4))

# 5. leakage smell test: any single feature with suspiciously high KS?
from common import ks_nan
hi = [(c, ks_nan(y, tr[c].cast(pl.Float64).to_numpy())) for c in num]
hi = sorted(hi, key=lambda t: -t[1])[:5]
print("top univariate KS (anything >0.6 would smell like leakage):", [(c, round(v, 3)) for c, v in hi])
