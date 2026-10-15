"""Does the preprocessing do what we think? (a) independent pandas re-computation on random customers,
(b) leakage injection: append rows dated AFTER each cutoff and confirm features do not move."""
import numpy as np
import pandas as pd
import polars as pl
from common import TABLES, features, load
from features import build_all

f = features().to_pandas().set_index("customer_id")
rng = np.random.default_rng(7)
hist = f.index[f["has_history"] == 1]
cust = rng.choice(hist, 300, replace=False)
ok = {}

rep = load("repayment_history").filter(pl.col("customer_id").is_in(list(cust))).to_pandas()
ec = load("ecommerce_transactions").filter(pl.col("customer_id").is_in(list(cust))).to_pandas()
bu = load("bureau_observations").filter(pl.col("customer_id").is_in(list(cust))).to_pandas()
ac = load("account_behavior").filter(pl.col("customer_id").is_in(list(cust))).to_pandas()
pv = load("previous_applications").filter(pl.col("customer_id").is_in(list(cust))).to_pandas()
for c in cust:
    cut, app = f.loc[c, "decision_datetime"], f.loc[c, "application_id"]
    r = rep[rep.customer_id == c]
    exp = (r.max_days_past_due > 0).mean() if len(r) else np.nan
    ok.setdefault("rp_frac_dpd1", []).append(np.isclose(exp, f.loc[c, "rp_frac_dpd1"], equal_nan=True))
    o = ec[ec.customer_id == c].drop_duplicates("order_id")
    age = (cut - o.transaction_datetime).dt.total_seconds() / 86400
    ok.setdefault("ec_orders_30d", []).append((age <= 30).sum() == f.loc[c, "ec_orders_30d"])
    ok.setdefault("ec_spend", []).append(o.loc[o.order_status == "completed", "order_total"].sum() == f.loc[c, "ec_spend"])
    b = bu[(bu.customer_id == c) & (bu.application_id == app)]
    ok.setdefault("bc_score_mean", []).append(np.isclose(b.score.mean(), f.loc[c, "bc_score_mean"], equal_nan=True))
    a = ac[ac.customer_id == c].sort_values("month").tail(3)
    ok.setdefault("ac_out_in_l3", []).append(np.isclose(a.outflows.sum() / a.inflows.sum(), f.loc[c, "ac_out_in_l3"]))
    p = pv[pv.customer_id == c]
    ok.setdefault("pv_n_rej", []).append((p.decision == "rejected").sum() == f.loc[c, "pv_n_rej"])
for k, v in ok.items():
    print(f"{k:<16} match {np.mean(v):.3f}")
    assert np.mean(v) == 1.0, k

# (b) leakage injection: future rows with extreme values must be ignored
T = {t: load(t) for t in TABLES}
apps = pl.concat([T["train_applications"], T["test_applications"]], how="diagonal_relaxed").select("customer_id", "decision_datetime")
fut = apps.sample(2000, seed=1)
d = pl.duration(days=5)
inj = {
    "repayment_history": T["repayment_history"].join(fut, on="customer_id").group_by("customer_id").first()
        .with_columns(due_date=pl.col("decision_datetime") + d, max_days_past_due=pl.lit(999), days_past_due=pl.lit(999),
                      installment_id=pl.lit("I_FAKE")).drop("decision_datetime"),
    "ecommerce_transactions": T["ecommerce_transactions"].join(fut, on="customer_id").group_by("customer_id").first()
        .with_columns(transaction_datetime=pl.col("decision_datetime") + pl.duration(hours=1), order_id=pl.lit("O_FAKE") + pl.col("customer_id"),
                      order_total=pl.lit(10**9)).drop("decision_datetime"),
    "bureau_observations": T["bureau_observations"].join(fut, on="customer_id").group_by("customer_id").first()
        .with_columns(available_datetime=pl.col("decision_datetime") + d, score=pl.lit(300.0)).drop("decision_datetime"),
    "account_behavior": T["account_behavior"].join(fut, on="customer_id").group_by("customer_id").first()
        .with_columns(month=pl.col("decision_datetime").dt.truncate("1mo").cast(pl.Date), outflows=pl.lit(10**10)).drop("decision_datetime"),
}
T2 = dict(T)
for t, extra in inj.items():
    T2[t] = pl.concat([T[t], extra.select(T[t].columns)], how="vertical_relaxed")
f2 = build_all(T2).sort("application_id")
f1 = features().sort("application_id")
num = [c for c, dt in f1.schema.items() if dt.is_numeric()]
diff = [c for c in num if not f1[c].cast(pl.Float64).fill_null(-7).fill_nan(-7).equals(f2[c].cast(pl.Float64).fill_null(-7).fill_nan(-7))]
print("features changed by injected future rows:", diff or "none -> cutoff filtering works")
assert not diff
