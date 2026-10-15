"""Verify the v5 features (features.v5_features): independent pandas recomputation, future-row injection over all
dated tables, redundancy inside the family, residual test against the v3 blend, decile plots."""
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from common import ROOT, TABLES, analyze, calibrate, features, ks_nan, load, resid_test
from features import build_all

f = features()
V5 = [c for c in f.columns if c.startswith("pp_") or c == "dec_month"]
print(f"{len(V5)} v5 features; matrix {f.shape}")
fp = f.to_pandas().set_index("customer_id")
e_all = load("ecommerce_transactions").join(f.select("customer_id", cutoff="decision_datetime"), on="customer_id") \
    .filter(pl.col("transaction_datetime") <= pl.col("cutoff")).to_pandas()
e_all["lu"] = np.log(e_all.item_total / e_all.quantity)
ref = e_all.groupby("product_category").lu.agg(["mean", "std"])          # independent reference, pandas std (ddof=1)
cust = np.random.default_rng(21).choice(fp.index, 300, replace=False)
sub = e_all[e_all.customer_id.isin(cust)].copy()
sub["z"] = (sub.lu - sub.product_category.map(ref["mean"])) / sub.product_category.map(ref["std"])
sub["age_d"] = (sub.cutoff - sub.transaction_datetime).dt.total_seconds() / 86400
ok = {}
for c in cust:
    s = sub[sub.customer_id == c]
    for name, exp in [("pp_mean", s.z.mean() if len(s) else np.nan),
                      ("pp_done", s.z[s.order_status == "completed"].mean() if (s.order_status == "completed").any() else np.nan),
                      ("pp_cat_groceries", s.z[s.product_category == "groceries"].mean() if (s.product_category == "groceries").any() else np.nan),
                      ("pp_qty_item_90", s.quantity[s.age_d <= 90].mean() if (s.age_d <= 90).any() else np.nan),
                      ("dec_month", fp.loc[c, "decision_datetime"].month)]:
        ok.setdefault(name, []).append(bool(np.isclose(exp, fp.loc[c, name], equal_nan=True, rtol=1e-9)))
print("(a) independent recomputation on 300 customers:")
for k, v in ok.items():
    print(f"    {k:<18} match {np.mean(v):.3f}")
    assert np.mean(v) == 1.0, k

T = {t: load(t) for t in TABLES}
apps = pl.concat([T["train_applications"], T["test_applications"]], how="diagonal_relaxed").select("customer_id", cut="decision_datetime")
fut = apps.sample(2000, seed=1)
first = lambda t: T[t].join(fut, on="customer_id").group_by("customer_id").first()  # noqa: E731
inj = {"ecommerce_transactions": first("ecommerce_transactions").with_columns(
           transaction_datetime=pl.col("cut") + pl.duration(hours=1), order_id=pl.lit("O_FAKE") + pl.col("customer_id"),
           item_total=pl.lit(10**9), quantity=pl.lit(1), product_category=pl.lit("groceries")),
       "repayment_history": first("repayment_history").with_columns(due_date=pl.col("cut") + pl.duration(days=5), max_days_past_due=pl.lit(999))}
T2 = dict(T)
for t, extra in inj.items():
    T2[t] = pl.concat([T[t], extra.select(T[t].columns)], how="vertical_relaxed")
f2 = build_all(T2).sort("application_id"); f1 = f.sort("application_id")
num = [c for c, d in f1.schema.items() if d.is_numeric()]
moved = [c for c in num if not np.allclose(f1[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(),
                                           f2[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(), rtol=1e-9, atol=1e-9)]
print(f"(b) features moved by future e-commerce/repayment rows (incl. a 1e9 groceries item that would shift the category reference): {moved or 'none'} of {len(num)}")
assert not moved

tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}
p = calibrate(y, sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000)
C = tr.select(V5).with_columns(pl.all().cast(pl.Float64).rank()).to_pandas().corr()
print("(c) pairs inside v5 with |rho|>0.9:", [(a, b, round(C.loc[a, b], 3)) for i, a in enumerate(V5) for b in V5[i + 1:] if abs(C.loc[a, b]) > 0.9] or "none")
print("(d) residual test vs v3 blend:")
for c in V5:
    x = tr[c].cast(pl.Float64).to_numpy(); chi, _, d = resid_test(y, p, x)
    print(f"    {c:<18} KS={ks_nan(y, x):.4f} chi2={chi:6.2f} dKS={d:+.4f}")
analyze(f, V5, "46_v5", top_plot=12)
