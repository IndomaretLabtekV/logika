"""Thin-file customers (30%, no previous apps / repayment / account): what information exists, what v2 leaves in
the residual, and new e-commerce / bureau / application constructs scored inside the segment only."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, ks_nan, load, resid_test, v2_oof

f = features()
tr = f.filter(pl.col("is_train") == 1)
y_all = tr["target"].to_numpy()
s_all = v2_oof()
thin = tr["has_history"].to_numpy() == 0
y, s = y_all[thin], s_all[thin]
p = calibrate(y, s, bins=30)
print(f"thin-file n={thin.sum()} bad={y.mean():.3f}  v2 GBDT KS inside segment={ks(y, s):.4f}  (history: {ks(y_all[~thin], s_all[~thin]):.4f})")
T = tr.filter(pl.col("has_history") == 0)
ID = {"application_id", "customer_id", "decision_datetime", "target", "is_train"}
num = [c for c in f.columns if c not in ID and f[c].dtype != pl.String]
avail = [c for c in num if T[c].null_count() < 0.5 * T.height and T[c].cast(pl.Float64).n_unique() > 1]
print(f"{len(avail)} of {len(num)} features are populated for thin-file customers")
rows = [(c, ks_nan(y, T[c].cast(pl.Float64).to_numpy()), *resid_test(y, p, T[c].cast(pl.Float64).to_numpy())) for c in avail]
r = pl.DataFrame(rows, schema=["feature", "ks_in_thin", "chi2_df", "max_lift", "dKS_cf"], orient="row").sort("chi2_df", descending=True)
with pl.Config(tbl_rows=15, float_precision=4):
    print("\n=== thin-file: residual left by v2 (top 15)"); print(r.head(15))
    print("\n=== thin-file: strongest univariate features"); print(r.sort("ks_in_thin", descending=True).head(15))

# new constructs inside the segment: category / payment bursts, spend volatility, first order age, weekday mix
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime")
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.with_columns(age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   done=pl.col("order_status") == "completed")
rec, old = pl.col("age_d") <= 56, pl.col("age_d") > 182
cats = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
pms = ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]
o = e.group_by("order_id").agg(pl.col("customer_id", "age_d", "order_total", "payment_method", "done", "transaction_datetime").first(),
                               n_items=pl.len(), qty=pl.col("quantity").sum())
X = o.group_by("customer_id").agg(
    tf_wk_cv=pl.col("age_d").filter(pl.col("age_d") <= 182).floordiv(7).value_counts().struct.field("count").std()
    / pl.col("age_d").filter(pl.col("age_d") <= 182).floordiv(7).value_counts().struct.field("count").mean(),
    tf_qty_28=pl.col("qty").filter(pl.col("age_d") <= 28).sum(),
    tf_weekend_share_56=(pl.col("transaction_datetime").dt.weekday() >= 6).filter(rec).mean(),
    tf_spend_std_56=pl.col("order_total").filter(rec & pl.col("done")).std(),
    tf_max_order_56=pl.col("order_total").filter(rec & pl.col("done")).max(),
    tf_orders_3d=(pl.col("age_d") <= 3).sum(),
    tf_orders_1d=(pl.col("age_d") <= 1).sum(),
    **{f"tf_pm_{m}_burst": (pl.col("payment_method") == m).filter(rec).sum() / 8
       / ((pl.col("payment_method") == m).filter(old).sum() / ((pl.col("age_d").max().clip(upper_bound=730) - 182).clip(lower_bound=14) / 7) + 0.05)
       for m in pms},
)
ci = e.filter("done").group_by("customer_id").agg(
    **{f"tf_cat_{c}_56": pl.col("item_total").filter((pl.col("product_category") == c) & (pl.col("age_d") <= 56)).sum() for c in cats},
    **{f"tf_cat_{c}_share_shift": (pl.col("item_total").filter((pl.col("product_category") == c) & (pl.col("age_d") <= 56)).sum()
                                   / (pl.col("item_total").filter(pl.col("age_d") <= 56).sum() + 1))
       - (pl.col("item_total").filter((pl.col("product_category") == c) & (pl.col("age_d") > 182)).sum()
          / (pl.col("item_total").filter(pl.col("age_d") > 182).sum() + 1)) for c in cats},
)
X = X.join(ci, on="customer_id", how="left")
Xt = T.select("customer_id").join(X, on="customer_id", how="left", maintain_order="left")
rows = []
for c in X.columns[1:]:
    x = Xt[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    rows.append((c, ks_nan(y, x), *resid_test(y, p, x)))
r2 = pl.DataFrame(rows, schema=["feature", "ks_in_thin", "chi2_df", "max_lift", "dKS_cf"], orient="row").sort("chi2_df", descending=True)
rng = np.random.default_rng(1)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(20)])
print(f"\nnoise (thin-file n): chi2/df p95={np.percentile(null[:, 0], 95):.2f}  dKS p95={np.percentile(null[:, 2], 95):+.4f}")
with pl.Config(tbl_rows=20, float_precision=4):
    print("=== new thin-file constructs vs v2 residual"); print(r2.head(20))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].barh(r["feature"].head(12).to_list()[::-1], r["chi2_df"].head(12).to_list()[::-1]); ax[0].axvline(np.percentile(null[:, 0], 95), color="r", ls="--")
ax[0].set_title("thin-file: v2 residual by existing feature")
ax[1].barh(r2["feature"].head(12).to_list()[::-1], r2["chi2_df"].head(12).to_list()[::-1]); ax[1].axvline(np.percentile(null[:, 0], 95), color="r", ls="--")
ax[1].set_title("thin-file: new constructs")
plt.tight_layout(); plt.savefig(OUT / "31_thin_file.png", dpi=80)
