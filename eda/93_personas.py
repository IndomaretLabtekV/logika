"""Hidden personas: does the generator draw customers from a few behavioural types that a tree cannot reassemble?

Composition vectors per customer (category spend shares, payment-method order shares, hour-of-day shares, weekday
shares, cancel share, mean items/quantity, account flow shape) are clustered with k-means for several k. This is
unsupervised preprocessing on all 100k customers (no labels). For each k: bad rate per cluster, and the O/E residual
test of cluster id against the v8 score. A persona structure shows as multimodality (silhouette, cluster size
stability) and, if it matters for default, as chi2/df >> 1 beyond v8."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.cluster.vq import kmeans2
from common9 import OUT, crossfit_dks, features, load, oe_test, v8_ref

tr, y, p = v8_ref()
cut = pl.concat([load("train_applications").drop("target"), load("test_applications")]).select("customer_id", cutoff="decision_datetime")
ec = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
o = ec.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "payment_method", "order_status").first(),
                                n=pl.len(), q=pl.col("quantity").sum())
share = lambda df, col, vals, w=None: df.group_by("customer_id").agg(  # noqa: E731
    **{f"{col}_{v}": ((pl.col(col) == v).cast(pl.Float64) * (pl.col(w) if w else 1)).sum() / ((pl.col(w) if w else pl.lit(1.0)).sum()) for v in vals})
ecd = ec.filter(pl.col("order_status") == "completed")
parts = [
    share(ecd, "product_category", ecd["product_category"].unique().to_list(), "item_total"),
    share(o, "payment_method", o["payment_method"].unique().to_list()),
    share(o.with_columns(hb=(pl.col("transaction_datetime").dt.hour() // 3)), "hb", list(range(8))),
    share(o.with_columns(wd=pl.col("transaction_datetime").dt.weekday()), "wd", list(range(1, 8))),
    o.group_by("customer_id").agg(cancel=(pl.col("order_status") == "cancelled").mean(), items=pl.col("n").mean(), qty=pl.col("q").mean()),
]
C = cut.select("customer_id")
for part in parts:
    C = C.join(part, on="customer_id", how="left")
cols = [c for c in C.columns if c != "customer_id"]
X = C.select(pl.col(cols).fill_null(0)).to_numpy()
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
print(f"composition matrix {X.shape}; customers without orders: {C['cancel'].null_count()}")
_, sv, Vt = np.linalg.svd(X[np.random.default_rng(0).choice(len(X), 30000, replace=False)], full_matrices=False)
print("PCA explained variance ratio (a few dominant axes would indicate personas):", np.round((sv ** 2 / (sv ** 2).sum())[:10], 3))

ids = C["customer_id"]
pos = tr.select("customer_id").join(pl.DataFrame({"customer_id": ids, "row": np.arange(len(ids))}), on="customer_id", maintain_order="left")["row"].to_numpy()
rows = []
for k in (2, 3, 4, 6, 8, 12, 20):
    _, lab = kmeans2(X, k, seed=0, minit='++')
    b = lab[pos]
    chi, tab = oe_test(y, p, b)
    rates = [y[b == j].mean() for j in range(k)]
    sizes = np.bincount(lab, minlength=k) / len(lab)
    rows.append((k, chi, crossfit_dks(y, p, b), float(np.min(rates)), float(np.max(rates)), float(sizes.min()),
                 float(max(abs(t[5]) for t in tab))))
R = pl.DataFrame(rows, schema=["k", "chi2_df_vs_v8", "dKS_cf", "min_bad_rate", "max_bad_rate", "min_size", "max_abs_z"], orient="row")
with pl.Config(float_precision=4):
    print(R)
R.write_csv(OUT / "93_personas.csv")
Z = X @ Vt[:2].T
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sub = np.random.default_rng(0).choice(len(Z), 20000, replace=False)
ax[0].scatter(Z[sub, 0], Z[sub, 1], s=1, alpha=0.3); ax[0].set_title("composition PCA (all customers)")
ax[1].scatter(Z[pos, 0][y == 0][:15000], Z[pos, 1][y == 0][:15000], s=1, alpha=0.3, label="good")
ax[1].scatter(Z[pos, 0][y == 1][:5000], Z[pos, 1][y == 1][:5000], s=1, alpha=0.3, label="bad"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "93_personas.png", dpi=80)
print("saved", OUT / "93_personas.png")
