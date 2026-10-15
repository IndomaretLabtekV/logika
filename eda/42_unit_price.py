"""Basket size vs unit price: v3 over-reads quantity as risk, and cheapest-item price is under-used. Decompose spend
into frequency x basket quantity x unit price (relative to the category median, computed without labels) and test
each part against the v3 residual. Also: decision month."""
import numpy as np
import polars as pl
from common import PQ, features, load, screen

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime", month=pl.col("decision_datetime").dt.month())
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") < pl.col("cutoff"))
e = e.with_columns(unit=pl.col("item_total") / pl.col("quantity"),
                   age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   done=pl.col("order_status") == "completed")
# category reference price: median unit price over all rows (train + test, no labels)
ref = e.group_by("product_category").agg(ref=pl.col("unit").median())
e = e.join(ref, on="product_category").with_columns(prem=(pl.col("unit") / pl.col("ref")).log())
r90 = pl.col("age_d") <= 90
X = e.group_by("customer_id").agg(
    up_qty_per_item=pl.col("quantity").mean(),
    up_qty_per_item_90=pl.col("quantity").filter(r90).mean(),
    up_items_per_order=pl.len() / pl.col("order_id").n_unique(),
    up_unit_mean=pl.col("unit").mean(), up_unit_median=pl.col("unit").median(), up_unit_min=pl.col("unit").min(),
    up_prem_mean=pl.col("prem").mean(), up_prem_median=pl.col("prem").median(), up_prem_max=pl.col("prem").max(),
    up_prem_mean_90=pl.col("prem").filter(r90).mean(),
    up_prem_done_mean=pl.col("prem").filter("done").mean(),
    up_item_min=pl.col("item_total").min(),
    up_n_items=pl.len(),
)
X = X.with_columns(up_prem_shift=pl.col("up_prem_mean_90") - pl.col("up_prem_mean"))
X = X.join(cut.select("customer_id", up_month="month"), on="customer_id", how="right")
r = screen(X, "42_unit_price", show=20)
# what is cheapest-item price really measuring?
f = features().filter(pl.col("is_train") == 1).select("customer_id", "target")
m = f.join(X, on="customer_id")
print("\nSpearman of cheapest item price with:", {c: round(m.select(pl.corr("up_item_min", c, method="spearman")).item(), 3)
                                              for c in ["up_n_items", "up_unit_median", "up_prem_mean", "up_qty_per_item"]})
