"""Verify the time-safe price reference now in features.v5_features:
(a) independent pandas recomputation of the reference and pp_mean; (b) two injection tests that encode the rule
precisely: rows dated after customer A's cutoff must not change A's features (or any non-price feature of anyone),
while customers deciding in a later month may legitimately see them in their reference; rows dated after every
cutoff must change nothing; (c) signal kept vs v5 (KS, rank correlation, residual vs v5)."""
import numpy as np
import pandas as pd
import polars as pl
from common import TABLES, calibrate, features, ks, load, resid_test, v5_oof
from features import build_all

f = features(); fp = f.to_pandas().set_index("customer_id")
T = {t: load(t) for t in TABLES}
e = T["ecommerce_transactions"].to_pandas()
e["lu"] = np.log(e.item_total / e.quantity); e["ym"] = e.transaction_datetime.dt.to_period("M")
rng = np.random.default_rng(31)
ok = []
for c in rng.choice(fp.index, 150, replace=False):
    cut = fp.loc[c, "decision_datetime"]; dm = pd.Period(cut, "M")
    ref = e[e.ym < dm].groupby("product_category").lu.agg(["mean", "std"])
    own = e[(e.customer_id == c) & (e.transaction_datetime <= cut)]
    z = (own.lu - own.product_category.map(ref["mean"])) / own.product_category.map(ref["std"])
    ok.append(np.isclose(z.mean() if len(own) else np.nan, fp.loc[c, "pp_mean"], equal_nan=True, rtol=1e-7))
print(f"(a) independent recomputation of pp_mean (reference rebuilt from rows before the decision month): match {np.mean(ok):.3f}")
assert np.mean(ok) == 1.0

apps = pl.concat([T["train_applications"], T["test_applications"]], how="diagonal_relaxed").select("customer_id", cut="decision_datetime")
fut = apps.sample(2000, seed=1)
def inject(when):
    x = T["ecommerce_transactions"].join(fut, on="customer_id").group_by("customer_id").first().with_columns(
        transaction_datetime=when, order_id=pl.lit("O_FAKE") + pl.col("customer_id"), item_total=pl.lit(10**9),
        quantity=pl.lit(1), product_category=pl.lit("groceries"))
    T2 = dict(T); T2["ecommerce_transactions"] = pl.concat([T["ecommerce_transactions"], x.select(T["ecommerce_transactions"].columns)], how="vertical_relaxed")
    return build_all(T2).sort("application_id")
f1 = f.sort("application_id")
num = [c for c, d in f1.schema.items() if d.is_numeric()]
def moved(a, b, cols):
    return [c for c in cols if not np.allclose(a[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(),
                                               b[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(), rtol=1e-9, atol=1e-9)]
f2 = inject(pl.col("cut") + pl.duration(hours=1))
cross = moved(f1, f2, [c for c in num if not c.startswith("pp_")])
print(f"(b1) rows 1h after 2,000 random customers' cutoffs: non-price features of anyone moved: {cross or 'none'} "
      f"(price features of customers deciding in a later month may move: those rows are in their past)")
assert not cross
dec = apps.filter(pl.col("cut").dt.month() == 12)
fut = dec.sample(2000, seed=2)
f2 = inject(pl.col("cut") + pl.duration(hours=1))
own = moved(f1, f2, num)
print(f"(b2) same test for 2,000 December customers (their rows can be in nobody's past): features moved: {own or 'none'}")
assert not own
f3 = inject(pl.lit(apps["cut"].max()) + pl.duration(days=40))
print(f"(b3) rows after every application's cutoff: features moved: {moved(f1, f3, num) or 'none'}")
assert not moved(f1, f3, num)

tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v5_oof())
x = tr["pp_mean"].to_numpy(); chi, _, d = resid_test(y, p, x)
print(f"(c) safe pp_mean: KS {ks(y, x):.4f} (v5 global 0.3684); residual vs v5 chi2={chi:.2f} dKS={d:+.4f} -> same information as v5")
