"""Follow-up of the DFS residual screen: item quantity, cheapest item price and decision month carry signal that v3
does not use. What is their structure (non-monotone? a specific value?), and is it real or a generator artefact?"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, PQ, ROOT, calibrate, features, ks, load

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}
p = calibrate(y, sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000)
D = tr.select("application_id").join(pl.read_parquet(PQ / "dfs.parquet"), on="application_id", how="left", maintain_order="left")

def table(name, key):
    d = pl.DataFrame({"k": key, "y": y, "p": p}).group_by("k").agg(n=pl.len(), bad=pl.col("y").mean(), expected=pl.col("p").mean())
    d = d.with_columns(o_over_e=pl.col("bad") / pl.col("expected"),
                       z=(pl.col("bad") - pl.col("expected")) * pl.col("n").sqrt() / (pl.col("expected") * (1 - pl.col("expected"))).sqrt())
    with pl.Config(tbl_rows=30, float_precision=4):
        print(f"\n=== {name}: bad rate, v3-expected bad rate, observed/expected, z"); print(d.sort("k"))
    return d

# raw quantity distribution
items = load("ecommerce_transactions")
print("item quantity distribution (all rows):", items["quantity"].value_counts().sort("quantity").rows())
qmax = D["dfs_item__quantity__max"].fill_null(-1).to_numpy()
table("max item quantity", qmax)
qmean = D["dfs_order__qty__mean"].to_numpy()
table("mean quantity per order (quintiles)", np.digitize(qmean, np.nanquantile(qmean, [.2, .4, .6, .8])))
imin = D["dfs_item__item_total__min"].to_numpy()
table("cheapest item price (deciles)", np.digitize(imin, np.nanquantile(imin, np.linspace(.1, .9, 9))))
month = tr["decision_datetime"].dt.month().to_numpy()
table("decision month", month)

# is quantity>1 per item linked to category / price? (generator structure)
it = items.with_columns(unit=pl.col("item_total") / pl.col("quantity").clip(lower_bound=1))
print("\nquantity by category (mean, share>1):")
print(it.group_by("product_category").agg(q_mean=pl.col("quantity").mean(), share_gt1=(pl.col("quantity") > 1).mean(),
                                         unit_price_median=pl.col("unit").median()).sort("product_category"))
print("zero-quantity rows:", (items["quantity"] == 0).sum())

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, (name, key) in zip(ax, [("max item quantity", qmax), ("cheapest item price decile", np.digitize(imin, np.nanquantile(imin, np.linspace(.1, .9, 9)))),
                               ("decision month", month)]):
    d = pl.DataFrame({"k": key, "y": y, "p": p}).group_by("k").agg(n=pl.len(), bad=pl.col("y").mean(), exp=pl.col("p").mean()).sort("k").filter(pl.col("n") >= 200)
    a.plot(d["k"], d["bad"], "o-", label="observed bad rate"); a.plot(d["k"], d["exp"], "s--", label="v3 expected")
    a.set_title(name); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "41_quantity_price_month.png", dpi=80)
