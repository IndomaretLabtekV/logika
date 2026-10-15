"""Fix the one cutoff weakness in the v5 price premium: the category reference (mean/std of log unit price) was
computed from every transaction in the data, including other customers' later purchases. Alternative used here:
an expanding reference per category built only from transactions of any customer (labels never used) dated before
the first day of the application's decision month. Compared with the v5 global reference and with the per-fold
reference Astra built (eda/v5_review/01)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, load, resid_test, v5_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
print(f"v5 blend reproduced: OOF KS {ks(y, v5_oof()):.4f} (notebook 0.5274)")
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime", dm=pl.col("decision_datetime").dt.truncate("1mo"))
raw = load("ecommerce_transactions").with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(),
                                                   ym=pl.col("transaction_datetime").dt.truncate("1mo"))
# monthly sufficient statistics per category over ALL rows (each row is a past fact once its month has ended)
ms = raw.group_by("product_category", "ym").agg(n=pl.len(), s=pl.col("lu").sum(), ss=(pl.col("lu") ** 2).sum()).sort("ym")
months = apps.select(dm=pl.col("decision_datetime").dt.truncate("1mo")).unique()
ref = (months.join(ms, how="cross").filter(pl.col("ym") < pl.col("dm"))
       .group_by("product_category", "dm").agg(n=pl.col("n").sum(), s=pl.col("s").sum(), ss=pl.col("ss").sum())
       .with_columns(mu=pl.col("s") / pl.col("n"), sd=((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt()))
print("reference size per decision month: min rows per category", ref["n"].min(), "| first months:")
print(ref.filter(pl.col("product_category") == "groceries").sort("dm").select("dm", "n", pl.col("mu").round(4), pl.col("sd").round(4)).head(3))
e = raw.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
glob = raw.group_by("product_category").agg(gmu=pl.col("lu").mean(), gsd=pl.col("lu").std())
e = e.join(ref.select("product_category", "dm", "mu", "sd"), on=["product_category", "dm"]).join(glob, on="product_category")
e = e.with_columns(z_safe=(pl.col("lu") - pl.col("mu")) / pl.col("sd"), z_glob=(pl.col("lu") - pl.col("gmu")) / pl.col("gsd"))
c = e.group_by("customer_id").agg(safe=pl.col("z_safe").mean(), glob=pl.col("z_glob").mean())
t = tr.select("customer_id", "pp_mean").join(c, on="customer_id", how="left", maintain_order="left")
p = calibrate(y, v5_oof())
print(f"\nv5 pp_mean equals the global reference here: max |diff| {np.nanmax(np.abs(t['pp_mean'] - t['glob'])):.2e}")
for col in ["glob", "safe"]:
    x = t[col].to_numpy()
    chi, _, d = resid_test(y, p, x)
    print(f"  {col:<5} KS={ks(y, x):.4f}  Spearman vs global={t.select(pl.corr(col, 'glob', method='spearman')).item():.5f}  residual vs v5: chi2={chi:.2f} dKS={d:+.4f}")
astra = pl.read_parquet("v5_review/out/01_candidates_oof.parquet")
col = [k for k in astra.columns if "mean" in k and "price" in k][:1]
if col:
    m = tr.select("application_id").join(astra.select("application_id", col[0]), on="application_id", how="left").with_columns(safe=t["safe"])
    print(f"  Spearman safe (this script) vs Astra per-fold '{col[0]}': {m.select(pl.corr('safe', col[0], method='spearman')).item():.5f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
g = ref.filter(pl.col("product_category") == "groceries").sort("dm")
ax[0].plot(g["dm"], g["mu"], "o-", label="expanding (safe)"); ax[0].axhline(glob.filter(pl.col("product_category") == "groceries")["gmu"][0], color="r", ls="--", label="global (v5)")
ax[0].set_title("groceries reference mean log price by decision month"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=45)
s = np.random.default_rng(0).choice(len(t), 5000, replace=False)
ax[1].scatter(t["glob"].to_numpy()[s], t["safe"].to_numpy()[s], s=3, c=y[s], cmap="coolwarm"); ax[1].set_xlabel("v5 global premium"); ax[1].set_ylabel("safe premium")
ax[1].set_title("customer premium: global vs time-safe reference")
plt.tight_layout(); plt.savefig(OUT / "50_price_reference.png", dpi=80)
