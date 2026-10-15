"""Is the rise of everyone's price z near the decision calendar inflation? Category x calendar-month reference prices
(label-free, train+test transactions) vs category-only, then the customer premium is re-estimated and tested."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, calibrate, features, ks_nan, load, resid_test

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}
p = calibrate(y, sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000)
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
e = (load("ecommerce_transactions").join(apps.select("customer_id", cutoff="decision_datetime"), on="customer_id")
     .filter(pl.col("transaction_datetime") < pl.col("cutoff"))
     .with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(), ym=pl.col("transaction_datetime").dt.truncate("1mo"),
                   age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   done=pl.col("order_status") == "completed"))
infl = e.group_by("ym").agg(lu=pl.col("lu").mean(), n=pl.len()).sort("ym")
print("mean log unit price by calendar month (all categories):")
print(infl.with_columns(pl.col("lu").round(4)).filter(pl.col("n") > 1000))
ref_c = e.group_by("product_category").agg(mc=pl.col("lu").mean(), sc=pl.col("lu").std())
ref_cm = e.group_by("product_category", "ym").agg(mcm=pl.col("lu").mean(), ncm=pl.len())
e = e.join(ref_c, on="product_category").join(ref_cm, on=["product_category", "ym"])
e = e.with_columns(z_c=(pl.col("lu") - pl.col("mc")) / pl.col("sc"), z_cm=(pl.col("lu") - pl.col("mcm")) / pl.col("sc"))
prof = (e.join(tr.select("customer_id", "target"), on="customer_id").with_columns(w=(pl.col("age_d") // 28).cast(pl.Int32))
        .filter(pl.col("w") < 18).group_by("w", "target").agg(z_c=pl.col("z_c").mean(), z_cm=pl.col("z_cm").mean()).sort("w", "target"))
print("\nprice z by 4-week block before decision, category-only vs category x month reference:")
print(prof.pivot(on="target", index="w", values=["z_c", "z_cm"]).sort("w").select(pl.all().round(3) if False else pl.all()))
V = e.group_by("customer_id").agg(
    pc_mean=pl.col("z_c").mean(), pcm_mean=pl.col("z_cm").mean(), pcm_done=pl.col("z_cm").filter("done").mean(),
    pcm_trim=pl.col("z_cm").filter((pl.col("z_cm") > pl.col("z_cm").quantile(0.1)) & (pl.col("z_cm") < pl.col("z_cm").quantile(0.9))).mean(),
    pcm_90=pl.col("z_cm").filter(pl.col("age_d") <= 90).mean(), pcm_old=pl.col("z_cm").filter(pl.col("age_d") > 182).mean())
Vt = tr.select("customer_id").join(V, on="customer_id", how="left", maintain_order="left")
print("\nestimators (KS, chi2/df vs v3, dKS vs v3):")
for c in V.columns[1:]:
    x = Vt[c].to_numpy(); chi, _, d = resid_test(y, p, x)
    print(f"  {c:<10} KS={ks_nan(y, x):.4f} chi2={chi:6.2f} dKS={d:+.4f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(infl["ym"], infl["lu"], "o-"); ax[0].set_title("mean log unit price by calendar month"); ax[0].tick_params(axis="x", rotation=45)
for col, ls in [("z_c", "-"), ("z_cm", "--")]:
    for t, colr in [(0, "C0"), (1, "C1")]:
        s = prof.filter(pl.col("target") == t)
        ax[1].plot(s["w"], s[col], ls, color=colr, label=f"{col} target={t}")
ax[1].set_xlabel("4-week blocks before decision"); ax[1].legend(fontsize=7); ax[1].set_title("price z, category vs category x month reference")
plt.tight_layout(); plt.savefig(OUT / "44_price_deflation.png", dpi=80)
