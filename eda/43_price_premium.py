"""Price premium: customers who pay more than the category's typical unit price default more. Is it real?
(1) residual test against the v3 blend; (2) time profile: trait or pre-application burst; (3) per category;
(4) price-grid structure of the generator; (5) estimator variants (mean/median/z/shrunk/windows); (6) segments;
(7) cutoff injection for the chosen estimator."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata
from common import OUT, ROOT, calibrate, features, ks, ks_nan, load, resid_test

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}
s3 = sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000
p = calibrate(y, s3)
rng = np.random.default_rng(5)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(20)])
print(f"noise vs v3: chi2/df p95={np.percentile(null[:, 0], 95):.2f}  dKS p95={np.percentile(null[:, 2], 95):+.4f}")

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime")
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") < pl.col("cutoff"))
e = e.with_columns(unit=pl.col("item_total") / pl.col("quantity"), lu=(pl.col("item_total") / pl.col("quantity")).log(),
                   age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   done=pl.col("order_status") == "completed")

# (4) price grid: how are unit prices generated?
print("\n(4) unit price granularity: share divisible by 1000:", (e["unit"] % 1000 == 0).mean().__round__(4),
      " distinct unit prices:", e["unit"].n_unique())
st = e.group_by("product_category").agg(mu=pl.col("lu").mean(), sd=pl.col("lu").std(), med=pl.col("unit").median(), n=pl.len()).sort("product_category")
print(st)
e = e.join(st.select("product_category", "mu", "sd"), on="product_category").with_columns(z=(pl.col("lu") - pl.col("mu")) / pl.col("sd"))
# does the premium carry across categories within a customer (a customer-level multiplier)?
cc = e.group_by("customer_id", "product_category").agg(z=pl.col("z").mean()).pivot(on="product_category", index="customer_id", values="z")
print("Pearson corr of a customer's mean z across categories (customer-level price multiplier?):")
print(cc.drop("customer_id").corr().select(pl.all().round(2)))

def T(col):
    return tr.select("customer_id").join(col, on="customer_id", how="left", maintain_order="left")

# (5) estimator variants
V = e.group_by("customer_id").agg(
    pz_mean=pl.col("z").mean(), pz_median=pl.col("z").median(),
    pz_trim=pl.col("z").filter((pl.col("z") > pl.col("z").quantile(0.1)) & (pl.col("z") < pl.col("z").quantile(0.9))).mean(),
    pz_done=pl.col("z").filter("done").mean(), pz_cancel=pl.col("z").filter(~pl.col("done")).mean(),
    pz_qw=(pl.col("z") * pl.col("quantity")).sum() / pl.col("quantity").sum(),
    pz_30=pl.col("z").filter(pl.col("age_d") <= 30).mean(), pz_90=pl.col("z").filter(pl.col("age_d") <= 90).mean(),
    pz_old=pl.col("z").filter(pl.col("age_d") > 182).mean(), pz_n=pl.len(), pz_sd=pl.col("z").std(),
)
V = V.with_columns(pz_shrunk=pl.col("pz_mean") * pl.col("pz_n") / (pl.col("pz_n") + 5), pz_shift=pl.col("pz_90") - pl.col("pz_old"))
Vt = T(V)
rows = []
for c in V.columns[1:]:
    x = Vt[c].cast(pl.Float64).to_numpy()
    rows.append((c, ks_nan(y, x), *resid_test(y, p, x)))
R = pl.DataFrame(rows, schema=["estimator", "ks", "chi2_df_vs_v3", "max_lift", "dKS_vs_v3"], orient="row").sort("dKS_vs_v3", descending=True)
with pl.Config(tbl_rows=20, float_precision=4):
    print("\n(1)+(5) estimators vs the v3 blend residual"); print(R)

# (2) time profile, (3) per category, (6) segments
wk = e.join(tr.select("customer_id", "target"), on="customer_id").with_columns(w=(pl.col("age_d") // 28).cast(pl.Int32)).filter(pl.col("w") < 18)
prof = wk.group_by("w", "target").agg(z=pl.col("z").mean()).pivot(on="target", index="w", values="z").sort("w")
print("\n(2) mean price z by 4-week block before decision (0=most recent):"); print(prof.with_columns(gap=pl.col("1") - pl.col("0")))
percat = []
for c in st["product_category"]:
    x = T(e.filter(pl.col("product_category") == c).group_by("customer_id").agg(v=pl.col("z").mean()))["v"].to_numpy()
    percat.append((c, ks_nan(y, x)))
print("(3) KS of the customer's mean price z within each category:", [(c, round(k, 3)) for c, k in percat])
x = Vt["pz_mean"].to_numpy(); hist = tr["has_history"].to_numpy()
print(f"(6) KS of pz_mean: thin-file {ks(y[hist == 0], x[hist == 0]):.4f}, history {ks(y[hist == 1], x[hist == 1]):.4f}; "
      f"corr with v3 score {np.corrcoef(rankdata(x), s3)[0, 1]:.3f}")
for lab, m in [("thin-file", hist == 0), ("history", hist == 1)]:
    pm = calibrate(y[m], s3[m], bins=30)
    print(f"    residual vs v3 inside {lab}: chi2/df={resid_test(y[m], pm, x[m])[0]:.2f} dKS={resid_test(y[m], pm, x[m])[2]:+.4f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for t in (0, 1):
    ax[0].hist(x[y == t], bins=60, alpha=.5, density=True, label=f"target={t}")
ax[0].set_title(f"customer mean price z (KS={ks(y, x):.3f})"); ax[0].legend()
ax[1].plot(prof["w"], prof["0"], "o-", label="good"); ax[1].plot(prof["w"], prof["1"], "o-", label="bad")
ax[1].set_xlabel("4-week blocks before decision"); ax[1].set_title("price z over time"); ax[1].legend()
dec = np.minimum((rankdata(x) / len(x) * 10).astype(int), 9)
ob = np.bincount(dec, weights=y) / np.bincount(dec); ex = np.bincount(dec, weights=p) / np.bincount(dec)
ax[2].plot(ob, "o-", label="observed"); ax[2].plot(ex, "s--", label="v3 expected"); ax[2].set_xlabel("pz_mean decile"); ax[2].legend()
ax[2].set_title("bad rate vs what v3 expects")
plt.tight_layout(); plt.savefig(OUT / "43_price_premium.png", dpi=80)
