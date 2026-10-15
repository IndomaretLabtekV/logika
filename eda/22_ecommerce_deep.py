"""E-commerce burst anatomy: weekly bad/good profile for counts/spend/cancels/categories/payment/hour, whether the
pre-application burst is multiplicative on the customer's own base rate, and smooth (exp-decay, likelihood-ratio)
burst scores instead of box windows. Screened vs v1."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, load, screen

tr = load("train_applications")
apps = pl.concat([tr.drop("target"), load("test_applications")])
cut = apps.select("customer_id", "application_datetime", cutoff="decision_datetime")
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
o = (e.group_by("order_id").agg(
        pl.col("customer_id", "transaction_datetime", "order_total", "payment_method", "order_status",
               "cutoff", "application_datetime").first(),
        n_items=pl.len(), qty=pl.col("quantity").sum(),
        cats=pl.col("product_category").unique())
     .with_columns(age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                   done=pl.col("order_status") == "completed",
                   hour=pl.col("transaction_datetime").dt.hour(),
                   after_app=pl.col("transaction_datetime") > pl.col("application_datetime"))
     .with_columns(wk=(pl.col("age_d") // 7).cast(pl.Int32)))
y_map = tr.select("customer_id", "target")
ot = o.join(y_map, on="customer_id")
nb, ng = tr["target"].sum(), tr.height - tr["target"].sum()
print(f"orders (dedup) {o.height:,}; history span (d) median={o.group_by('customer_id').agg(pl.col('age_d').max())['age_d'].median():.0f}")

# 1) weekly profile, bad/good ratio per customer for several order-level measures
def prof(expr, name):
    g = ot.filter(pl.col("wk") < 52).group_by("wk", "target").agg(v=expr.sum())
    g = g.pivot(on="target", index="wk", values="v").sort("wk")
    return g.with_columns(r=(pl.col("1") / nb) / (pl.col("0") / ng)).select("wk", pl.col("r").alias(name))
P = prof(pl.col("qty").is_not_null().cast(pl.Int32), "orders")
for expr, name in [(pl.col("order_total") * pl.col("done"), "spend"), ((~pl.col("done")).cast(pl.Int32), "cancels"),
                   (pl.col("qty"), "qty"), ((pl.col("hour") < 5).cast(pl.Int32), "night_orders"),
                   ((pl.col("payment_method") == "COD").cast(pl.Int32), "cod"),
                   ((pl.col("payment_method") == "credit_card").cast(pl.Int32), "credit_card"),
                   ((pl.col("payment_method") == "e_wallet").cast(pl.Int32), "e_wallet")]:
    P = P.join(prof(expr, name), on="wk")
with pl.Config(tbl_rows=30, float_precision=3):
    print("\n=== bad/good per-customer ratio by week before decision")
    print(P.head(26))

# category-level burst: share of items per category in last 4 weeks vs >=17 weeks, bad vs good
et = e.join(y_map, on="customer_id").with_columns(
    rec=((pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days() < 28))
cs = (et.group_by("target", "rec", "product_category").agg(n=pl.len())
      .with_columns(sh=pl.col("n") / pl.col("n").sum().over("target", "rec"))
      .pivot(on=["target", "rec"], index="product_category", values="sh"))
print("\n=== category share of items (target, recent<28d)"); print(cs)

# 2) is the burst multiplicative on the customer's base rate?
cust = o.group_by("customer_id").agg(
    base=((pl.col("age_d") > 120) & (pl.col("age_d") <= 365)).sum() / 35.0,
    rec=(pl.col("age_d") <= 28).sum() / 4.0).join(y_map, on="customer_id")
cust = cust.with_columns(bq=pl.col("base").rank("ordinal").truediv(pl.len()).mul(4).floor().clip(upper_bound=3))
print("\n=== recent weekly rate vs base weekly rate by base quartile")
print(cust.group_by("bq", "target").agg(base=pl.col("base").mean(), rec=pl.col("rec").mean())
      .with_columns(mult=pl.col("rec") / pl.col("base"), add=pl.col("rec") - pl.col("base")).sort("bq", "target"))

# 3) orders between application and decision (inside the cutoff, after the customer applied)
aa = ot.group_by("customer_id", "target").agg(n_after=pl.col("after_app").sum())
print("\n=== orders placed after application_datetime (still before decision):")
print(aa.group_by("target").agg(mean=pl.col("n_after").mean(), any=(pl.col("n_after") > 0).mean()))

# 4) candidates: exp-decay burst relative to own base, LR with a fitted profile, recent-vs-base shares
r_wk = P.filter(pl.col("wk") < 26)["orders"].to_numpy()
r_wk = np.maximum(r_wk, 1.0)
lr_w = pl.DataFrame({"wk": np.arange(len(r_wk), dtype=np.int32), "lw": np.log(r_wk), "excess": r_wk - 1})
excess_total = float((r_wk - 1).sum())   # weeks of extra intensity a bad customer shows at unit base
o2 = o.join(lr_w, on="wk", how="left").with_columns(pl.col("lw").fill_null(0.0))
base = (((pl.col("age_d") > 182) & (pl.col("age_d") <= 730)).sum()
        / ((pl.col("age_d").max().clip(upper_bound=730) - 182).clip(lower_bound=14) / 7))
aggs = dict(
    ec2_base=base,
    ec2_lr_raw=pl.col("lw").sum(),
    ec2_spend_base=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") > 182)).sum()
    / ((pl.col("age_d").max().clip(upper_bound=730) - 182).clip(lower_bound=14) / 7),
    ec2_spend_28=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") <= 28)).sum() / 4,
    ec2_cancel_28=(~pl.col("done")).filter(pl.col("age_d") <= 28).sum(),
    ec2_cancel_rate_90=(~pl.col("done")).filter(pl.col("age_d") <= 90).mean(),
    ec2_night_share_90=(pl.col("hour") < 5).filter(pl.col("age_d") <= 90).mean(),
    ec2_cod_share_28=(pl.col("payment_method") == "COD").filter(pl.col("age_d") <= 28).mean(),
    ec2_cod_share_old=(pl.col("payment_method") == "COD").filter(pl.col("age_d") > 182).mean(),
    ec2_cc_share_28=(pl.col("payment_method") == "credit_card").filter(pl.col("age_d") <= 28).mean(),
    ec2_items_per_order_28=pl.col("n_items").filter(pl.col("age_d") <= 28).mean(),
    ec2_aov_28=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") <= 28)).mean(),
    ec2_aov_old=pl.col("order_total").filter(pl.col("done") & (pl.col("age_d") > 182)).mean(),
    ec2_gap_min_28=pl.col("age_d").filter(pl.col("age_d") <= 28).sort().diff().min(),
    ec2_n_after_app=pl.col("after_app").sum(),
    ec2_hours_since_last=pl.col("age_d").min() * 24,
)
for tau in (3, 7, 14, 28, 56):
    aggs[f"ec2_ewm{tau}"] = (-pl.col("age_d") / tau).exp().sum()
c = o2.group_by("customer_id").agg(**aggs)
c = c.with_columns(
    ec2_lr=pl.col("ec2_lr_raw") - pl.col("ec2_base") * excess_total,
    ec2_spend_accel=pl.col("ec2_spend_28") / (pl.col("ec2_spend_base") + 1e4),
    ec2_cod_shift=pl.col("ec2_cod_share_28") - pl.col("ec2_cod_share_old"),
    ec2_aov_shift=pl.col("ec2_aov_28") / pl.col("ec2_aov_old"),
    **{f"ec2_ewm{t}_rel": pl.col(f"ec2_ewm{t}") / (pl.col("ec2_base") * t / 7 + 0.05) for t in (3, 7, 14, 28, 56)},
    **{f"ec2_ewm{t}_exc": pl.col(f"ec2_ewm{t}") - pl.col("ec2_base") * t / 7 for t in (3, 7, 14, 28, 56)},
)
r = screen(c, "22_ecom", show=45)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for col in ["orders", "spend", "cancels", "qty", "cod", "night_orders"]:
    ax[0].plot(P["wk"].head(26), P[col].head(26), label=col)
ax[0].axhline(1, color="k", lw=1); ax[0].set_xlabel("weeks before decision"); ax[0].set_title("bad/good per-customer ratio"); ax[0].legend(fontsize=7)
g = cust.group_by("bq", "target").agg(rec=pl.col("rec").mean(), base=pl.col("base").mean()).sort("bq")
for t in (0, 1):
    gg = g.filter(pl.col("target") == t)
    ax[1].plot(gg["base"], gg["rec"], "o-", label=f"target={t}")
ax[1].plot([0, g["base"].max()], [0, g["base"].max()], "k:"); ax[1].set_xlabel("base weekly orders"); ax[1].set_ylabel("last-4-week weekly orders")
ax[1].set_title("burst vs own base rate"); ax[1].legend()
ax[2].barh(r["feature"].head(15).to_list()[::-1], r["dKS_cf"].head(15).to_list()[::-1]); ax[2].set_title("cross-fit dKS vs v1")
plt.tight_layout(); plt.savefig(OUT / "22_ecom.png", dpi=80)
