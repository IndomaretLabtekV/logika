"""How far back does each behavioural signal live? Bad vs good activity by time before decision.
Drives the choice of aggregation windows in features.py."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from common import OUT, load

tr = load("train_applications").select("customer_id", "decision_datetime", "target")
DAY = 86400.0
fig, ax = plt.subplots(2, 3, figsize=(17, 8))

# e-commerce: orders per customer per week, by class
e = load("ecommerce_transactions").join(tr, on="customer_id")
o = e.group_by("order_id").agg(pl.col("customer_id", "decision_datetime", "transaction_datetime", "order_total",
                                       "order_status", "target", "payment_method").first())
o = o.with_columns(wk=((pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_seconds() / DAY / 7).floor().cast(pl.Int32))
ncls = tr.group_by("target").len()
g = o.filter(pl.col("wk") < 104).group_by("wk", "target").agg(n=pl.len(), spend=pl.col("order_total").sum(),
                                                               canc=(pl.col("order_status") == "cancelled").sum(),
                                                               cod=(pl.col("payment_method") == "COD").sum()).join(ncls, on="target")
g = g.with_columns(per=pl.col("n") / pl.col("len"), sp=pl.col("spend") / pl.col("len"), cr=pl.col("canc") / pl.col("n"),
                   codr=pl.col("cod") / pl.col("n")).sort("wk")
for k in (0, 1):
    q = g.filter(pl.col("target") == k)
    ax[0, 0].plot(q["wk"], q["per"], label=f"target={k}"); ax[0, 1].plot(q["wk"], q["cr"], label=f"target={k}")
    ax[0, 2].plot(q["wk"], q["codr"], label=f"target={k}")
r = g.pivot(on="target", index="wk", values="per").sort("wk")
print("ecom orders/customer/week ratio bad/good (first 20 weeks):\n", r.with_columns(ratio=pl.col("1") / pl.col("0")).head(20))
for a, t in zip(ax[0], ["orders / customer / week", "cancel rate by week", "COD share by week"]):
    a.set_title(t); a.set_xlabel("weeks before decision"); a.legend(); a.invert_xaxis()

# account: by months-ago
a = load("account_behavior").join(tr, on="customer_id").with_columns(
    mago=((pl.col("decision_datetime").dt.year() - pl.col("month").dt.year()) * 12
          + pl.col("decision_datetime").dt.month() - pl.col("month").dt.month()))
ga = a.group_by("mago", "target").agg(oi=pl.col("outflows").sum() / pl.col("inflows").sum(),
                                       low=pl.col("low_balance_days").mean(), inc=pl.col("income_like_credits").median()).sort("mago")
for k in (0, 1):
    q = ga.filter(pl.col("target") == k)
    ax[1, 0].plot(q["mago"], q["oi"], label=f"target={k}"); ax[1, 1].plot(q["mago"], q["low"], label=f"target={k}")
print("account by months ago:\n", ga.pivot(on="target", index="mago", values=["oi", "low", "inc"]).sort("mago").head(12))
ax[1, 0].set_title("outflow/inflow by month ago"); ax[1, 1].set_title("low_balance_days by month ago")

# repayment: share of installments with max dpd>0 by months ago of due date
r = load("repayment_history").join(tr, on="customer_id").with_columns(
    mago=((pl.col("decision_datetime") - pl.col("due_date")).dt.total_seconds() / DAY / 30.4).floor().cast(pl.Int32))
gr = r.filter(pl.col("mago") < 30).group_by("mago", "target").agg(d=(pl.col("max_days_past_due") > 0).mean()).sort("mago")
for k in (0, 1):
    q = gr.filter(pl.col("target") == k); ax[1, 2].plot(q["mago"], q["d"], label=f"target={k}")
print("repay late share by month ago:\n", gr.pivot(on="target", index="mago", values="d").sort("mago").head(14))
ax[1, 2].set_title("share of installments ever late, by due month ago")
for x in ax[1]:
    x.legend(); x.invert_xaxis(); x.set_xlabel("months before decision")
plt.tight_layout(); plt.savefig(OUT / "09_temporal.png", dpi=80)
