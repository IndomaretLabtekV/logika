"""Lateness has two dimensions: how often (frequency) and how late when late (severity). v1 mixes them (late_mean
includes zeros). Distribution of days-late among late installments by target, and severity-only features."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, load, screen

apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
y = load("train_applications").select("customer_id", "target")
r = load("repayment_history").join(apps.select("customer_id", cutoff="decision_datetime"), on="customer_id") \
    .filter(pl.col("due_date") <= pl.col("cutoff"))
r = r.with_columns(paid=pl.col("amount_paid") >= pl.col("amount_due"),
                   late_d=(pl.col("last_payment_datetime") - pl.col("due_date")).dt.total_days())
lt = r.filter(pl.col("paid") & (pl.col("late_d") > 0)).join(y, on="customer_id")
print("days late among paid-late installments, by target (quantiles):")
print(lt.group_by("target").agg([pl.col("late_d").quantile(q).alias(f"q{int(q*100)}") for q in (.1, .25, .5, .75, .9, .99)] + [pl.len()]).sort("target"))
print("max_dpd among late installments (incl. unpaid) by target:")
print(r.filter(pl.col("max_days_past_due") > 0).join(y, on="customer_id").group_by("target")
      .agg([pl.col("max_days_past_due").quantile(q).alias(f"q{int(q*100)}") for q in (.1, .5, .9)] + [pl.col("max_days_past_due").mean().alias("mean")]).sort("target"))
lateall = r.filter(pl.col("max_days_past_due") > 0)
S = r.group_by("customer_id").agg(
    sv_late_d_mean=pl.col("late_d").filter(pl.col("paid") & (pl.col("late_d") > 0)).mean(),
    sv_late_d_median=pl.col("late_d").filter(pl.col("paid") & (pl.col("late_d") > 0)).median(),
    sv_maxdpd_mean_late=pl.col("max_days_past_due").filter(pl.col("max_days_past_due") > 0).mean(),
    sv_maxdpd_log_late=(pl.col("max_days_past_due").filter(pl.col("max_days_past_due") > 0)).log().mean(),
    sv_share30_of_late=((pl.col("max_days_past_due") >= 30).sum() / (pl.col("max_days_past_due") > 0).sum()),
    sv_unpaid_of_late=((~pl.col("paid")).sum() / (pl.col("max_days_past_due") > 0).sum()),
)
screen(S, "26_severity")
fig, ax = plt.subplots(figsize=(7, 4))
for t in (0, 1):
    v = lt.filter(pl.col("target") == t)["late_d"].to_numpy()
    ax.hist(np.clip(v, 0, 120), bins=60, density=True, alpha=.5, label=f"target={t}")
ax.set_xlabel("days late (paid-late installments)"); ax.legend(); plt.tight_layout(); plt.savefig(OUT / "26_severity.png", dpi=80)
