"""Audit: are the auxiliary tables truncated at decision_datetime for TEST customers as for train? Counts rows dated
after each customer's cutoff per table, split train/test (information only; features.py filters on the cutoff
anyway and any post-cutoff row would be forbidden leakage under rule 3). Also compares how close to the cutoff the
latest available record is (train vs test), which would reveal a different extraction window.
Usage: cd eda && ../.venv/bin/python 139_cutoff_audit.py"""
import polars as pl

from common import load

apps = pl.concat([load("train_applications").with_columns(split=pl.lit("train")),
                  load("test_applications").with_columns(split=pl.lit("test"))], how="diagonal_relaxed")
cut = apps.select("customer_id", "split", cutoff="decision_datetime")
checks = {
    "previous_applications": "decision_datetime",
    "repayment_history": "due_date",
    "bureau_observations": "available_datetime",
    "ecommerce_transactions": "transaction_datetime",
}
for t, col in checks.items():
    d = load(t).join(cut, on="customer_id")
    after = d.filter(pl.col(col) > pl.col("cutoff"))
    gap = d.group_by("customer_id", "split").agg(last=(pl.col("cutoff").first() - pl.col(col).max()).dt.total_seconds() / 86400)
    print(f"{t:24s} rows after cutoff: " + ", ".join(f"{s} {after.filter(pl.col('split') == s).height}" for s in ("train", "test"))
          + " | days from latest record to cutoff (median): "
          + ", ".join(f"{s} {gap.filter(pl.col('split') == s)['last'].median():.2f}" for s in ("train", "test")))
rp = load("repayment_history").join(cut, on="customer_id")
late = rp.filter(pl.col("last_payment_datetime") > pl.col("cutoff"))
print("repayment last_payment_datetime after cutoff:", ", ".join(f"{s} {late.filter(pl.col('split') == s).height}" for s in ("train", "test")))
acc = load("account_behavior").join(cut, on="customer_id")
lm = acc.filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) > pl.col("cutoff"))
print("account months not complete before cutoff:", ", ".join(f"{s} {lm.filter(pl.col('split') == s).height}" for s in ("train", "test")))
for s in ("train", "test"):
    sub = rp.filter(pl.col("split") == s)
    print(f"repayment {s}: rows/customer {sub.height / sub['customer_id'].n_unique():.2f}, share unpaid rows "
          f"{(sub['amount_paid'] < sub['amount_due']).mean():.4f}, mean dpd {sub['days_past_due'].mean():.3f}")
