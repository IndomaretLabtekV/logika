"""Target rate, temporal target drift, per-table coverage of train vs test, and post-cutoff (leakage) row counts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from common import OUT, load

tr, te = load("train_applications"), load("test_applications")
apps = pl.concat([tr.with_columns(split=pl.lit("train")), te.with_columns(target=pl.lit(None, pl.Int64), split=pl.lit("test"))], how="diagonal")
print("target rate:", tr["target"].mean(), "n_pos:", tr["target"].sum())
print("customers overlap train/test:", len(set(tr["customer_id"]) & set(te["customer_id"])))

# temporal
m = apps.with_columns(ym=pl.col("decision_datetime").dt.strftime("%Y-%m"))
tm = m.group_by("ym", "split").agg(n=pl.len(), rate=pl.col("target").mean()).sort("ym")
print(tm.pivot(on="split", index="ym", values=["n", "rate"]))
lag = (pl.col("decision_datetime") - pl.col("application_datetime")).dt.total_hours()
print("decision lag hours by split:\n", m.group_by("split").agg(lag.describe() if False else [lag.mean().alias("mean"), lag.median().alias("med"), lag.max().alias("max")]))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
p = tm.filter(pl.col("split") == "train")
ax[0].bar(p["ym"], p["rate"]); ax[0].set_title("train target rate by decision month"); ax[0].tick_params(axis="x", rotation=60)
for s in ["train", "test"]:
    q = tm.filter(pl.col("split") == s); ax[1].plot(q["ym"], q["n"] / q["n"].sum(), marker="o", label=s)
ax[1].legend(); ax[1].set_title("share of apps by month"); ax[1].tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(OUT / "01_target_time.png", dpi=90)

# coverage + post-cutoff
cut = apps.select("customer_id", "application_id", "decision_datetime", "split")
specs = {
    "previous_applications": ("customer_id", "decision_datetime"),
    "repayment_history": ("customer_id", "due_date"),
    "account_behavior": ("customer_id", "month"),
    "bureau_observations": ("customer_id", "available_datetime"),
    "ecommerce_transactions": ("customer_id", "transaction_datetime"),
}
for t, (k, tcol) in specs.items():
    d = load(t)
    j = d.join(cut.rename({"decision_datetime": "cutoff", "application_id": "sel_app"}), on=k, how="left")
    if t == "account_behavior":  # month is complete only once the month ends
        after = pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) > pl.col("cutoff")
    else:
        after = pl.col(tcol) > pl.col("cutoff")
    cov = j.group_by("split").agg(rows=pl.len(), custs=pl.col(k).n_unique(), post_cutoff_rows=after.sum())
    print(f"\n{t}:\n", cov)

# bureau: is it keyed to the selected application?
b = load("bureau_observations")
sel = set(apps["application_id"])
bb = b.with_columns(is_sel=pl.col("application_id").is_in(list(sel)))
print("\nbureau rows on selected app:", bb["is_sel"].sum(), " on previous apps:", (~bb["is_sel"]).sum())
bj = bb.filter("is_sel").join(cut.rename({"application_id": "application_id"}), on="application_id")
print("bureau(selected) available after decision:", (bj["available_datetime"] > bj["decision_datetime"]).sum(),
      " report after decision:", (bj["report_datetime"] > bj["decision_datetime"]).sum(), "of", len(bj))
print("bureau rows per selected app by split:\n", bj.group_by("split", "application_id").agg(n=pl.len()).group_by("split").agg(pl.col("n").mean().alias("mean"), pl.col("n").max().alias("max")))
# previous apps decision after selected decision?
pa = load("previous_applications").join(cut.rename({"application_id": "sel_app"}), on="customer_id")
print("prev apps with application_datetime > selected decision:", (pa["application_datetime"] > pa["decision_datetime_right"]).sum())
