"""Historical-loan labels: how many previous loans have an observed 12-month outcome before the selected cutoff, is
the label built correctly, and does it behave like the competition target?

Checks:
  1) counts, bad rate by term / origination quarter / rate tier, next to the competition target;
  2) label definition audit: the 90-DPD day must fall inside the 12-month window; loans that reach 90 DPD only later
     are counted separately (they must be 0 here);
  3) installment trails of random bad and good loans saved to CSV for eyeballing;
  4) customer-level link: bad rate of the SELECTED loan by number of bad historical loans (train only).
Writes data_pq/hist_labels.parquet."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from common import OUT, PQ, TABLES, load
from hist_snapshots import hist_labels

T = {t: load(t) for t in TABLES}
lab = hist_labels(T)
lab.write_parquet(PQ / "hist_labels.parquet")
tr = T["train_applications"]
print(f"labelled historical loans: {lab.height:,} from {lab['customer_id'].n_unique():,} customers; "
      f"bad rate {lab['hist_target'].mean():.4f} (selected-loan target {tr['target'].mean():.4f})")
is_test = pl.col("customer_id").is_in(T["test_applications"]["customer_id"].implode())
print(lab.group_by(test_customer=is_test).agg(n=pl.len(), bad=pl.col("hist_target").mean()))

by_term = lab.group_by("term_months").agg(n=pl.len(), bad=pl.col("hist_target").mean()).sort("term_months")
by_q = lab.group_by(q=pl.col("decision_datetime").dt.strftime("%Y-Q") + pl.col("decision_datetime").dt.quarter().cast(pl.String)) \
    .agg(n=pl.len(), bad=pl.col("hist_target").mean()).sort("q")
by_rate = lab.group_by("annual_interest_rate").agg(n=pl.len(), bad=pl.col("hist_target").mean()).sort("annual_interest_rate")
sel_rate = tr.group_by("annual_interest_rate").agg(n_sel=pl.len(), bad_sel=pl.col("target").mean())
print(by_term); print(by_q); print(by_rate.join(sel_rate, on="annual_interest_rate").sort("annual_interest_rate"))

# definition audit on installment level
r = T["repayment_history"].join(lab.select("loan_id", "hist_target", orig="decision_datetime"), on="loan_id")
r = r.with_columns(hit_day=pl.col("due_date") + pl.duration(days=90), end=pl.col("orig").dt.offset_by("12mo"))
late = r.group_by("loan_id").agg(pl.col("hist_target").first(), any90=(pl.col("max_days_past_due") >= 90).any(),
                                 in_window=((pl.col("max_days_past_due") >= 90) & (pl.col("hit_day") <= pl.col("end"))).any())
chk = late.group_by("hist_target", "any90", "in_window").len().sort("hist_target", "any90", "in_window")
print("definition audit (hist_target must equal in_window):"); print(chk)
assert (late["hist_target"].cast(pl.Boolean) == late["in_window"]).all()
print(f"loans reaching 90 DPD only after their 12-month window (labelled 0): {(late['any90'] & ~late['in_window']).sum():,}")

# installment trails, five per class
samp = pl.concat([lab.filter(pl.col("hist_target") == c).sample(5, seed=80) for c in (0, 1)])
trail = (T["repayment_history"].join(samp.select("loan_id", "hist_target", orig="decision_datetime", term="term_months"), on="loan_id")
         .sort("hist_target", "loan_id", "due_date")
         .select("hist_target", "loan_id", "term", "orig", "due_date", "amount_due", "amount_paid", "last_payment_datetime",
                 "max_days_past_due"))
trail.write_csv(OUT / "80_trails.csv")
with pl.Config(tbl_rows=40, tbl_width_chars=200):
    print(trail.group_by("hist_target", "loan_id", maintain_order=True).agg(
        term=pl.col("term").first(), dpd_path=pl.col("max_days_past_due").cast(pl.String).str.join(" ")))

# customer link to the selected loan
cl = lab.group_by("customer_id").agg(n_hist=pl.len(), n_bad=pl.col("hist_target").sum())
link = tr.join(cl, on="customer_id", how="left").with_columns(
    grp=pl.when(pl.col("n_hist").is_null()).then(pl.lit("no labelled loan"))
    .otherwise(pl.concat_str(pl.lit("bad hist = "), pl.col("n_bad").clip(upper_bound=2).cast(pl.String))))
lt = link.group_by("grp").agg(n=pl.len(), bad_rate=pl.col("target").mean()).sort("grp")
print(lt)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(by_q["q"], by_q["bad"]); ax[0].axhline(tr["target"].mean(), color="r", ls="--", label="selected-loan target")
ax[0].set_title("historical 90-DPD-in-12m rate by origination quarter"); ax[0].tick_params(axis="x", rotation=45); ax[0].legend()
x = by_rate.join(sel_rate, on="annual_interest_rate").sort("annual_interest_rate")
ax[1].plot(x["annual_interest_rate"], x["bad"], "o-", label="historical loans"); ax[1].plot(x["annual_interest_rate"], x["bad_sel"], "s-", label="selected loans")
ax[1].set_xlabel("annual interest rate tier"); ax[1].set_title("bad rate by price tier"); ax[1].legend()
ax[2].bar(lt["grp"], lt["bad_rate"]); ax[2].set_title("selected-loan bad rate by historical outcome"); ax[2].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(OUT / "80_historical_labels.png", dpi=80)
print("saved", OUT / "80_historical_labels.png")
