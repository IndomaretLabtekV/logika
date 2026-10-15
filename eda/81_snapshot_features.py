"""Feature rows for the historical loans, built AS OF each loan's decision time, with three leakage checks.

  A) identity: reconstructing every customer's tables at T = selected cutoff must reproduce the production feature
     matrix exactly (the repayment reset is a no-op there). This tests the reset rule against the dataset's own
     snapshot semantics (unpaid DPD = floor(cutoff - due)).
  B) future invariance: for the snapshots of order 2 (order 1 has no earlier installments), change everything dated
     after T (late payments become early full payments, extra bureau / order / account / previous-application rows
     after T). The category price reference may legitimately use an item dated after customer A's T for a customer B
     deciding in a later month, so (as in the production leak test) the future rows are injected only for snapshots
     decided in the last month: then no feature of any snapshot may move, price features included.
  C) samples of snapshot rows per class, saved for inspection.
Writes data_pq/hist_feats.parquet (one row per labelled historical loan, build_all columns + hist_term, snap_order)."""
import time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, TABLES, ks_nan, load
from features import build_all
from hist_snapshots import build_hist, hist_labels, snapshot_apps, snapshot_tables

T = {t: load(t) for t in TABLES}
t0 = time.time()
base = build_all(T).sort("application_id")
print(f"production features {base.shape} ({time.time() - t0:.0f}s)")
NUM = [c for c, d in base.schema.items() if d.is_numeric() and c not in ("target", "is_train")]


def moved(a, b, cols):
    a, b = a.sort("application_id"), b.sort("application_id")
    assert a["application_id"].equals(b["application_id"])
    f = lambda d, c: d[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy()  # noqa: E731
    return [c for c in cols if not np.allclose(f(a, c), f(b, c), rtol=1e-9, atol=1e-9)]


# A) identity at the selected cutoff
apps = pl.concat([T["train_applications"].drop("target"), T["test_applications"]], how="diagonal_relaxed")
snap = apps.select("customer_id", sel_cutoff="decision_datetime", T="decision_datetime", snap_app=pl.lit("none"))
TA = snapshot_tables(T, snap)
TA["train_applications"], TA["test_applications"] = T["train_applications"], T["test_applications"]
bad = moved(base, build_all(TA), NUM)
print(f"A) identity at the selected cutoff: {len(NUM) - len(bad)}/{len(NUM)} features identical; moved: {bad}")
assert not bad

# B) future invariance on the order-2 snapshots
lab = hist_labels(T).with_columns(k=pl.col("decision_datetime").rank("ordinal").over("customer_id"))
l1 = lab.filter(pl.col("k") == 2)
s1 = l1.select("customer_id", "sel_cutoff", T="decision_datetime", snap_app="application_id")


def snap_build(tables):
    tk = snapshot_tables(tables, s1)
    tk["train_applications"] = snapshot_apps(l1, T["train_applications"].schema)
    tk["test_applications"] = T["test_applications"].clear()
    return build_all(tk)


ref = snap_build(T)
TB = dict(T)
cutT = s1.select("customer_id", "T")
r = T["repayment_history"].join(cutT, on="customer_id", how="left")
late_after = pl.col("T").is_not_null() & (pl.col("last_payment_datetime") > pl.col("T"))
print(f"   installments due before T and paid after T in these snapshots: "
      f"{r.filter(late_after & (pl.col('due_date') <= pl.col('T'))).height:,}")
TB["repayment_history"] = r.with_columns(
    amount_paid=pl.when(late_after).then(pl.col("amount_due")).otherwise(pl.col("amount_paid")),
    last_payment_datetime=pl.when(late_after).then(pl.col("T") + pl.duration(days=1)).otherwise(pl.col("last_payment_datetime")),
    max_days_past_due=pl.when(late_after).then(0).otherwise(pl.col("max_days_past_due")),
    days_past_due=pl.when(late_after).then(0).otherwise(pl.col("days_past_due")),
).drop("T")
last_month = s1["T"].dt.truncate("1mo").max()
fut = s1.filter(pl.col("T").dt.truncate("1mo") == last_month).select("customer_id", "T")
print(f"   injecting future rows for the {fut.height:,} snapshots decided in the last month ({last_month:%Y-%m})")
first = lambda t: T[t].join(fut, on="customer_id").group_by("customer_id").first()  # noqa: E731
after = pl.col("T") + pl.duration(days=3)
inj = {
    "bureau_observations": first("bureau_observations").with_columns(available_datetime=after, report_datetime=after, score=pl.lit(300.0),
                                                                     inquiries_last_6m=pl.lit(99)),
    "ecommerce_transactions": first("ecommerce_transactions").with_columns(transaction_datetime=after, item_total=pl.lit(10**9),
                                                                           order_id=pl.lit("O_FUT") + pl.col("customer_id")),
    "account_behavior": first("account_behavior").with_columns(month=pl.col("T").dt.truncate("1mo").cast(pl.Date), outflows=pl.lit(10**10)),
    "previous_applications": first("previous_applications").with_columns(application_datetime=after, decision_datetime=after,
                                                                         decision=pl.lit("rejected"),
                                                                         application_id=pl.lit("A_FUT") + pl.col("customer_id"),
                                                                         loan_id=pl.lit("L_FUT") + pl.col("customer_id")),
}
for t, extra in inj.items():
    TB[t] = pl.concat([TB[t], extra.select(T[t].columns)], how="vertical_relaxed")
leak = snap_build(TB)
mv = moved(ref, leak, NUM)
non_price = [c for c in mv if not c.startswith("pp_")]
print(f"B) future invariance on {s1.height:,} order-2 snapshots: features moved {mv}")
assert not mv, mv

# build every snapshot
t0 = time.time()
H = build_hist(T, build_all)
H.write_parquet(PQ / "hist_feats.parquet")
print(f"historical feature rows {H.shape} in {time.time() - t0:.0f}s; bad rate {H['target'].mean():.4f}")
print(H.group_by("snap_order").agg(n=pl.len(), bad=pl.col("target").mean()).sort("snap_order"))

# C) samples per class
show = ["application_id", "target", "hist_term", "snap_order", "decision_datetime", "loan_amount", "annual_interest_rate",
        "bc_score_mean", "rp_n_inst", "rp_frac_dpd1", "rp_maxdpd_max", "ac_n_months", "ec_n_orders", "pp_mean", "pv_n"]
smp = pl.concat([H.filter(pl.col("target") == c).sample(6, seed=1) for c in (0, 1)]).select(show)
smp.write_csv(OUT / "81_class_samples.csv")
with pl.Config(tbl_cols=20, tbl_width_chars=250):
    print(smp)

# coverage: history depth of snapshot rows vs selected rows
cov = pl.DataFrame({
    "feature": ["rp_n_inst", "ac_n_months", "ec_n_orders", "pv_n", "bc_score_mean", "pp_mean"],
}).with_columns(
    main_cov=pl.Series([1 - base.filter(pl.col("is_train") == 1)[c].null_count() / 70000 for c in
                        ["rp_n_inst", "ac_n_months", "ec_n_orders", "pv_n", "bc_score_mean", "pp_mean"]]),
    hist_cov=pl.Series([1 - H[c].null_count() / H.height for c in ["rp_n_inst", "ac_n_months", "ec_n_orders", "pv_n", "bc_score_mean", "pp_mean"]]),
    main_median=pl.Series([base.filter(pl.col("is_train") == 1)[c].median() for c in ["rp_n_inst", "ac_n_months", "ec_n_orders", "pv_n", "bc_score_mean", "pp_mean"]]),
    hist_median=pl.Series([H[c].median() for c in ["rp_n_inst", "ac_n_months", "ec_n_orders", "pv_n", "bc_score_mean", "pp_mean"]]),
)
print(cov)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
mtr = base.filter(pl.col("is_train") == 1)
for i, c in enumerate(["ec_n_orders", "ac_n_months", "rp_n_inst"]):
    ax[i].hist(mtr[c].drop_nulls().clip(upper_bound=mtr[c].quantile(0.99)), 50, alpha=0.5, density=True, label="selected (2025)")
    ax[i].hist(H[c].drop_nulls().clip(upper_bound=mtr[c].quantile(0.99)), 50, alpha=0.5, density=True, label="historical snapshots")
    ax[i].set_title(c); ax[i].legend()
plt.tight_layout(); plt.savefig(OUT / "81_snapshot_depth.png", dpi=80)
print("univariate KS on historical rows, key features:",
      {c: round(ks_nan(H["target"].to_numpy(), H[c].cast(pl.Float64).to_numpy()), 3) for c in ["pp_mean", "rp_frac_dpd1", "bc_score_mean", "ec_accel_30d", "annual_interest_rate"]})
