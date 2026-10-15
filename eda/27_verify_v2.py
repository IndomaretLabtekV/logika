"""Verify the v2 features: (a) independent pandas re-computation on 300 random customers with history,
(b) leakage injection on all five dated tables incl. previous_applications, (c) screen + decile plots + redundancy."""
import numpy as np
import pandas as pd
import polars as pl
from common import TABLES, analyze, features, load, screen
from features import build_all

f = features()
V2 = [c for c in f.columns if c.startswith(("bz_", "ob_", "rq_", "sv_", "cf_", "ec2_"))]
print(f"{len(V2)} v2 features:", V2)
fp = f.to_pandas().set_index("customer_id")
cust = np.random.default_rng(11).choice(fp.index[fp["has_history"] == 1], 300, replace=False)
sel = lambda t: load(t).filter(pl.col("customer_id").is_in(list(cust))).to_pandas()  # noqa: E731
rep, ec, bu, ac, pv = (sel(t) for t in ["repayment_history", "ecommerce_transactions", "bureau_observations",
                                         "account_behavior", "previous_applications"])
ok = {}
def chk(name, exp):
    ok.setdefault(name, []).append(bool(np.isclose(exp, fp.loc[c, name], equal_nan=True, rtol=1e-9)))
for c in cust:
    cut = fp.loc[c, "decision_datetime"]
    r = rep[(rep.customer_id == c) & (rep.due_date <= cut)].sort_values(["due_date", "loan_id"])
    first = r.sort_values("due_date").groupby("loan_id").head(1)
    chk("rq_first_inst_late", (first.max_days_past_due > 0).mean() if len(r) else np.nan)
    late = r.max_days_past_due > 0
    chk("sv_maxdpd_mean_late", r.max_days_past_due[late].mean() if late.any() else np.nan)
    runs = [0]
    for _, g in r.groupby("loan_id"):
        cur_run = 0
        for v in (g.max_days_past_due > 0):
            cur_run = cur_run + 1 if v else 0
            runs.append(cur_run)
    chk("rq_streak_max", max(runs) if late.any() else np.nan)
    a = ac[(ac.customer_id == c) & (ac.month.apply(lambda m: pd.Timestamp(m) + pd.DateOffset(months=1)) <= cut)]
    chk("cf_nonincome_share", 1 - a.income_like_credits.sum() / a.inflows.sum() if len(a) else np.nan)
    o = ec[(ec.customer_id == c) & (ec.transaction_datetime <= cut)].drop_duplicates("order_id")
    age = (cut - o.transaction_datetime).dt.total_seconds() / 86400
    chk("ec2_cancel_28", ((age <= 28) & (o.order_status != "completed")).sum())
    m90 = age <= 90
    chk("ec2_night_share_90", (o.transaction_datetime.dt.hour[m90] < 5).mean() if m90.any() else np.nan)
    p = pv[(pv.customer_id == c) & (pv.decision_datetime <= cut)]
    pa = p[p.decision == "approved"]
    chk("ob_days_since_appr", ((cut - pa.decision_datetime).dt.total_seconds() / 86400).min() if len(pa) else np.nan)
    b = bu[(bu.customer_id == c) & (bu.available_datetime <= cut)].sort_values("report_datetime")
    chk("bz_newest", b.score.dropna().iloc[-1] if b.score.notna().any() else np.nan)
    own6 = ((cut - p.application_datetime).dt.total_seconds() / 86400 <= 183).sum()
    chk("bz_inq_external", b.loc[b.application_id == fp.loc[c, "application_id"], "inquiries_last_6m"].mean() - own6)
print("\n(a) independent recomputation")
for k, v in ok.items():
    print(f"  {k:<22} match {np.mean(v):.3f}")
    assert np.mean(v) == 1.0, k

# (b) leakage injection, now also a future approved previous application
T = {t: load(t) for t in TABLES}
apps = pl.concat([T["train_applications"], T["test_applications"]], how="diagonal_relaxed").select("customer_id", "decision_datetime")
fut = apps.sample(2000, seed=1).rename({"decision_datetime": "cut"})   # own name: previous_applications has decision_datetime too
first = lambda t: T[t].join(fut, on="customer_id").group_by("customer_id").first()  # noqa: E731
d5 = pl.duration(days=5)
inj = {
    "repayment_history": first("repayment_history").with_columns(due_date=pl.col("cut") + d5,
                                                                   max_days_past_due=pl.lit(999), days_past_due=pl.lit(999)),
    "ecommerce_transactions": first("ecommerce_transactions").with_columns(
        transaction_datetime=pl.col("cut") + pl.duration(hours=1), order_id=pl.lit("O_FAKE") + pl.col("customer_id"),
        order_total=pl.lit(10**9), order_status=pl.lit("cancelled")),
    "bureau_observations": first("bureau_observations").with_columns(available_datetime=pl.col("cut") + d5,
                                                                       score=pl.lit(300.0), inquiries_last_6m=pl.lit(99)),
    "account_behavior": first("account_behavior").with_columns(
        month=pl.col("cut").dt.truncate("1mo").cast(pl.Date), outflows=pl.lit(10**10), income_like_credits=pl.lit(0)),
    "previous_applications": first("previous_applications").with_columns(
        application_datetime=pl.col("decision_datetime") - pl.duration(hours=1), decision_datetime=pl.col("cut") + d5,
        decision=pl.lit("approved"), approved_amount=pl.lit(10**9), term_months=pl.lit(6), application_id=pl.lit("A_FAKE") + pl.col("customer_id"),
        loan_id=pl.lit("L_FAKE") + pl.col("customer_id")),
}
T2 = dict(T)
for t, extra in inj.items():
    T2[t] = pl.concat([T[t], extra.select(T[t].columns)], how="vertical_relaxed")
f2 = build_all(T2).sort("application_id")
f1 = f.sort("application_id")
num = [c for c, dt in f1.schema.items() if dt.is_numeric()]
def moved(c):   # float sums may differ in the last bits when row order changes -> relative tolerance
    a, b = (x[c].cast(pl.Float64).fill_nan(None).to_numpy() for x in (f1, f2))
    return not np.allclose(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0), rtol=1e-9, atol=1e-9)
diff = [c for c in num if moved(c)]
print("\n(b) features changed by injected future rows:", diff or "none -> cutoff filtering works for all", len(num), "columns")
assert not diff

# (c) screen against the v1 residual + redundancy inside v2 + decile plots
r = screen(f.select("customer_id", *V2), "27_v2", show=len(V2))
tr = f.filter(pl.col("is_train") == 1)
C = tr.select(V2).with_columns(pl.all().cast(pl.Float64).fill_nan(None).rank()).to_pandas().corr()
pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(V2) for b in V2[i + 1:] if abs(C.loc[a, b]) > 0.9]
print("\nv2 pairs with |rho|>0.9:", pairs or "none")
analyze(f, r["feature"].head(16).to_list(), "27_v2", top_plot=16)
