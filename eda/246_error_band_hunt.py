"""Mechanism hunt in the KS-deciding band. ~40 NEW raw aggregates (not in main232/teammate pools) are built
label-free from the raw tables (pre-decision only). Within the K64 2-seed OOF rank band 0.50-0.85 (where the
KS cutoff lives), each candidate is scored by |KS(positives vs negatives)| inside the band and by Spearman
with the calibrated residual. Exploratory screen on all 70k OOF rows: the top 5 MUST be confirmed with a
5-fold paired LGB (238) before any claim. Writes results/v66/series_oof/hunt_f<k>.parquet (label-free).
Figure: eda/out/246_band_hunt.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

import valid2 as v
from common import ROOT

D = ROOT / "data"
apps = pl.concat([pl.read_csv(D / f"{s}_applications.csv", try_parse_dates=True).select("application_id", "customer_id", "decision_datetime", "proposed_monthly_installment", "loan_amount")
                  for s in ("train", "test")])
cut = apps.select("customer_id", "decision_datetime")

# repayment: partial payments, payment-before-due habits, timing within month, loan-level worst-early behaviour
r = pl.read_csv(D / "repayment_history.csv", try_parse_dates=True).join(cut, on="customer_id").filter(pl.col("due_date") < pl.col("decision_datetime"))
r = r.sort("loan_id", "due_date").with_columns(
    paid=(pl.col("amount_paid") / pl.col("amount_due").clip(1)), ino=pl.int_range(pl.len()).over("loan_id"),
    early_d=((pl.col("due_date") - pl.col("last_payment_datetime")).dt.total_seconds() / 86400),
    pay_dom=pl.col("last_payment_datetime").dt.day(), due_dom=pl.col("due_date").dt.day(),
    age_d=((pl.col("decision_datetime") - pl.col("due_date")).dt.total_seconds() / 86400))
rf = r.group_by("customer_id").agg(
    h_partial_share=((pl.col("paid") > 0) & (pl.col("paid") < 1)).mean(),
    h_zero_share=(pl.col("paid") == 0).mean(),
    h_overpay_share=(pl.col("paid") > 1.001).mean(),
    h_early_mean=pl.col("early_d").mean(), h_early_std=pl.col("early_d").std(),
    h_paid_same_day_share=(pl.col("early_d").abs() < 1).mean(),
    h_paydom_std=pl.col("pay_dom").std(),
    h_first3_maxdpd=pl.col("max_days_past_due").filter(pl.col("ino") < 3).max(),
    h_last_inst_dpd=pl.col("days_past_due").sort_by("due_date").last(),
    h_dpd_slope=pl.corr("ino", "max_days_past_due"),
    h_recent90_paid=pl.col("paid").filter(pl.col("age_d") <= 90).mean(),
    h_recent90_dpd=pl.col("max_days_past_due").filter(pl.col("age_d") <= 90).max(),
    h_cure_share=((pl.col("max_days_past_due") > 0) & (pl.col("days_past_due") == 0)).mean(),
    h_dpd_gt60_cnt=(pl.col("max_days_past_due") >= 60).sum())
# ecommerce: weekday/night timing, basket entropy, cancellation by method, quantity spikes, price dispersion
e = pl.read_csv(D / "ecommerce_transactions.csv", try_parse_dates=True).join(cut, on="customer_id").filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
e = e.with_columns(unit=(pl.col("item_total") / pl.col("quantity").clip(1)).log(), hr=pl.col("transaction_datetime").dt.hour(),
                   age_d=((pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_seconds() / 86400))
e = e.with_columns(rel=pl.col("unit") - pl.col("unit").median().over("product_category"))
ef = e.group_by("customer_id").agg(
    h_night_share=((pl.col("hr") >= 23) | (pl.col("hr") <= 4)).mean(),
    h_weekend_share=(pl.col("transaction_datetime").dt.weekday() >= 6).mean(),
    h_rel_std=pl.col("rel").std(), h_rel_max=pl.col("rel").max(), h_rel_q90=pl.col("rel").quantile(.9),
    h_rel_trend=pl.corr("age_d", "rel"),
    h_qty_max=pl.col("quantity").max(), h_qty_mean=pl.col("quantity").mean(),
    h_cod_cancel=((pl.col("payment_method") == "COD") & (pl.col("order_status") == "cancelled")).mean(),
    h_cc_share=(pl.col("payment_method") == "credit_card").mean(),
    h_cat_n=pl.col("product_category").n_unique(),
    h_elec_rel=pl.col("rel").filter(pl.col("product_category") == "electronics").mean(),
    h_recent30_rel=pl.col("rel").filter(pl.col("age_d") <= 30).mean(),
    h_gap_max=pl.col("transaction_datetime").sort().diff().dt.total_seconds().max() / 86400)
# account: drawdowns, balance trend vs installment, income regularity
a = pl.read_csv(D / "account_behavior.csv", try_parse_dates=True).sort("customer_id", "month")
af = a.group_by("customer_id").agg(
    h_close_trend=pl.corr(pl.int_range(pl.len()), "closing_balance"),
    h_neg_close_share=(pl.col("closing_balance") < 0).mean(),
    h_maxdd=(pl.col("closing_balance").cum_max() - pl.col("closing_balance")).max(),
    h_inc_zero_share=(pl.col("income_like_credits") == 0).mean(),
    h_last_close=pl.col("closing_balance").last(), h_last_lowdays=pl.col("low_balance_days").last(),
    h_inc_last_vs_mean=pl.col("income_like_credits").last() / pl.col("income_like_credits").mean().clip(1))
# bureau: provider disagreement, score change across reports, inquiry acceleration
b = pl.read_csv(D / "bureau_observations.csv", try_parse_dates=True).join(cut, on="customer_id").filter(pl.col("available_datetime") < pl.col("decision_datetime"))
bf = b.sort("report_datetime").group_by("customer_id").agg(
    h_score_range=pl.col("score").max() - pl.col("score").min(),
    h_score_last_first=pl.col("score").drop_nulls().last() - pl.col("score").drop_nulls().first(),
    h_unscored_share=pl.col("score").is_null().mean(),
    h_inq_last=pl.col("inquiries_last_6m").last(), h_inq_max=pl.col("inquiries_last_6m").max(),
    h_delinq_last=pl.col("delinquent_accounts_last_12m").last())
f = apps.select("application_id", "customer_id", "proposed_monthly_installment").join(rf, on="customer_id", how="left").join(ef, on="customer_id", how="left") \
    .join(af, on="customer_id", how="left").join(bf, on="customer_id", how="left")
f = f.with_columns(h_inst_vs_lastclose=pl.col("proposed_monthly_installment") / pl.col("h_last_close").clip(1)).drop("customer_id", "proposed_monthly_installment")
cands = [c for c in f.columns if c.startswith("h_")]
for k in range(5):  # label-free features, identical per fold
    path = ROOT / f"results/v66/series_oof/hunt_f{k}.parquet"
    tmp = path.with_name(path.name + ".tmp"); f.write_parquet(tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)
oof = pl.read_parquet(ROOT / "results/v60/candidates/K__K64_2seed/oof.parquet").select("application_id", "target", "K")
t = oof.join(f, on="application_id")
y, s = t["target"].to_numpy(), t["K"].to_numpy()
p = IsotonicRegression(out_of_bounds="clip").fit(s, y).predict(s)
rk = (np.argsort(np.argsort(s)) + 1) / len(s)
band = (rk >= .5) & (rk <= .85)
rows = []
for c in cands:
    x = t[c].to_numpy().astype(float)
    ok = band & np.isfinite(x)
    kb = v.ks(y[ok], x[ok]) if ok.sum() > 500 and len(np.unique(x[ok])) > 1 and 0 < y[ok].mean() < 1 else np.nan
    okr = np.isfinite(x)
    rho = spearmanr(x[okr], (y - p)[okr]).statistic if okr.sum() > 500 else np.nan
    rows.append((c, kb, rho, okr.mean()))
rows.sort(key=lambda r: -np.nan_to_num(abs(r[2])))
print(f"{'candidate':24s} bandKS  rho_resid coverage")
for c, kb, rho, cov in rows:
    print(f"{c:24s} {kb:.4f}  {rho:+.4f}  {cov:.2f}")
# null reference: |rho| of a random permutation is ~1/sqrt(n)
print("null |rho| scale ~", round(1 / np.sqrt(len(y)), 4), "; top5 by |rho|:", [r[0] for r in rows[:5]])
fig, ax = plt.subplots(figsize=(7, 9))
ax.barh([r[0] for r in rows][::-1], [r[2] for r in rows][::-1]); ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("Spearman with calibrated residual (K64 2-seed OOF)"); ax.set_title("246 band hunt")
fig.tight_layout(); fig.savefig(ROOT / "eda/out/246_band_hunt.png", dpi=120)
