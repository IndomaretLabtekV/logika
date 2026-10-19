"""Cross-table timing hypotheses not in the catalogue, tested against the clean v31 OOF (trees + Causilo + Kumo ctx
full, rank 1/3 each, bin-calibrated). Same residual tests as eda/103 (O/E chi2/df per safe bin, cross-fit dKS) with a
permuted-feature null. All inputs dated <= decision_datetime (account months complete before the cutoff).
  A job change vs credit / shopping history: tenure (days) - days since first previous application / first order
  B stress coincidence: e-commerce order rate in the 30 days before an installment's due date (relative to the
    customer's own rate) for late vs on-time installments
  C cash crunch coincidence: low_balance_days of the due month for late vs on-time installments
  D e-commerce spend not covered by account outflows (same months) -> spending financed elsewhere
Usage: cd eda && ../.venv/bin/python 114_cross_timing.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, ROOT, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

DAY = 86400.0
R = ROOT / "results"
f = features()
apps = f.select("customer_id", "application_id", "is_train", cutoff="decision_datetime")
cut = apps.select("customer_id", "cutoff")

# A: job change vs history
A = f.select("customer_id",
             A_tenure_minus_hist=pl.col("employment_tenure_months") * 30.44 - pl.col("pv_days_since_first"),
             A_tenure_minus_ec=pl.col("employment_tenure_months") * 30.44 - pl.col("ec_tenure_d"))

# installments due before the cutoff
rep = (load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
       .select("customer_id", "installment_id", "due_date", late=(pl.col("max_days_past_due") > 0)))
# B: order rate in the 30 days before each due date vs the customer's own 30-day rate
o = (load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
     .group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "cutoff").first()))
rate = o.group_by("customer_id").agg(
    n=pl.len(), span=((pl.col("cutoff").first() - pl.col("transaction_datetime").min()).dt.total_seconds() / DAY))
rate = rate.with_columns(r30=pl.col("n") / (pl.col("span").clip(lower_bound=30) / 30))
win = (rep.join(o.select("customer_id", "transaction_datetime"), on="customer_id")
       .filter((pl.col("transaction_datetime") <= pl.col("due_date"))
               & (pl.col("transaction_datetime") > pl.col("due_date") - pl.duration(days=30)))
       .group_by("installment_id").agg(k=pl.len()))
b = (rep.join(win, on="installment_id", how="left").with_columns(pl.col("k").fill_null(0))
     .join(rate.select("customer_id", "r30"), on="customer_id", how="left")
     .with_columns(rel=pl.col("k") / (pl.col("r30") + 0.1)))
B = b.group_by("customer_id").agg(
    B_rel_late=pl.col("rel").filter("late").mean(), B_rel_ontime=pl.col("rel").filter(~pl.col("late")).mean(),
).with_columns(B_burst_late_gap=pl.col("B_rel_late") - pl.col("B_rel_ontime"))
# C: low balance days in the due month, late vs on time
acc = (load("account_behavior").join(cut, on="customer_id")
       .filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) <= pl.col("cutoff")))
c = rep.with_columns(month=pl.col("due_date").dt.truncate("1mo").cast(pl.Date)).join(
    acc.select("customer_id", "month", "low_balance_days"), on=["customer_id", "month"], how="inner")
C = c.group_by("customer_id").agg(
    C_low_late=pl.col("low_balance_days").filter("late").mean(),
    C_low_ontime=pl.col("low_balance_days").filter(~pl.col("late")).mean(),
).with_columns(C_crunch_late_gap=pl.col("C_low_late") - pl.col("C_low_ontime"))
# D: completed e-commerce spend vs account outflows over the same complete months
e = (load("ecommerce_transactions").join(cut, on="customer_id")
     .filter((pl.col("transaction_datetime") <= pl.col("cutoff")) & (pl.col("order_status") == "completed"))
     .with_columns(month=pl.col("transaction_datetime").dt.truncate("1mo").cast(pl.Date)))
em = e.group_by("customer_id", "month").agg(spend=pl.col("item_total").sum(),
                                           spend_card=pl.col("item_total").filter(pl.col("payment_method") == "credit_card").sum())
d = acc.select("customer_id", "month", "outflows").join(em, on=["customer_id", "month"], how="left").fill_null(0)
D = d.group_by("customer_id").agg(
    D_spend_over_out=pl.col("spend").sum() / (pl.col("outflows").sum() + 1),
    D_months_spend_gt_half_out=(pl.col("spend") > 0.5 * pl.col("outflows")).mean(),
    D_card_over_out=pl.col("spend_card").sum() / (pl.col("outflows").sum() + 1),
)

cand = apps.select("customer_id", "application_id", "is_train")
for part in (A, B, C, D):
    cand = cand.join(part, on="customer_id", how="left")
cand.drop("customer_id", "is_train").write_parquet(OUT / "114_cross_timing.parquet")
tr = cand.filter(pl.col("is_train") == 1).sort("application_id")

# reference: clean v31 OOF
X, Xt, y, ids, tids = v.frames(f)
assert (tr["application_id"] == ids).all()


def load_oof(path, col):
    return pl.read_parquet(path).to_pandas().set_index("application_id")[col].loc[ids].to_numpy()


trees = np.mean([rankdata(load_oof(R / f"v22/v22/gbdt/oof_{m}.parquet", m)) for m in
                 ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
ref = (rankdata(trees) + rankdata(load_oof(R / "v22/v22/fm/oof_causilo.parquet", "causilo"))
       + rankdata(load_oof(R / "v28m/oof_kumo_large_c56000.parquet", "kumo_large_c56000"))) / 3
p = calibrate(y, ref)
print(f"reference v31 OOF KS {v.ks(y, ref):.5f}")

rng = np.random.default_rng(0)
null = [(oe_test(y, p, safe_bins(rng.permutation(X["pp_mean"].to_numpy()), 10))[0],
         crossfit_dks(y, p, safe_bins(rng.permutation(X["pp_mean"].to_numpy()), 10))) for _ in range(30)]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null (permuted feature): chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for col in [c for c in tr.columns if c[:2] in ("A_", "B_", "C_", "D_")]:
    x = tr[col].cast(pl.Float64).to_numpy()
    bb = safe_bins(x, 10)
    chi, tab = oe_test(y, p, bb)
    rows.append(dict(feature=col, cover=float(np.isfinite(x).mean()), ks_uni=v.ks_nan(y, x), chi2=chi,
                     dks=crossfit_dks(y, p, bb), oe=" ".join(f"{t[4]:.2f}" for t in tab)))
res = pl.DataFrame(rows).sort("chi2", descending=True)
with pl.Config(tbl_rows=40, tbl_cols=10, fmt_str_lengths=80, tbl_width_chars=200):
    print(res)
res.write_csv(OUT / "114_cross_timing.csv")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(res["chi2"], res["dks"], s=12)
ax.axvline(c95, color="r", lw=.5)
ax.axhline(d95, color="r", lw=.5)
for r in res.iter_rows(named=True):
    ax.annotate(r["feature"], (r["chi2"], r["dks"]), fontsize=6)
ax.set(xlabel="chi2/df vs v31 OOF", ylabel="cross-fit dKS", title="Cross-table timing candidates")
fig.tight_layout()
fig.savefig(OUT / "114_cross_timing.png", dpi=110)
