"""H1: bureau covers every open credit account; previous_applications + repayment_history describe this lender's own
loans. Does bureau active_accounts / outstanding_balance contain the own loans, and is the external remainder new
information against the v8 OOF reference? Prints numbers, saves eda/out/100_*.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, load, ks_nan
from common9 import v8_ref, safe_bins, oe_test, crossfit_dks

apps = pl.concat([load("train_applications"), load("test_applications")], how="diagonal_relaxed")
cut = apps.select("customer_id", sel=pl.col("application_id"), cutoff=pl.col("decision_datetime"))

# own loans: approved before cutoff, scheduled end after cutoff -> open at cutoff
prev = load("previous_applications").join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff"))
ap = prev.filter(pl.col("decision") == "approved")
rep = load("repayment_history").join(cut.select("customer_id", "cutoff"), on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
paid = rep.group_by("loan_id").agg(k_due=pl.len(), unpaid=(pl.col("amount_due") - pl.col("amount_paid")).sum(),
                                   k_open=(pl.col("amount_paid") < pl.col("amount_due")).sum())
ap = ap.join(paid, on="loan_id", how="left").with_columns(pl.col("k_due", "unpaid", "k_open").fill_null(0))
r = pl.col("annual_interest_rate") / 12
k = pl.col("k_due")
ap = ap.with_columns(
    open_sched=(pl.col("k_due") < pl.col("term_months")) | (pl.col("k_open") > 0),
    # remaining principal after k scheduled payments (annuity), plus arrears
    rem_principal=(pl.col("approved_amount") * (1 + r) ** k
                   - pl.col("scheduled_monthly_installment") * ((1 + r) ** k - 1) / r).clip(lower_bound=0) + pl.col("unpaid"),
)
own = ap.group_by("customer_id").agg(
    own_active=pl.col("open_sched").sum(),
    own_out=pl.col("rem_principal").filter("open_sched").sum(),
    own_n_appr=pl.len(),
)

b = load("bureau_observations").join(cut, on="customer_id").filter(
    (pl.col("available_datetime") <= pl.col("cutoff")) & (pl.col("application_id") == pl.col("sel")))
bc = b.group_by("customer_id").agg(
    b_active_max=pl.col("active_accounts").max(), b_active_min=pl.col("active_accounts").min(),
    b_active_mean=pl.col("active_accounts").mean(),
    b_out_max=pl.col("outstanding_balance").max(), b_out_mean=pl.col("outstanding_balance").mean(),
    b_out_min=pl.col("outstanding_balance").min(),
    b_rev=pl.col("revolving_outstanding_balance").max(), b_n=pl.len(),
)
d = cut.join(bc, on="customer_id", how="left").join(own, on="customer_id", how="left").with_columns(
    pl.col("own_active", "own_out", "own_n_appr").fill_null(0))

print("== does bureau contain own open loans? ==")
print(d.group_by("own_active").agg(n=pl.len(), b_active_mean=pl.col("b_active_mean").mean(),
                                    b_active_min=pl.col("b_active_min").mean(),
                                    b_out_mean=pl.col("b_out_mean").mean(), own_out=pl.col("own_out").mean()).sort("own_active"))
x = d.drop_nulls(["b_out_mean"])
print("corr(b_out_mean, own_out) =", np.corrcoef(x["b_out_mean"], x["own_out"])[0, 1])
print("corr(b_active_mean, own_active) =", np.corrcoef(x["b_active_mean"], x["own_active"])[0, 1])
# per provider: outstanding minus own (should be >= revolving part if own installment loans are included)
pv = b.join(own, on="customer_id", how="left").with_columns(pl.col("own_active", "own_out").fill_null(0)).with_columns(
    inst_out=pl.col("outstanding_balance") - pl.col("revolving_outstanding_balance"))
print("share of provider rows with outstanding - revolving >= own_out:",
      pv.select(((pl.col("inst_out") >= pl.col("own_out") - 1) ).mean()).item())
print("share with active_accounts >= own_active:", pv.select((pl.col("active_accounts") >= pl.col("own_active")).mean()).item())
print("quantiles inst_out - own_out (own_active>0):",
      pv.filter(pl.col("own_active") > 0).select((pl.col("inst_out") - pl.col("own_out")).quantile(q).alias(str(q)) for q in (0.01, 0.1, 0.5, 0.9)).row(0))

cand = d.with_columns(
    ext_active=pl.col("b_active_max") - pl.col("own_active"),
    ext_active_mean=pl.col("b_active_mean") - pl.col("own_active"),
    ext_out=pl.col("b_out_max") - pl.col("own_out"),
    ext_out_mean=pl.col("b_out_mean") - pl.col("own_out"),
    ext_inst_out=pl.col("b_out_max") - pl.col("b_rev") - pl.col("own_out"),
    own_out_share=pl.col("own_out") / (pl.col("b_out_max") + 1),
    act_gap=pl.col("b_active_max") - pl.col("b_active_min"),
    out_gap=(pl.col("b_out_max") - pl.col("b_out_min")) / (pl.col("b_out_max") + 1),
    tot_out=pl.col("b_out_max") + pl.col("own_out"),
    tot_out_mean=pl.col("b_out_mean") + pl.col("own_out"),
).select("customer_id", "ext_active", "ext_active_mean", "ext_out", "ext_out_mean", "ext_inst_out", "own_out_share",
         "act_gap", "out_gap", "own_active", "own_out", "tot_out", "tot_out_mean")

inc = load("customer_profiles").select("customer_id", "declared_monthly_income")
cand = cand.join(inc, on="customer_id", how="left").with_columns(
    tot_dti=pl.col("tot_out") / pl.col("declared_monthly_income"),
    own_dti=pl.col("own_out") / pl.col("declared_monthly_income")).drop("declared_monthly_income")
tr, y, p = v8_ref()
t = tr.select("customer_id", "has_history").join(cand, on="customer_id", how="left", maintain_order="left")
print("\n== candidate vs v8 reference (chi2/df ~1 = nothing new; dKS cross-fit) ==")
rows = []
for c in cand.columns[1:]:
    v = t[c].cast(pl.Float64).to_numpy()
    bins = safe_bins(v, 10)
    chi, _ = oe_test(y, p, bins)
    rows.append((c, ks_nan(y, v), chi, crossfit_dks(y, p, bins)))
    print(f"{c:18s} KS {rows[-1][1]:.4f}  chi2/df {chi:6.2f}  dKS_cf {rows[-1][3]:+.5f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
hh = t["has_history"].to_numpy() == 1
for a, c in zip(ax, ["ext_active", "ext_out"]):
    v = t[c].to_numpy()
    m = hh & np.isfinite(v)
    q = np.unique(np.quantile(v[m], np.linspace(0, 1, 11)))
    bb = np.digitize(v[m], q[1:-1])
    a.plot([v[m][bb == i].mean() for i in np.unique(bb)], [y[m][bb == i].mean() for i in np.unique(bb)], "o-", label="observed")
    a.plot([v[m][bb == i].mean() for i in np.unique(bb)], [p[m][bb == i].mean() for i in np.unique(bb)], "s--", label="v8 expected")
    a.set(title=c + " (history customers)", ylabel="bad rate")
    a.legend()
fig.tight_layout()
fig.savefig(OUT / "100_external_credit.png", dpi=110)
cand.write_parquet(OUT / "100_external_credit.parquet")
