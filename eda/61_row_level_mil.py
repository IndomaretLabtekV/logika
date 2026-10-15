"""Multiple-instance test: is there information in the JOINT per-row attributes of each child table that our
customer aggregates miss? (The price premium was exactly such a per-row attribute, found by luck.)

For every child table each row gets a discrete signature (product of a few binned attributes). The customer label is
copied to its rows; with 5-fold cross-fitting by customer, each signature gets a smoothed log-likelihood ratio
log P(sig|bad)/P(sig|good) (rows weighted 1/n_rows of their customer, so heavy customers do not dominate).
A held-out customer is scored by the mean / sum / max LLR of its rows. No model is trained: these are counts.
Each score is then tested against the calibrated v6 OOF blend (residual chi2 and cross-fit dKS) with a noise baseline.
A table whose MIL score is far above noise carries row-level signal the v6 features do not capture."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, ks_nan, load, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v6_oof())
rng = np.random.default_rng(7)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise vs v6: chi2/df p95={CHI:.2f}  dKS p95={DKS:+.4f}")

lab = tr.select("customer_id", "target", fold=pl.Series(np.random.default_rng(0).integers(0, 5, tr.height)))
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
cut = apps.select("customer_id", cutoff="decision_datetime", inst_due="proposed_monthly_installment")
prof = load("customer_profiles").select("customer_id", inc="declared_monthly_income")
days = lambda c: (pl.col("cutoff") - pl.col(c)).dt.total_seconds() / 86400  # noqa: E731
qcut = lambda c, k: pl.col(c).qcut(k, labels=[str(i) for i in range(k)], allow_duplicates=True)  # noqa: E731
rec = lambda d: pl.when(d <= 30).then(pl.lit("r30")).when(d <= 90).then(pl.lit("r90")).when(d <= 365).then(pl.lit("r365")).otherwise(pl.lit("old"))  # noqa: E731


def sig(*exprs):
    return pl.concat_str([e.cast(pl.String).fill_null("na") for e in exprs], separator="|").alias("sig")


tables = {}
# e-commerce items (two signatures: with and without the known price premium)
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.with_columns(lu=(pl.col("item_total") / pl.col("quantity").clip(1)).log())
e = e.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"), d=days("transaction_datetime"))
tables["ec_price"] = e.select("customer_id", sig(pl.col("product_category"), qcut("z", 10), pl.col("quantity").clip(0, 3), pl.col("order_status"), rec(pl.col("d"))))
tables["ec_noprice"] = e.select("customer_id", sig(pl.col("product_category"), pl.col("payment_method"), pl.col("order_status"),
                                                   (pl.col("transaction_datetime").dt.hour() // 6), rec(pl.col("d"))))
# repayment installments
r = load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff")).join(prof, on="customer_id")
r = r.with_columns(paid=pl.col("amount_paid") / pl.col("amount_due"), d=days("due_date"), burden=pl.col("amount_due") / pl.col("inc"))
tables["rp"] = r.select("customer_id", sig(pl.col("max_days_past_due").cut([0, 29, 59, 89]), pl.col("paid").cut([0, 0.999, 1.001]),
                                           pl.col("days_past_due") > 0, rec(pl.col("d")), qcut("burden", 5)))
# account months
a = load("account_behavior").join(cut, on="customer_id").filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us")) <= pl.col("cutoff"))
a = a.with_columns(net=(pl.col("inflows") - pl.col("outflows")) / (pl.col("inflows") + 1e5), inc_share=pl.col("income_like_credits") / (pl.col("inflows") + 1e5),
                   d=days("month"))
tables["ac"] = a.select("customer_id", sig(pl.col("low_balance_days").cut([0, 5, 15]), qcut("net", 5), qcut("inc_share", 4),
                                           pl.col("minimum_balance") == 0, pl.col("d") <= 120))
# bureau reports
b = load("bureau_observations").join(cut, on="customer_id").filter(pl.col("available_datetime") <= pl.col("cutoff"))
tables["bu"] = b.select("customer_id", sig(pl.col("bureau_provider"), qcut("score", 10), qcut("utilization", 5),
                                           pl.col("delinquent_accounts_last_12m").clip(0, 2), pl.col("inquiries_last_6m").cut([0, 2])))
# previous applications
v = load("previous_applications").join(cut, on="customer_id").filter(pl.col("decision_datetime") <= pl.col("cutoff")).join(prof, on="customer_id")
v = v.with_columns(req=pl.col("requested_amount") / pl.col("inc"), d=days("decision_datetime"))
tables["pv"] = v.select("customer_id", sig(pl.col("decision"), qcut("annual_interest_rate", 5), qcut("req", 5), pl.col("term_months").cut([6, 12]), pl.col("d") <= 180))


def mil(rows: pl.DataFrame, alpha=20.0):
    """Cross-fitted per-customer mean/sum/max LLR of the row signatures, in tr order (NaN for no rows)."""
    rows = rows.join(lab, on="customer_id").with_columns(w=1 / pl.len().over("customer_id"))
    outs = []
    for k in range(5):
        fit = rows.filter(pl.col("fold") != k)
        W1, W0 = fit.filter(pl.col("target") == 1)["w"].sum(), fit.filter(pl.col("target") == 0)["w"].sum()
        nsig = fit["sig"].n_unique()
        t = fit.group_by("sig").agg(c1=(pl.col("w") * pl.col("target")).sum(), c0=(pl.col("w") * (1 - pl.col("target"))).sum())
        t = t.with_columns(llr=((pl.col("c1") + alpha / nsig) / (W1 + alpha)).log() - ((pl.col("c0") + alpha / nsig) / (W0 + alpha)).log())
        held = rows.filter(pl.col("fold") == k).join(t.select("sig", "llr"), on="sig", how="left").with_columns(pl.col("llr").fill_null(0.0))
        outs.append(held.group_by("customer_id").agg(m=pl.col("llr").mean(), s=pl.col("llr").sum(), mx=pl.col("llr").max(), mn=pl.col("llr").min()))
    return tr.select("customer_id").join(pl.concat(outs), on="customer_id", how="left", maintain_order="left")


res, scores = [], {}
for name, rows in tables.items():
    print(f"{name}: {rows.height:,} rows, {rows['sig'].n_unique()} signatures")
    M = mil(rows)
    for agg in ("m", "s", "mx", "mn"):
        x = M[agg].to_numpy().astype(float)
        scores[f"{name}_{agg}"] = x
        chi, lift, d = resid_test(y, p, x)
        res.append((f"{name}_{agg}", ks_nan(y, x), chi, d, float(np.mean(~np.isnan(x)))))
# naive Bayes over all tables: sum of per-table sums (independence across tables is an assumption)
allsum = np.nansum(np.column_stack([scores[f"{t}_s"] for t in tables if t != "ec_noprice"]), 1)
chi, lift, d = resid_test(y, p, allsum); res.append(("ALL_sum", ks(y, allsum), chi, d, 1.0))
R = pl.DataFrame(res, schema=["feature", "ks", "chi2_df", "dKS_cf", "coverage"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS)).sort("dKS_cf", descending=True)
R.write_csv(OUT / "61_row_level_mil.csv")
with pl.Config(tbl_rows=40, float_precision=4):
    print(R)
print(f"v6 blend KS {ks(y, p):.4f} ; NB-all-tables score alone KS {ks(y, allsum):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].barh(R["feature"], R["dKS_cf"]); ax[0].axvline(DKS, color="r", ls="--"); ax[0].set_title("cross-fit dKS of MIL scores vs v6 (red = noise p95)")
for lbl, m in (("good", y == 0), ("bad", y == 1)):
    ax[1].hist(allsum[m], 80, alpha=0.5, density=True, label=lbl)
ax[1].set_title("naive-Bayes row-level score (all tables) by class"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "61_row_level_mil.png", dpi=80)
