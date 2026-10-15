"""Sequence information test. eda/61 scored rows independently; here every row is paired with the PREVIOUS row of
the same unit (installment t-1 -> t of the same loan, account month m-1 -> m, order k-1 -> k of the customer), and
the bigram signature gets a cross-fitted naive-Bayes log-likelihood ratio (counting only, 5 folds by customer).
If bigram scores add nothing over the v6 blend, order/transition structure is already captured, and a sequence model
(GRU/transformer over raw events) has no information to find that the aggregates miss."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks_nan, load, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v6_oof())
rng = np.random.default_rng(72)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise vs v6: chi2/df p95={CHI:.2f} dKS p95={DKS:+.4f}")
lab = tr.select("customer_id", "target", fold=pl.Series(np.random.default_rng(0).integers(0, 5, tr.height)))
cut = tr.select("customer_id", cutoff="decision_datetime")
S = lambda *e: pl.concat_str([x.cast(pl.String).fill_null("na") for x in e], separator="|").alias("sig")  # noqa: E731

r = load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff")).sort("loan_id", "due_date", "installment_id")
r = r.with_columns(st=pl.col("max_days_past_due").cut([0, 29, 59, 89], labels=["0", "1", "30", "60", "90"]).cast(pl.String),
                   pd_=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Int8))
r = r.with_columns(prev=pl.col("st").shift(1).over("loan_id"), prev2=pl.col("st").shift(2).over("loan_id"))
a = load("account_behavior").join(cut, on="customer_id").filter(pl.col("month").dt.offset_by("1mo").cast(pl.Datetime("us")) <= pl.col("cutoff")).sort("customer_id", "month")
a = a.with_columns(lb=pl.col("low_balance_days").cut([0, 5, 15], labels=["0", "1", "6", "16"]).cast(pl.String),
                   nf=((pl.col("inflows") - pl.col("outflows")) / (pl.col("inflows") + 1e5)).cut([-0.1, 0, 0.1], labels=["--", "-", "+", "++"]).cast(pl.String),
                   inc=(pl.col("income_like_credits") / (pl.col("income_like_credits").shift(1).over("customer_id") + 1e5)).cut([0.8, 1.2], labels=["drop", "flat", "rise"]).cast(pl.String))
a = a.with_columns(plb=pl.col("lb").shift(1).over("customer_id"), pnf=pl.col("nf").shift(1).over("customer_id"),
                   recent=(pl.col("month").rank("ordinal", descending=True).over("customer_id") <= 3))
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log())
e = e.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"))
o = e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "cutoff", "order_status", "payment_method").first(), z=pl.col("z").mean())
o = o.sort("customer_id", "transaction_datetime", "order_id").with_columns(
    gap=(pl.col("transaction_datetime").diff().over("customer_id").dt.total_hours() / 24).cut([1, 7, 30], labels=["<1d", "<7d", "<30d", "30d+"]).cast(pl.String),
    zb=pl.col("z").cut([-0.5, 0, 0.5], labels=["vlo", "lo", "hi", "vhi"]).cast(pl.String),
    rec=((pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days() <= 90))
o = o.with_columns(pzb=pl.col("zb").shift(1).over("customer_id"), pst=pl.col("order_status").shift(1).over("customer_id"))

tables = {
    "rp_bigram": r.select("customer_id", S(pl.col("prev"), pl.col("st"), pl.col("pd_"))),
    "rp_trigram": r.select("customer_id", S(pl.col("prev2"), pl.col("prev"), pl.col("st"))),
    "ac_bigram": a.select("customer_id", S(pl.col("plb"), pl.col("lb"), pl.col("pnf"), pl.col("nf"), pl.col("inc"), pl.col("recent"))),
    "ec_bigram": o.select("customer_id", S(pl.col("gap"), pl.col("pzb"), pl.col("zb"), pl.col("pst"), pl.col("order_status"), pl.col("rec"))),
}


def mil(rows, alpha=20.0):
    rows = rows.join(lab, on="customer_id").with_columns(w=1 / pl.len().over("customer_id"))
    outs = []
    for k in range(5):
        fit = rows.filter(pl.col("fold") != k)
        W1, W0 = fit.filter(pl.col("target") == 1)["w"].sum(), fit.filter(pl.col("target") == 0)["w"].sum()
        ns = fit["sig"].n_unique()
        t = fit.group_by("sig").agg(c1=(pl.col("w") * pl.col("target")).sum(), c0=(pl.col("w") * (1 - pl.col("target"))).sum())
        t = t.with_columns(llr=((pl.col("c1") + alpha / ns) / (W1 + alpha)).log() - ((pl.col("c0") + alpha / ns) / (W0 + alpha)).log())
        h = rows.filter(pl.col("fold") == k).join(t.select("sig", "llr"), on="sig", how="left").with_columns(pl.col("llr").fill_null(0.0))
        outs.append(h.group_by("customer_id").agg(m=pl.col("llr").mean(), s=pl.col("llr").sum(), mx=pl.col("llr").max()))
    return tr.select("customer_id").join(pl.concat(outs), on="customer_id", how="left", maintain_order="left")


res = []
for name, rows in tables.items():
    M = mil(rows)
    print(f"{name}: {rows.height:,} rows, {rows['sig'].n_unique()} signatures", flush=True)
    for agg in ("m", "s", "mx"):
        x = M[agg].to_numpy().astype(float)
        chi, lift, d = resid_test(y, p, x)
        res.append((f"{name}_{agg}", ks_nan(y, x), chi, d))
R = pl.DataFrame(res, schema=["feature", "ks", "chi2_df", "dKS_cf"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS)).sort("dKS_cf", descending=True)
with pl.Config(tbl_rows=20, float_precision=4):
    print(R)
R.write_csv(OUT / "72_sequence_bigrams.csv")
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(R["feature"], R["dKS_cf"]); ax.axvline(DKS, color="r", ls="--"); ax.set_title("sequence (bigram) MIL scores: dKS vs v6")
plt.tight_layout(); plt.savefig(OUT / "72_sequence_bigrams.png", dpi=80)
