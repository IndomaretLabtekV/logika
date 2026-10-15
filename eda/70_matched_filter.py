"""Matched filter for the pre-application spending burst. For a Poisson order process whose rate depends on the class
and on the week before the decision, the optimal (Neyman-Pearson) statistic is
    LLR = sum over orders of log(lambda_bad(week) / lambda_good(week))  -  integral of (lambda_bad - lambda_good)
The features in use count orders in hand-picked windows (7..90 days), which approximates this. Here the weekly
profiles are estimated per class with 5-fold cross-fitting (label counting only, no model), and the exact LLR is
computed per customer, also jointly with the price z of every item (week x price-decile profile).
Tested: univariate KS vs ec_accel_30d / pp_mean, and residual dKS vs the v6 blend with a noise baseline."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, ks, ks_nan, load, resid_test, v6_oof

W = 78  # weeks of history used
f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
p = calibrate(y, v6_oof())
rng = np.random.default_rng(70)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
print(f"noise vs v6: chi2/df p95={CHI:.2f} dKS p95={DKS:+.4f}")

lab = tr.select("customer_id", "target", fold=pl.Series(np.random.default_rng(0).integers(0, 5, tr.height)))
cut = tr.select("customer_id", cutoff="decision_datetime")
e = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
e = e.with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log())
e = e.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"))
o = e.group_by("order_id").agg(pl.col("customer_id", "cutoff", "transaction_datetime").first(), z=pl.col("z").mean())
o = o.with_columns(wk=((pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_days() // 7).cast(pl.Int32),
                   zb=pl.col("z").qcut(5, labels=[str(i) for i in range(5)]).cast(pl.String).cast(pl.Int32))
o = o.filter(pl.col("wk") < W).join(lab, on="customer_id")
first = o.group_by("customer_id").agg(span=pl.col("wk").max())  # exposure proxy: weeks since first order (capped)


def llr(keys):
    out = []
    for k in range(5):
        fit = o.filter(pl.col("fold") != k)
        ncls = lab.filter(pl.col("fold") != k).group_by("target").len()
        n1, n0 = [ncls.filter(pl.col("target") == t)["len"][0] for t in (1, 0)]
        g = fit.group_by(keys).agg(c1=(pl.col("target") == 1).sum(), c0=(pl.col("target") == 0).sum())
        g = g.with_columns(l1=(pl.col("c1") + 0.5) / n1, l0=(pl.col("c0") + 0.5) / n0).with_columns(w=(pl.col("l1") / pl.col("l0")).log(),
                                                                                                  d=pl.col("l1") - pl.col("l0"))
        comp = float(g.filter(pl.col(keys[0]) < W)["d"].sum())  # integral term for a customer observed over all weeks
        held = o.filter(pl.col("fold") == k).join(g.select(*keys, "w"), on=keys, how="left")
        out.append(held.group_by("customer_id").agg(s=pl.col("w").sum() - comp))
    M = tr.select("customer_id").join(pl.concat(out), on="customer_id", how="left", maintain_order="left")
    return M["s"].fill_null(0.0).to_numpy()


cand = {"mf_week": llr(["wk"]), "mf_week_x_price": llr(["wk", "zb"])}
rows = []
for c, x in cand.items():
    rows.append((c, ks(y, x), *resid_test(y, p, x)))
for c in ("ec_accel_30d", "ec_orders_30d", "pp_mean"):
    x = tr[c].cast(pl.Float64).to_numpy()
    rows.append((c, ks_nan(y, x), *resid_test(y, p, x)))
R = pl.DataFrame(rows, schema=["feature", "ks", "chi2_df", "max_lift", "dKS_cf"], orient="row").with_columns(
    above=(pl.col("chi2_df") > CHI) & (pl.col("dKS_cf") > DKS))
with pl.Config(float_precision=4):
    print(R)
R.write_csv(OUT / "70_matched_filter.csv")
print("corr(mf_week, ec_accel_30d) =", round(float(pl.DataFrame({"a": cand["mf_week"], "b": tr["ec_accel_30d"]}).select(pl.corr("a", "b", method="spearman"))[0, 0]), 3))

prof = o.group_by("wk", "target").len().join(lab.group_by("target").agg(n=pl.len()), on="target").with_columns(r=pl.col("len") / pl.col("n")).sort("wk")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for t in (0, 1):
    q = prof.filter(pl.col("target") == t); ax[0].plot(q["wk"], q["r"], label=f"target={t}")
ax[0].invert_xaxis(); ax[0].set_title("orders per customer per week (filter shape)"); ax[0].legend()
for t, lbl in ((0, "good"), (1, "bad")):
    ax[1].hist(cand["mf_week_x_price"][y == t], 80, alpha=0.5, density=True, label=lbl)
ax[1].set_title("matched-filter LLR (week x price) by class"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "70_matched_filter.png", dpi=80)
