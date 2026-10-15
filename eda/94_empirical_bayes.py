"""Empirical-Bayes (shrinkage) versions of the main signal channels.

Each strong channel is an average over a customer-specific number of noisy observations: lateness over n
installments, price premium z over n items, recent order rate over a short window, bureau score over 1-3 providers.
The raw mean of 2 observations is as extreme as the mean of 60, so a tree has to learn an n-dependent reliability
from the data. The posterior mean under a conjugate prior (beta-binomial, normal-normal, gamma-Poisson) does this
explicitly. Prior parameters are fitted by the method of moments on ALL customers (no labels).
Reported: univariate KS raw vs shrunk (on all train rows, NaN as own extreme bin) and the O/E score test vs v8."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import ks_nan
from common9 import OUT, crossfit_dks, load, oe_test, safe_bins, v8_ref

tr, y, p = v8_ref()
cut = pl.concat([load("train_applications").drop("target"), load("test_applications")]).select("customer_id", cutoff="decision_datetime")
D = cut.select("customer_id")

# 1 lateness: beta-binomial on installments with max_dpd > 0 (and >= 30)
r = load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
g = r.group_by("customer_id").agg(n=pl.len(), k1=(pl.col("max_days_past_due") > 0).sum(), k30=(pl.col("max_days_past_due") >= 30).sum(),
                                  dpd=pl.col("max_days_past_due").clip(upper_bound=120).cast(pl.Float64).sum())


def beta_mom(k, n):
    """Method-of-moments beta prior from per-customer proportions (customers with n >= 6)."""
    m = n >= 6
    q = k[m] / n[m]
    mu, var = q.mean(), q.var()
    within = np.mean(mu * (1 - mu) / n[m])
    s = mu * (1 - mu) / max(var - within, 1e-6) - 1
    return mu * s, (1 - mu) * s


n = g["n"].to_numpy().astype(float)
for col in ("k1", "k30"):
    k = g[col].to_numpy().astype(float)
    a, b = beta_mom(k, n)
    print(f"beta prior for {col}: a={a:.3f} b={b:.3f} (prior strength {a + b:.1f} installments)")
    g = g.with_columns(pl.Series(f"eb_{col}", (k + a) / (n + a + b)), pl.Series(f"raw_{col}", k / n))
g = g.with_columns(raw_dpd=pl.col("dpd") / pl.col("n"))
mu_d = (g["dpd"] / g["n"]).mean(); s_d = 8.0
g = g.with_columns(eb_dpd=(pl.col("dpd") + mu_d * s_d) / (pl.col("n") + s_d))
D = D.join(g.drop("n", "k1", "k30", "dpd"), on="customer_id", how="left")

# 2 price premium: normal-normal shrinkage of mean z over items
ec = load("ecommerce_transactions").join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
ec = ec.with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log())
ec = ec.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"),
                     age=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400)
zz = ec.group_by("customer_id").agg(m=pl.col("z").mean(), v=pl.col("z").var(), n=pl.len())
m_, v_, n_ = zz["m"].to_numpy(), zz["v"].to_numpy(), zz["n"].to_numpy().astype(float)
sw = np.nanmean(v_)                               # within-customer variance of item z
tau2 = max(np.nanvar(m_[n_ >= 10]) - np.nanmean(sw / n_[n_ >= 10]), 1e-6)  # between-customer variance
print(f"price premium: within var {sw:.3f}, between var {tau2:.4f} -> prior strength {sw / tau2:.1f} items")
zz = zz.with_columns(raw_pp=pl.col("m"), eb_pp=pl.col("m") * pl.col("n") / (pl.col("n") + sw / tau2))
D = D.join(zz.select("customer_id", "raw_pp", "eb_pp"), on="customer_id", how="left")

# 3 recent order rate vs own base rate: gamma-Poisson
o = ec.group_by("order_id").agg(pl.col("customer_id", "age").first())
oc = o.group_by("customer_id").agg(c30=(pl.col("age") <= 30).sum(), base=((pl.col("age") > 120) & (pl.col("age") <= 730)).sum(),
                                   span=(pl.col("age").max().clip(upper_bound=730) - 120).clip(lower_bound=14))
oc = oc.with_columns(rate=pl.col("base") / pl.col("span"))
pop = (oc["base"].sum() / oc["span"].sum())
for a0 in (2.0,):
    oc = oc.with_columns(raw_acc=(pl.col("c30") / 30) / (pl.col("rate") + 1e-3),
                         eb_acc=(pl.col("c30") + a0) / (30 * (pl.col("base") + a0 * pop / pop) / (pl.col("span") + a0 / pop) + a0))
D = D.join(oc.select("customer_id", "raw_acc", "eb_acc"), on="customer_id", how="left")

# 4 bureau score at the selected pull: mean of providers vs shrunk mean (reliability per number of providers)
bu = load("bureau_observations").join(load("train_applications").select("application_id").vstack(load("test_applications").select("application_id")),
                                      on="application_id")
bs = bu.group_by("customer_id").agg(m=pl.col("score").mean(), k=pl.col("score").is_not_null().sum())
rho = 0.815
mu_s = bs["m"].mean()
bs = bs.with_columns(raw_bc=pl.col("m"), eb_bc=mu_s + (pl.col("m") - mu_s) * (pl.col("k") * rho / (1 + (pl.col("k") - 1) * rho)))
D = D.join(bs.select("customer_id", "raw_bc", "eb_bc"), on="customer_id", how="left")

T = tr.select("customer_id").join(D, on="customer_id", how="left", maintain_order="left")
rows = []
for c in [c for c in D.columns if c.startswith("eb_")]:
    raw = "raw_" + c[3:]
    for name in (raw, c):
        x = T[name].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
        b = safe_bins(x, 10)
        chi, _ = oe_test(y, p, b)
        rows.append((c[3:], name, ks_nan(y, x), float(np.isfinite(x).mean()), chi, crossfit_dks(y, p, b)))
R = pl.DataFrame(rows, schema=["channel", "version", "KS_univariate", "coverage", "chi2_df_vs_v8", "dKS_cf"], orient="row")
with pl.Config(tbl_rows=30, float_precision=4):
    print(R)
R.write_csv(OUT / "94_empirical_bayes.csv")
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (c, nn) in zip(axs, [("k1", "rp n"), ("pp", "items"), ("acc", "orders")]):
    x0, x1 = T[f"raw_{c}"].to_numpy().astype(float), T[f"eb_{c}"].to_numpy().astype(float)
    m = np.isfinite(x0) & np.isfinite(x1)
    ax.scatter(x0[m][:8000], x1[m][:8000], s=2, c=y[m][:8000], cmap="coolwarm"); ax.set_xlabel(f"raw {c}"); ax.set_ylabel(f"shrunk {c}")
plt.tight_layout(); plt.savefig(OUT / "94_empirical_bayes.png", dpi=80)
print("saved", OUT / "94_empirical_bayes.png")
