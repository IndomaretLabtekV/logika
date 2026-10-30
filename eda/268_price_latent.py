"""Bet 6: a sharper estimate of the customer latent behind e-commerce unit prices.

In the no-history segment (30%, only application + bureau + e-commerce) pp_mean is the strongest single feature
(KS 0.39). pp_mean is a raw mean of item z-scores (log unit price vs the category norm of earlier months), so a
customer with 3 items is ranked as confidently as one with 300. The Markov block won by shrinking per-customer
counts to the population; this is the same idea for prices. All label-free, items up to the decision time.

px_shr{k}   sum z / (n + k): posterior mean under a N(0, tau2) latent with k = sigma2 / tau2
px_eb       the same with k estimated from the between/within customer variance (method of moments)
px_rank     mean within category-month percentile of the unit price (robust to the shape of the z distribution)
px_fac      category-weighted mean, weights from a one-factor model on per-customer category means (label-free)
px_fac_shr  px_fac shrunk with the factor-weighted information
px_n        item count
"""
import numpy as np
import polars as pl

import common as c
from features import PP_CATS


def items(apps):
    cut = apps.select("customer_id", cutoff=pl.col("decision_datetime"), dm=pl.col("decision_datetime").dt.truncate("1mo"))
    raw = c.load("ecommerce_transactions").with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(),
                                                        ym=pl.col("transaction_datetime").dt.truncate("1mo"))
    ms = raw.group_by("product_category", "ym").agg(n=pl.len(), s=pl.col("lu").sum(), ss=(pl.col("lu") ** 2).sum())
    ref = (cut.select("dm").unique().join(ms, how="cross").filter(pl.col("ym") < pl.col("dm"))
           .group_by("product_category", "dm").agg(pl.col("n", "s", "ss").sum())
           .with_columns(mu=pl.col("s") / pl.col("n"),
                         sd=((pl.col("ss") - pl.col("s") ** 2 / pl.col("n")) / (pl.col("n") - 1)).sqrt()))
    e = raw.join(cut, on="customer_id").filter(pl.col("transaction_datetime") <= pl.col("cutoff"))
    e = e.join(ref.select("product_category", "dm", "mu", "sd"), on=["product_category", "dm"], how="left")
    e = e.with_columns(z=(pl.col("lu") - pl.col("mu")) / pl.col("sd"),
                       pr=pl.col("lu").rank("average").over("product_category", "ym") / pl.len().over("product_category", "ym") - 0.5)
    return e.drop_nulls("z")


def main():
    apps = c.features().select("application_id", "customer_id", "decision_datetime", "target", "is_train", "has_history")
    e = items(apps)
    g = e.group_by("customer_id").agg(n=pl.len(), s=pl.col("z").sum(), ss=(pl.col("z") ** 2).sum(), pr=pl.col("pr").mean())
    # method of moments: within-customer variance sigma2, between-customer variance of the latent tau2
    within = ((g["ss"] - g["s"] ** 2 / g["n"]).sum() / (g["n"] - 1).sum())
    m = (g["s"] / g["n"]).to_numpy()
    tau2 = max(np.var(m) - within * np.mean(1 / g["n"].to_numpy()), 1e-6)
    k_eb = within / tau2
    print(f"items {e.height}, customers {g.height}, n median {g['n'].median()}, sigma2 {within:.3f}, tau2 {tau2:.4f}, k {k_eb:.1f}")

    # one-factor model on per-customer category means: loading_c ~ cov with the leave-category-out mean
    cm = e.group_by("customer_id", "product_category").agg(zc=pl.col("z").mean(), nc=pl.len())
    w = cm.pivot(on="product_category", index="customer_id", values="zc")
    tot = g.select("customer_id", "s", "n")
    cms = cm.join(tot, on="customer_id").with_columns(loo=(pl.col("s") - pl.col("zc") * pl.col("nc")) / (pl.col("n") - pl.col("nc")))
    wt = {}
    for cat in PP_CATS:
        x = cms.filter((pl.col("product_category") == cat) & (pl.col("n") > pl.col("nc")) & (pl.col("nc") >= 3))
        lam = np.cov(x["zc"].to_numpy(), x["loo"].to_numpy())[0, 1]
        resid = e.filter(pl.col("product_category") == cat)["z"].var()
        wt[cat] = lam / resid
    s = np.array(list(wt.values()))
    wt = {k: v / s.mean() for k, v in wt.items()}
    print("factor weights", {k: round(v, 2) for k, v in wt.items()})
    e = e.with_columns(wz=pl.col("product_category").replace_strict(wt, default=1.0) * pl.col("z"),
                       wi=pl.col("product_category").replace_strict(wt, default=1.0) ** 2)
    f = e.group_by("customer_id").agg(sw=pl.col("wz").sum(), si=pl.col("wi").sum(), sa=pl.col("product_category").replace_strict(wt, default=1.0).sum())
    g = g.join(f, on="customer_id")
    out = g.select("customer_id", px_n=pl.col("n").cast(pl.Float64), px_mean=pl.col("s") / pl.col("n"), px_rank=pl.col("pr"),
                   px_eb=pl.col("s") / (pl.col("n") + k_eb),
                   **{f"px_shr{k}": pl.col("s") / (pl.col("n") + k) for k in (1, 3, 10, 30)},
                   px_fac=pl.col("sw") / pl.col("sa"), px_fac_shr=pl.col("sw") / (pl.col("si") + k_eb))
    d = apps.join(out, on="customer_id", how="left", maintain_order="left")
    d.select("application_id", *[x for x in out.columns if x.startswith("px_")]).write_parquet(c.PQ / "f268_price_latent.parquet")

    t = d.filter(pl.col("is_train") == 1)
    for seg in (0, 1):
        q = t.filter(pl.col("has_history") == seg)
        y = q["target"].to_numpy()
        print(f"has_history={seg}: " + "  ".join(f"{k} {c.ks(y, np.nan_to_num(q[k].to_numpy(), nan=0.0)):.4f}"
                                                  for k in out.columns if k.startswith("px_")))


if __name__ == "__main__":
    main()
