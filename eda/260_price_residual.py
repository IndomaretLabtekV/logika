"""Bet 5: price premium cleaned of non-risk drivers. The customer's mean category-month price z (pp_mean) correlates 0.25
with log income, so part of it is wealth, not risk. Regress the customer price level on profile covariates over all
customers (label-free OLS) and keep the residual; also a shrunken (empirical-Bayes) version using within-customer noise.
"""
import numpy as np
import polars as pl

import common as c


def main():
    f = c.features()
    cov = ["declared_monthly_income", "ac_income_mean", "age", "dependents", "employment_tenure_months", "ec_aov", "ec_n_orders"]
    d = f.select("application_id", "customer_id", "pp_mean", "pp_trim", "pp_done", *cov, "employment_type", "housing_status", "region")
    e = (c.load("ecommerce_transactions").join(f.select("customer_id", "decision_datetime"), on="customer_id")
         .filter(pl.col("transaction_datetime") <= pl.col("decision_datetime"))
         .with_columns(lu=(pl.col("item_total") / pl.col("quantity")).log(), ym=pl.col("transaction_datetime").dt.truncate("1mo"))
         .with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category", "ym")) / pl.col("lu").std().over("product_category", "ym")))
    cm = e.group_by("customer_id").agg(cz=pl.col("z").mean(), wv=pl.col("z").var(), n=pl.len())
    d = d.join(cm, on="customer_id", how="left")
    X = [np.ones(d.height)]
    for k in cov:
        v = d[k].cast(pl.Float64).to_numpy()
        v = np.log1p(np.clip(v, 0, None)) if k in ("declared_monthly_income", "ac_income_mean", "ec_aov", "ec_n_orders") else v
        miss = np.isnan(v)
        X += [np.where(miss, np.nanmean(v), v), miss.astype(float)]
    for k in ("employment_type", "housing_status", "region"):
        for lev in d[k].unique().drop_nulls().to_list()[1:]:
            X.append((d[k] == lev).to_numpy().astype(float))
    X = np.column_stack(X)
    y = d["cz"].to_numpy()
    m = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    pred = X @ beta
    res = np.where(m, y - pred, np.nan)
    print(f"R2 of price level on covariates {1 - np.nanvar(res[m]) / np.var(y[m]):.3f}")
    # empirical Bayes shrinkage of the residual: between variance tau2 vs noise wv/n
    noise = d["wv"].to_numpy() / d["n"].to_numpy()
    tau2 = max(np.nanvar(res) - np.nanmean(noise), 1e-6)
    shr = res * tau2 / (tau2 + noise)
    out = d.select("application_id").with_columns(pz_resid=pl.Series(res), pz_resid_eb=pl.Series(shr),
                                                   pz_wealth=pl.Series(np.where(m, pred, np.nan)))
    out.write_parquet(c.PQ / "f260_price_residual.parquet")
    t = f.select("application_id", "target", "is_train").join(out, on="application_id").join(
        d.select("application_id", "pp_mean", "cz"), on="application_id").filter(pl.col("is_train") == 1)
    yy = t["target"].to_numpy()
    for k in ("pp_mean", "cz", "pz_resid", "pz_resid_eb", "pz_wealth"):
        print(f"{k:12s} KS {c.ks(yy, t[k].to_numpy()):.4f}")


if __name__ == "__main__":
    main()
