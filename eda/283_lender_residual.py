"""Bet 12: the lender's private information.

In a synthetic generator the offered interest rate is most likely a rate card (bureau score, purpose, amount, ...)
plus the customer's latent risk. The trees see the raw rate; the part of it that the rate card does not explain is
the lender's own read of the latent, and "rate minus E[rate | 200 other features]" is hard for trees to build.
Same idea for the decision lag (manual review time) and the loan amount asked for.

For each lender-side variable a label-free LightGBM regression on all other features (train + test, 5-fold
cross-fit) gives E[var | x]; features: lr_<var>_res = var - E, and lr_<var>_z = res / sd(res).
Inputs leave out anything computed from the variable (the installment is an annuity of amount and rate).
Usage: cd eda && ../.venv/bin/python 283_lender_residual.py
"""
import re

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import KFold

import common as c
import valid2 as v

TARGETS = {
    "rate": ("annual_interest_rate", r"annual_interest_rate|inst|dti|rate_delta"),
    "lag": ("decision_lag_h", r"decision_lag_h"),
    "amount": ("loan_amount", r"loan_amount|inst|dti|lti|amt_vs|annual_interest_rate|rate_delta"),
}


def main():
    f = c.features().sort("application_id")
    out = f.select("application_id")
    for tag, (col, drop) in TARGETS.items():
        keep = [k for k in f.columns if k not in v.ID_COLS and not re.search(drop, k)]
        X = f.select(keep).to_pandas()
        for k in keep:
            if k in v.CAT_COLS or f[k].dtype == pl.String:
                X[k] = X[k].astype("category")
            else:
                X[k] = X[k].astype(float).replace([np.inf, -np.inf], np.nan)
        t = f[col].cast(pl.Float64).to_numpy()
        if tag == "amount":
            t = np.log(t)
        pred = np.zeros(len(t))
        p = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
                 bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=8)
        for a, b in KFold(5, shuffle=True, random_state=283).split(t):
            fi, st = a[: int(len(a) * 0.9)], a[int(len(a) * 0.9):]
            m = lgb.train(p, lgb.Dataset(X.iloc[fi], t[fi]), 5000, valid_sets=[lgb.Dataset(X.iloc[st], t[st])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            pred[b] = m.predict(X.iloc[b])
        res = t - pred
        r2 = 1 - res.var() / t.var()
        print(f"{tag}: {len(keep)} inputs, cross-fitted R2 {r2:.4f}, residual sd {res.std():.5f}")
        out = out.with_columns(**{f"lr_{tag}_res": pl.Series(res), f"lr_{tag}_z": pl.Series(res / res.std())})
    out.write_parquet(c.PQ / "f283_lender_residual.parquet")
    tr = f.select("application_id", "target", "is_train", "has_history", "annual_interest_rate").join(out, on="application_id", maintain_order="left").filter(pl.col("is_train") == 1)
    y = tr["target"].to_numpy()
    for k in ["annual_interest_rate"] + [x for x in out.columns if x.endswith("_res")]:
        print(f"   KS {k:22s} {c.ks(y, tr[k].to_numpy()):.4f}")


if __name__ == "__main__":
    main()
