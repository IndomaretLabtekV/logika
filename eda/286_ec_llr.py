"""Bet 14 (from 285): the e-commerce pre-decision surge as a likelihood-ratio statistic.

285 showed orders of later defaulters arrive at 2.8x the good rate in the last day, decaying smoothly to ~1x at 90 days,
and below 1 (0.7-0.9) for orders older than 120 days. For two point processes with intensities lb(t), lg(t) the optimal
per-customer statistic is sum_i log(lb(t_i)/lg(t_i)) - integral(lb - lg); window counts with flat weights only
approximate it. Same for the order value premium that also ramps up toward the decision.

w(t) = log(rate_bad(t) / rate_good(t)) on 40 log-spaced age bins, smoothed over neighbouring bins. The curve uses
labels, so it is cross-fitted: train rows of outer fold k (seed 42) get the curve fitted on the other folds, test rows
the curve fitted on all train rows.
  el_llr        sum of w over the customer's orders, minus the expected sum for the customer's exposure
  el_llr_raw    sum of w
  el_llr_90     the part from orders younger than 90 days
  el_val        sum over orders of (log value - customer mean log value) * v(t), v = bad-minus-good value gap curve
Usage: cd eda && ../.venv/bin/python 286_ec_llr.py
"""
import numpy as np
import polars as pl

import common as c
import valid2 as v

EDGES = np.r_[0, np.geomspace(0.25, 800, 40)]


def orders(apps):
    e = c.load("ecommerce_transactions")
    o = e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "order_total").first())
    return (o.join(apps.select("customer_id", "decision_datetime"), on="customer_id")
            .with_columns(age=(pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                          lval=pl.col("order_total").cast(pl.Float64).log())
            .filter(pl.col("age") >= 0)
            .with_columns(lval_c=pl.col("lval") - pl.col("lval").mean().over("customer_id")))


def curves(o, y_of):
    """o: orders of fitting customers with column target; returns (w per bin, v per bin)."""
    b = np.clip(np.searchsorted(EDGES, o["age"].to_numpy(), side="right") - 1, 0, len(EDGES) - 2)
    t = o["target"].to_numpy()
    nb, ng = y_of.sum(), (1 - y_of).sum()
    cb = np.bincount(b[t == 1], minlength=len(EDGES) - 1) + 1.0
    cg = np.bincount(b[t == 0], minlength=len(EDGES) - 1) + 1.0
    w = np.log((cb / nb) / (cg / ng))
    lv = o["lval_c"].to_numpy()
    sb = np.bincount(b[t == 1], weights=lv[t == 1], minlength=len(EDGES) - 1) / cb
    sg = np.bincount(b[t == 0], weights=lv[t == 0], minlength=len(EDGES) - 1) / cg
    k = np.array([0.25, 0.5, 0.25])
    sm = lambda x: np.convolve(np.r_[x[0], x, x[-1]], k, mode="valid")  # noqa: E731
    return sm(w), sm(sb - sg)


def score(o, w, vv):
    b = np.clip(np.searchsorted(EDGES, o["age"].to_numpy(), side="right") - 1, 0, len(EDGES) - 2)
    d = o.select("customer_id", "age").with_columns(w=pl.Series(w[b]), vt=pl.Series(vv[b]) * o["lval_c"])
    return d.group_by("customer_id").agg(el_llr_raw=pl.col("w").sum(), el_llr_90=pl.col("w").filter(pl.col("age") < 90).sum(),
                                        el_val=pl.col("vt").sum(), el_n=pl.len())


def main():
    f = c.features().sort("application_id")
    apps = f.select("application_id", "customer_id", "decision_datetime", "target", "is_train")
    o = orders(apps).join(apps.select("customer_id", "target", "is_train"), on="customer_id")
    tr = apps.filter(pl.col("is_train") == 1)
    y = tr["target"].to_numpy()
    cust = tr["customer_id"].to_numpy()
    parts = []
    for fit_idx, va_idx in v.outer(y, 42):
        fit_c = set(cust[fit_idx])
        of = o.filter(pl.col("customer_id").is_in(list(fit_c)))
        w, vv = curves(of, y[fit_idx])
        parts.append(score(o.filter(pl.col("customer_id").is_in(list(cust[va_idx]))), w, vv))
    w, vv = curves(o.filter(pl.col("is_train") == 1), y)
    print("w by age bin (days):", " ".join(f"{EDGES[i]:.1f}:{w[i]:+.2f}" for i in range(0, len(w), 3)))
    parts.append(score(o.filter(pl.col("is_train") == 0), w, vv))
    s = pl.concat(parts)
    # expected sum of w for a customer with this many orders spread like the population (removes the volume effect)
    out = apps.select("application_id", "customer_id").join(s, on="customer_id", how="left", maintain_order="left")
    mw = out.select((pl.col("el_llr_raw").sum() / pl.col("el_n").sum())).item()
    out = out.with_columns(el_llr=pl.col("el_llr_raw") - pl.col("el_n") * mw).drop("customer_id", "el_n")
    out.write_parquet(c.PQ / "f286_ec_llr.parquet")
    d = f.select("application_id", "target", "is_train", "has_history", "ec_accel_30d").join(out, on="application_id", maintain_order="left").filter(pl.col("is_train") == 1)
    for g in (0, 1):
        q = d.filter(pl.col("has_history") == g)
        yy = q["target"].to_numpy()
        print(f"has_history={g}: " + "  ".join(f"{k} {c.ks(yy, np.nan_to_num(q[k].to_numpy(), nan=0)):.4f}"
                                                for k in ["ec_accel_30d", "el_llr", "el_llr_raw", "el_llr_90", "el_val"]))


if __name__ == "__main__":
    main()
