"""Bet 7 (no-history segment): which e-commerce event attributes carry the residual of the trees+Markov model?
Each event inherits its customer's residual (target - calibrated OOF probability). Mean residual by attribute bin,
weighted so every customer counts once; z = mean / standard error over customers. Large |z| = signal the model misses.
"""
import importlib

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import common as c

scr = importlib.import_module("255_residual_screen")


def main():
    ids, ref = scr.family("f251_repayment_markov")
    f = c.features().select("application_id", "customer_id", "target", "has_history", "decision_datetime")
    d = pl.DataFrame({"application_id": ids, "ref": ref}).join(f, on="application_id")
    p = IsotonicRegression(out_of_bounds="clip").fit(d["ref"], d["target"]).predict(d["ref"])
    d = d.with_columns(res=pl.Series(d["target"].to_numpy() - p))
    for seg in (0, 1):
        s = d.filter(pl.col("has_history") == seg)
        print(f"\n===== has_history={seg}: {s.height} customers, residual sd {s['res'].std():.3f}")
        e = (c.load("ecommerce_transactions").join(s.select("customer_id", "res", "decision_datetime"), on="customer_id")
             .filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
             .with_columns(hour=pl.col("transaction_datetime").dt.hour(), dow=pl.col("transaction_datetime").dt.weekday(),
                           dom=pl.col("transaction_datetime").dt.day(), minute=pl.col("transaction_datetime").dt.minute(),
                           second=pl.col("transaction_datetime").dt.second(),
                           age_bin=((pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_days() // 30).clip(0, 24),
                           qty=pl.col("quantity").clip(1, 6), unit=(pl.col("item_total") / pl.col("quantity")),
                           n_items=pl.len().over("order_id"))
             .with_columns(unit_dec=(pl.col("unit").rank() / pl.len() * 10).over("product_category").floor().clip(0, 9),
                           cents=(pl.col("unit") % 1000).round(0), last_digit=(pl.col("item_total") % 10)))
        for col in ("hour", "dow", "dom", "minute", "second", "age_bin", "qty", "product_category", "payment_method",
                    "order_status", "unit_dec", "n_items", "last_digit"):
            # per customer share of events in each bin, then residual-weighted association
            g = (e.group_by("customer_id", col).agg(pl.len().alias("k"))
                 .with_columns(share=pl.col("k") / pl.col("k").sum().over("customer_id"))
                 .join(s.select("customer_id", "res"), on="customer_id"))
            allc = s.select("customer_id", "res").join(e.select("customer_id").unique(), on="customer_id")
            n = allc.height
            rows = []
            for lev, sub in g.group_by(col):
                x = np.zeros(n)
                idx = allc.with_row_index().join(sub.select("customer_id", "share"), on="customer_id")
                x[idx["index"].to_numpy()] = idx["share"].to_numpy()
                r = allc["res"].to_numpy()
                xc = x - x.mean()
                cor = (xc * r).sum() / np.sqrt((xc ** 2).sum() * (r ** 2).sum() + 1e-12)
                rows.append((lev[0], round(x.mean(), 4), round(cor * np.sqrt(n), 1)))
            rows.sort(key=lambda t: -abs(t[2]))
            print(f"{col:16s} top |z|:", rows[:4])


if __name__ == "__main__":
    main()
