"""Generator forensics: does the data carry a 'pre-default' signature at a specific distance before the decision?
Recent e-commerce activity is among the strongest no-history signals (ec_accel_30d 0.345, ec_orders_14d 0.32), so the
generator may change behaviour from day X before the decision for customers who later default. If so, a window cut
at the true X is sharper than the 7/14/30/90-day windows in features.parquet.

For train customers: per age band (days before the decision) the per-customer rate of each event and attribute,
for bads and goods, and their ratio. Bands are fine near the decision.
Usage: cd eda && ../.venv/bin/python 285_time_profile.py
"""
import numpy as np
import polars as pl

import common as c

BANDS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730, 2000]


def band(col):
    return pl.col(col).cut(BANDS[1:-1], left_closed=True)


def show(name, t):
    with pl.Config(tbl_rows=40, tbl_width_chars=160, float_precision=4):
        print(f"== {name}")
        print(t)


def main():
    apps = c.features().filter(pl.col("is_train") == 1).select("customer_id", "target", "decision_datetime", "has_history")
    nb, ng = apps["target"].sum(), (1 - apps["target"]).sum()
    width = pl.DataFrame({"b": BANDS[:-1], "w": np.diff(BANDS)}).with_columns(pl.col("b").cast(pl.Int64))

    def per_day(df, tag, extra=()):
        d = df.join(apps, on="customer_id").with_columns(age=(pl.col("decision_datetime") - pl.col("t")).dt.total_seconds() / 86400)
        d = d.filter(pl.col("age") >= 0).with_columns(b=pl.col("age").cast(pl.Float64).map_elements(lambda a: max(x for x in BANDS[:-1] if x <= a), return_dtype=pl.Int64))
        g = d.group_by("b").agg(n_bad=(pl.col("target") == 1).sum(), n_good=(pl.col("target") == 0).sum(),
                                *[pl.col(e).filter(pl.col("target") == 1).mean().alias(f"{e}_bad") for e in extra],
                                *[pl.col(e).filter(pl.col("target") == 0).mean().alias(f"{e}_good") for e in extra])
        g = g.join(width, on="b").sort("b").with_columns(
            rate_bad=pl.col("n_bad") / nb / pl.col("w"), rate_good=pl.col("n_good") / ng / pl.col("w")).with_columns(
            ratio=pl.col("rate_bad") / pl.col("rate_good"))
        cols = ["b", "w", "rate_bad", "rate_good", "ratio"] + [x for e in extra for x in (f"{e}_bad", f"{e}_good")]
        show(tag, g.select(cols))

    e = c.load("ecommerce_transactions")
    o = (e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "order_status", "payment_method", "order_total").first(),
                                   items=pl.len(), units=pl.col("quantity").sum())
         .with_columns(t=pl.col("transaction_datetime"), cancel=(pl.col("order_status") != "completed").cast(pl.Float64),
                       cod=(pl.col("payment_method") == "COD").cast(pl.Float64), lval=pl.col("order_total").log()))
    per_day(o, "e-commerce orders per customer per day (by days before decision)", ("cancel", "cod", "lval", "items"))
    r = c.load("repayment_history").with_columns(t=pl.col("due_date"), late=(pl.col("max_days_past_due") > 0).cast(pl.Float64),
                                                 short=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Float64))
    per_day(r, "repayment installments due per customer per day", ("late", "short"))
    b = c.load("bureau_observations").with_columns(t=pl.col("report_datetime"))
    per_day(b, "bureau reports per customer per day", ("score", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m"))
    p = c.load("previous_applications").with_columns(t=pl.col("application_datetime"),
                                                     rej=(pl.col("decision") == "rejected").cast(pl.Float64),
                                                     lreq=pl.col("requested_amount").log())
    per_day(p, "previous applications per customer per day", ("rej", "lreq"))
    a = c.load("account_behavior").with_columns(t=pl.col("month").dt.offset_by("1mo").cast(pl.Datetime),
                                                lowshare=pl.col("low_balance_days") / 30.0,
                                                net=(pl.col("inflows") - pl.col("outflows")) / (pl.col("income_like_credits") + 1e5))
    per_day(a, "account months (age from month end)", ("lowshare", "net"))


if __name__ == "__main__":
    main()
