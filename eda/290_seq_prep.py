"""H2 data: padded per-customer event tensors (label-free, train + test) for the sequence model in 291.

Rows follow features.parquet sorted by application_id (valid2.frames order: train rows then test rows by is_train).
Tables, most recent L events, features standardised over all valid events:
  ec   e-commerce orders (48): log age, log value, items, units, categories, cancelled, payment one-hot, mean price z
  rp   repayment installments (36): log age, index, state one-hot, previous state one-hot, dpd/90, paid share,
       log amount due, unpaid, loan order
  bu   bureau pulls (24): log age, provider one-hot, score, utilisation, inquiries, delinquent, active, log debt, is current
  ac   account months (24): log age, net/income, low-balance share, log closing, log in, log out, log income, min/avg
  pv   previous applications (12): log age, decision one-hot, log requested, log approved, rate, term, log installment
  st   static application fields
Writes results/v73/seq_cache/seq.npz.
Usage: cd eda && ../.venv/bin/python 290_seq_prep.py
"""
import importlib

import numpy as np
import polars as pl

import common as c

px = importlib.import_module("268_price_latent")
OUT = c.ROOT / "results/v73/seq_cache"
L = {"ec": 48, "rp": 36, "bu": 24, "ac": 24, "pv": 12}
EDGES = [0, 29, 59, 89]


def days(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds() / 86400


def pad(df, cols, n_rows, length):
    """df: row (int customer row), t (sort key), feature cols. Keeps the last `length` events per row, left-aligned."""
    d = df.sort("row", "t").with_columns(k=pl.col("t").rank("ordinal", descending=True).over("row")).filter(pl.col("k") <= length)
    d = d.with_columns(pos=pl.col("t").rank("ordinal").over("row") - 1)
    X = d.select(cols).to_numpy().astype(np.float32)
    ok = np.isfinite(X)
    mu = np.array([X[ok[:, j], j].mean() for j in range(X.shape[1])], dtype=np.float32)
    sd = np.array([X[ok[:, j], j].std() + 1e-6 for j in range(X.shape[1])], dtype=np.float32)
    X = np.where(ok, (X - mu) / sd, 0.0).astype(np.float32)
    out = np.zeros((n_rows, length, len(cols)), dtype=np.float32)
    mask = np.zeros((n_rows, length), dtype=bool)
    r, p = d["row"].to_numpy(), d["pos"].to_numpy()
    out[r, p] = X
    mask[r, p] = True
    return out, mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = c.features().sort("application_id")
    apps = f.select("application_id", "customer_id", "decision_datetime", "is_train").with_row_index("row")
    n = apps.height
    cust = apps.select("row", "customer_id", "decision_datetime")
    arrays = {}

    # e-commerce orders
    it = px.items(f.select("application_id", "customer_id", "decision_datetime"))
    oz = it.group_by("order_id").agg(z=pl.col("z").mean())
    e = c.load("ecommerce_transactions")
    o = (e.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "order_total", "payment_method", "order_status").first(),
                                   items=pl.len(), units=pl.col("quantity").sum(), cats=pl.col("product_category").n_unique())
         .join(oz, on="order_id", how="left").join(cust, on="customer_id")
         .with_columns(t=pl.col("transaction_datetime"), lage=days("decision_datetime", "transaction_datetime").log1p(),
                       lval=pl.col("order_total").cast(pl.Float64).log(), cancel=(pl.col("order_status") != "completed").cast(pl.Float64))
         .with_columns(*[(pl.col("payment_method") == m).cast(pl.Float64).alias(f"pm_{m}") for m in e["payment_method"].unique().sort().to_list()],
                       z=pl.col("z").fill_null(0)))
    ec_cols = ["lage", "lval", "items", "units", "cats", "cancel", "z"] + [k for k in o.columns if k.startswith("pm_")]
    arrays["ec"], arrays["ec_m"] = pad(o, ec_cols, n, L["ec"])

    # repayment
    r = (c.load("repayment_history").join(cust, on="customer_id").sort("customer_id", "loan_id", "due_date")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > x).cast(pl.Int8) for x in EDGES]),
                       idx=pl.col("due_date").rank("ordinal").over("loan_id").cast(pl.Float64),
                       start=pl.col("due_date").min().over("loan_id"))
         .with_columns(ps=pl.col("s").shift(1).over("loan_id").fill_null(-1),
                       order=pl.col("start").rank("dense").over("customer_id").cast(pl.Float64),
                       t=pl.col("due_date"), lage=days("decision_datetime", "due_date").clip(lower_bound=0).log1p(),
                       dpd=(pl.col("max_days_past_due") / 90).clip(upper_bound=3),
                       paid=pl.col("amount_paid") / pl.col("amount_due"), lamt=pl.col("amount_due").cast(pl.Float64).log(),
                       unpaid=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Float64))
         .with_columns(*[(pl.col("s") == k).cast(pl.Float64).alias(f"s{k}") for k in range(5)],
                       *[(pl.col("ps") == k).cast(pl.Float64).alias(f"ps{k}") for k in range(-1, 5)]))
    rp_cols = ["lage", "idx", "dpd", "paid", "lamt", "unpaid", "order"] + [f"s{k}" for k in range(5)] + [f"ps{k}" for k in range(-1, 5)]
    arrays["rp"], arrays["rp_m"] = pad(r, rp_cols, n, L["rp"])

    # bureau
    b = (c.load("bureau_observations").join(cust.join(apps.select("row", sel="application_id"), on="row"), on="customer_id")
         .with_columns(t=pl.col("report_datetime"), lage=days("decision_datetime", "report_datetime").clip(lower_bound=0).log1p(),
                       cur=(pl.col("application_id") == pl.col("sel")).cast(pl.Float64),
                       ldebt=pl.col("outstanding_balance").cast(pl.Float64).log1p(), noscore=pl.col("score").is_null().cast(pl.Float64))
         .with_columns(*[(pl.col("bureau_provider") == p).cast(pl.Float64).alias(f"bp_{p[-1]}") for p in ("Bureau_A", "Bureau_B", "Bureau_C")]))
    bu_cols = ["lage", "score", "utilization", "inquiries_last_6m", "delinquent_accounts_last_12m", "active_accounts", "ldebt", "cur",
               "noscore", "bp_A", "bp_B", "bp_C"]
    arrays["bu"], arrays["bu_m"] = pad(b, bu_cols, n, L["bu"])

    # account months
    a = (c.load("account_behavior").join(cust, on="customer_id")
         .with_columns(t=pl.col("month").cast(pl.Datetime), lage=days("decision_datetime", "month").log1p(),
                       net=(pl.col("inflows") - pl.col("outflows")) / (pl.col("income_like_credits") + 1e5),
                       low=pl.col("low_balance_days") / 30.0, lclose=pl.col("closing_balance").cast(pl.Float64).log1p(),
                       lin=pl.col("inflows").cast(pl.Float64).log1p(), lout=pl.col("outflows").cast(pl.Float64).log1p(),
                       linc=pl.col("income_like_credits").cast(pl.Float64).log1p(),
                       minavg=pl.col("minimum_balance") / (pl.col("average_balance") + 1)))
    arrays["ac"], arrays["ac_m"] = pad(a, ["lage", "net", "low", "lclose", "lin", "lout", "linc", "minavg"], n, L["ac"])

    # previous applications
    p = (c.load("previous_applications").drop("application_id").join(cust, on="customer_id")
         .with_columns(t=pl.col("application_datetime"), lage=days("decision_datetime", "application_datetime").log1p(),
                       lreq=pl.col("requested_amount").cast(pl.Float64).log(), lapp=pl.col("approved_amount").cast(pl.Float64).log1p(),
                       linst=pl.col("scheduled_monthly_installment").cast(pl.Float64).log1p(), term=pl.col("term_months") / 18,
                       rate=pl.col("annual_interest_rate"))
         .with_columns(*[(pl.col("decision") == d).cast(pl.Float64).alias(f"d_{d}") for d in ("approved", "rejected", "cancelled")]))
    arrays["pv"], arrays["pv_m"] = pad(p, ["lage", "lreq", "lapp", "linst", "term", "rate", "d_approved", "d_rejected", "d_cancelled"], n, L["pv"])

    # static application fields
    s = f.select(pl.col("loan_amount").cast(pl.Float64).log(), "annual_interest_rate", pl.col("proposed_monthly_installment").cast(pl.Float64).log(),
                 "age", "employment_tenure_months", pl.col("declared_monthly_income").cast(pl.Float64).log1p().fill_null(0),
                 pl.col("declared_monthly_income").is_null().cast(pl.Float64).alias("inc_null"), "dependents",
                 (2 * np.pi * pl.col("dec_month") / 12).sin().alias("ms"), (2 * np.pi * pl.col("dec_month") / 12).cos().alias("mc"),
                 "app_hour", "decision_lag_h",
                 *[(pl.col(k) == v).cast(pl.Float64).alias(f"{k}_{v}") for k in ("loan_purpose", "region", "employment_type", "housing_status")
                   for v in f[k].unique().sort().to_list()])
    S = s.to_numpy().astype(np.float32)
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    arrays["st"] = np.nan_to_num(S)
    arrays["is_train"] = (apps["is_train"] == 1).to_numpy()
    arrays["application_id"] = apps["application_id"].to_numpy().astype(str)
    np.savez(OUT / "seq.npz", **arrays)
    for k, val in arrays.items():
        if k.endswith("_m"):
            print(f"{k[:-2]}: rows with events {val.any(1).mean():.3f}, mean length {val.sum(1)[val.any(1)].mean():.1f}")
    print("static", arrays["st"].shape)


if __name__ == "__main__":
    main()
