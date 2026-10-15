"""Are customers linked to each other? (hidden duplication / rings / clones)

A synthetic generator can reuse a template for several customers, or simulate groups (households, fraud rings)
that default together. Then a test customer's linked train customers carry label information that no per-customer
feature can see. For each candidate link key we count customers sharing an exact value and test whether the
leave-one-out bad rate of a train customer's linked train customers explains the target beyond the v8 score.

Keys: exact timestamps (selected / previous applications, bureau reports, e-commerce orders, last payments),
exact monetary values that should be unique (account opening balances, bureau balances, order totals with time),
and the full profile tuple. Chance collisions are expected for coarse keys; the residual test decides."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common9 import OUT, crossfit_dks, load, oe_test, v8_ref

tr, y, p = v8_ref()
cust_tr = tr.select("customer_id", y=pl.col("target"), p=pl.Series(p))
apps = pl.concat([load("train_applications").drop("target"), load("test_applications")])
prev, bur, ec = load("previous_applications"), load("bureau_observations"), load("ecommerce_transactions")
rep, acc, prof = load("repayment_history"), load("account_behavior"), load("customer_profiles")
orders = ec.group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "order_total", "payment_method").first(),
                                     n_items=pl.len())

KEYS = {
    "sel_app_datetime": apps.select("customer_id", k=pl.col("application_datetime").cast(pl.String)),
    "sel_decision_datetime": apps.select("customer_id", k=pl.col("decision_datetime").cast(pl.String)),
    "prev_app_datetime": prev.select("customer_id", k=pl.col("application_datetime").cast(pl.String)),
    "prev_decision_datetime": prev.select("customer_id", k=pl.col("decision_datetime").cast(pl.String)),
    "bureau_report_datetime": bur.select("customer_id", k=pl.col("report_datetime").cast(pl.String)),
    "bureau_balance_tuple": bur.filter(pl.col("outstanding_balance") > 0).select(
        "customer_id", k=pl.concat_str(["bureau_provider", "outstanding_balance", "revolving_credit_limit"], separator="|")),
    "order_datetime": orders.select("customer_id", k=pl.col("transaction_datetime").cast(pl.String)),
    "order_total_datetime": orders.select("customer_id", k=pl.concat_str([pl.col("transaction_datetime").cast(pl.String), "order_total"], separator="|")),
    "order_total_exact": orders.filter(pl.col("n_items") >= 3).select("customer_id", k=pl.col("order_total").cast(pl.String)),
    "last_payment_datetime": rep.drop_nulls("last_payment_datetime").select("customer_id", k=pl.col("last_payment_datetime").cast(pl.String)),
    "account_opening_balance": acc.filter(pl.col("opening_balance").abs() > 1000).select(
        "customer_id", k=pl.concat_str([pl.col("month").cast(pl.String), "opening_balance"], separator="|")),
    "account_income_credit": acc.filter(pl.col("income_like_credits") > 0).select("customer_id", k=pl.col("income_like_credits").cast(pl.String)),
    "profile_tuple": prof.select("customer_id", k=pl.concat_str(pl.exclude("customer_id"), separator="|")),
    "declared_income": prof.select("customer_id", k=pl.col("declared_monthly_income").cast(pl.String)),
}

rows, figs = [], []
for name, kv in KEYS.items():
    kv = kv.unique()
    g = kv.group_by("k").agg(n=pl.col("customer_id").n_unique()).filter(pl.col("n") >= 2)
    links = kv.join(g.select("k"), on="k")
    pairs = links.join(links, on="k", suffix="_m").filter(pl.col("customer_id") != pl.col("customer_id_m")).select(
        "customer_id", "customer_id_m").unique()
    mates = (pairs.join(cust_tr.select(customer_id_m="customer_id", y_m="y"), on="customer_id_m", how="left")
             .group_by("customer_id").agg(n_link=pl.len(), n_tr=pl.col("y_m").is_not_null().sum(), n_bad=pl.col("y_m").sum()))
    d = cust_tr.join(mates, on="customer_id", how="left").fill_null(0)
    yy, pp = d["y"].to_numpy(), d["p"].to_numpy()
    n_tr, n_bad = d["n_tr"].to_numpy(), d["n_bad"].to_numpy()
    b = np.where(n_tr == 0, 0, np.where(n_bad == 0, 1, 2))   # no train mate / only good mates / >=1 bad mate
    chi, tab = oe_test(yy, pp, b)
    dks = crossfit_dks(yy, pp, b)
    oe = {t[0]: t for t in tab}
    mate_rate = n_bad.sum() / max(n_tr.sum(), 1)
    linked = float((d["n_link"].to_numpy() > 0).mean())
    rows.append((name, g.height, int(g["n"].max()) if g.height else 0, linked, float(mate_rate),
                 oe.get(1, (0, 0, 0, 0, np.nan, 0))[4], oe.get(2, (0, 0, 0, 0, np.nan, 0))[4],
                 oe.get(2, (0, 0, 0, 0, 0, 0))[5], oe.get(2, (0, 0))[1], chi, dks))
R = pl.DataFrame(rows, schema=["key", "groups", "max_group", "share_linked", "mate_bad_rate", "OE_good_mates",
                               "OE_bad_mate", "z_bad_mate", "n_bad_mate", "chi2_df", "dKS_cf"], orient="row")
with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=220):
    print("base bad rate", round(y.mean(), 4), "| O/E = observed/expected bad given the v8 score; signal needs |z| >> 3")
    print(R)
R.write_csv(OUT / "90_cross_customer_links.csv")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(R["key"], R["z_bad_mate"]); ax.axhline(3, color="r", ls="--"); ax.axhline(-3, color="r", ls="--")
ax.set_ylabel("z of O/E for customers with a bad linked train customer"); ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(OUT / "90_cross_customer_links.png", dpi=80)
print("saved", OUT / "90_cross_customer_links.png")
