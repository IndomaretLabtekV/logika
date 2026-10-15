"""Generic detector for signal the v8 blend has not used: a score test per (event type x time window).

For every train customer, r = y - p where p is the calibrated v8 OOF probability. For an event type e (an order in
category c, an installment paid late, a rejected previous application, ...) and an age window a (days before the
decision cutoff), x = number (or summed value) of such events. Under "v8 already uses everything about x",
E[r | x] = 0, and z = sum(r * x) / sqrt(sum(p(1-p) x^2)) is ~N(0,1). Every (event, window) pair is tested, so
|z| > 4 is the bar. A pattern of large z concentrated in a window shows an unexploited channel and its timing.
The same z for x centred within customers (x - mean over windows) shows timing information beyond the level."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common9 import OUT, load, v8_ref

tr, y, p = v8_ref()
cut = pl.concat([load("train_applications").drop("target"), load("test_applications")]).select("customer_id", cutoff="decision_datetime")
idx = tr.select("customer_id").with_row_index("i")
r, v = y - p, p * (1 - p)
EDGES = [0, 1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 10_000]
WIN = [f"{a}-{b}d" for a, b in zip(EDGES[:-1], EDGES[1:])]


def ages(df, col):
    d = df.join(cut, on="customer_id").filter(pl.col(col) <= pl.col("cutoff"))
    d = d.with_columns(age=(pl.col("cutoff") - pl.col(col)).dt.total_seconds() / 86400)
    return d.with_columns(w=pl.col("age").cut(EDGES[1:-1], labels=WIN, left_closed=True).cast(pl.String)).join(idx, on="customer_id")


def scan(d, table, kind, value=None):
    """d: rows with i, w and a 'kind' column; value: optional numeric column to sum instead of counting."""
    out = []
    val = pl.col(value) if value else pl.lit(1.0)
    g = d.group_by("i", "w", kind).agg(x=val.sum())
    for k in g[kind].unique().drop_nulls().to_list():
        X = np.zeros((len(y), len(WIN)))
        gk = g.filter(pl.col(kind) == k)
        X[gk["i"].to_numpy(), [WIN.index(w) for w in gk["w"].to_list()]] = gk["x"].to_numpy()
        Xc = X - X.mean(1, keepdims=True)
        for j, w in enumerate(WIN):
            for tag, x in (("level", X[:, j]), ("timing", Xc[:, j])):
                den = np.sqrt((v * x * x).sum())
                if den > 0 and (x != 0).sum() >= 300:
                    out.append((table, f"{kind}={k}" + (f" sum({value})" if value else ""), w, tag, float((r * x).sum() / den),
                                int((x != 0).sum())))
    return out


rows = []
ec = load("ecommerce_transactions").with_columns(
    lu=(pl.col("item_total") / pl.col("quantity")).log(), done=pl.col("order_status"))
ec = ec.with_columns(z=(pl.col("lu") - pl.col("lu").mean().over("product_category")) / pl.col("lu").std().over("product_category"))
e = ages(ec, "transaction_datetime").with_columns(all=pl.lit("item"), hour=(pl.col("transaction_datetime").dt.hour() // 4).cast(pl.String),
                                                  qty=pl.col("quantity").cast(pl.Float64))
for kind in ("all", "product_category", "payment_method", "done", "hour"):
    rows += scan(e, "ecommerce", kind)
rows += scan(e, "ecommerce", "product_category", "z")
rows += scan(e, "ecommerce", "product_category", "qty")
rp = load("repayment_history").with_columns(
    st=pl.when(pl.col("amount_paid") == 0).then(pl.lit("unpaid")).when(pl.col("amount_paid") < pl.col("amount_due")).then(pl.lit("partial"))
    .when(pl.col("max_days_past_due") == 0).then(pl.lit("on_time")).when(pl.col("max_days_past_due") < 30).then(pl.lit("late_1_29"))
    .otherwise(pl.lit("late_30p")),
    early=((pl.col("due_date") - pl.col("last_payment_datetime")).dt.total_seconds() / 86400).clip(lower_bound=0))
rows += scan(ages(rp, "due_date"), "repayment", "st")
rows += scan(ages(rp, "due_date").with_columns(k=pl.lit("paid")), "repayment", "k", "early")
pv = load("previous_applications").with_columns(term=pl.col("decision") + "_" + pl.col("term_months").cast(pl.String))
rows += scan(ages(pv, "application_datetime"), "previous", "decision")
rows += scan(ages(pv, "application_datetime"), "previous", "term")
bu = load("bureau_observations").with_columns(sc=669 - pl.col("score"), k=pl.lit("pull"), inq=pl.col("inquiries_last_6m").cast(pl.Float64),
                                              dq=pl.col("delinquent_accounts_last_12m").cast(pl.Float64))
b = ages(bu.filter(pl.col("available_datetime") <= pl.col("report_datetime") + pl.duration(days=400)), "available_datetime")
for val in (None, "sc", "inq", "dq"):
    rows += scan(b, "bureau", "k", val)
ac = load("account_behavior").with_columns(m=pl.col("month").cast(pl.Datetime) + pl.duration(days=31), k=pl.lit("month"),
                                           low=pl.col("low_balance_days").cast(pl.Float64),
                                           oi=(pl.col("outflows") / (pl.col("inflows") + 1)).clip(upper_bound=5),
                                           inc=pl.col("income_like_credits") / 1e6)
a = ages(ac, "m")
for val in (None, "low", "oi", "inc"):
    rows += scan(a, "account", "k", val)

R = pl.DataFrame(rows, schema=["table", "event", "window", "test", "z", "n_customers"], orient="row")
R = R.with_columns(az=pl.col("z").abs()).sort("az", descending=True)
print(f"{R.height} tests; expected max |z| under the null ~ {np.sqrt(2 * np.log(R.height)):.1f}")
with pl.Config(tbl_rows=40, fmt_str_lengths=50, tbl_width_chars=200, float_precision=2):
    print(R.head(40))
    print("count with |z|>4 per table/test:")
    print(R.filter(pl.col("az") > 4).group_by("table", "test").len().sort("len", descending=True))
R.write_csv(OUT / "91_residual_event_scan.csv")
fig, ax = plt.subplots(figsize=(12, 4))
for t, g in R.group_by("table"):
    ax.scatter(g["window"].to_list(), g["z"], s=8, label=t[0])
ax.axhline(4, color="r", ls="--"); ax.axhline(-4, color="r", ls="--"); ax.legend(); ax.set_ylabel("score-test z vs v8 residual")
plt.tight_layout(); plt.savefig(OUT / "91_residual_event_scan.png", dpi=80)
print("saved", OUT / "91_residual_event_scan.png")
