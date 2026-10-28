"""Bet 3 screen: new label-free process blocks scored on what the current best trees miss.

Reference = equal rank mean of the 18 OOF predictions of the 6-GBDT family with f251 (seeds 42/7/2026).
A candidate is scored by mean KS inside deciles of the reference (new information), against two anchors:
  mk_hit3 inside deciles of the base family (a block that later gave +0.003 public), and a shuffled noise floor.
Candidates are per-customer state chains in the spirit of 251, shrunk to the population matrix (no label used):
  ab  account months, low-balance-days state (0 / 1-7 / 8+)
  ai  account months, income state (income_like_credits below half of the customer's own median, or zero)
  rs  repayment shortfall chain (amount_paid < amount_due)
  rw  251 chain with transitions weighted by recency (half-life 180 days before decision)
  rc  251 hit probability started from the customer's most recent installment state instead of state 0
Writes data_pq/f255_process_chains.parquet.
"""
import importlib

import numpy as np
import polars as pl
from scipy.stats import rankdata

import common as c

mk = importlib.import_module("251_repayment_markov")
FAM = c.ROOT / "results/v71/family_add_eval"
MODELS = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
ALPHA = 5.0


def family(tag):
    cols = []
    for m in MODELS:
        for s in (42, 7, 2026):
            d = pl.read_parquet(FAM / f"oof_{m}__{tag}_s{s}.parquet")
            cols.append(d)
    ids = cols[0]["application_id"]
    for d in cols:
        assert (d["application_id"] == ids).all()
    return ids, np.mean([rankdata(d[d.columns[-1]].to_numpy()) for d in cols], 0)


def counts(df, pos, k, w=None):
    """(n_customers, k, k) transition counts from columns customer_id, prev, s (optional weight w)."""
    g = df.group_by("customer_id", "prev", "s").agg((pl.col(w).sum() if w else pl.len()).alias("n"))
    N = np.zeros((len(pos), k, k))
    N[np.array([pos[x] for x in g["customer_id"].to_list()]), g["prev"].to_numpy(), g["s"].to_numpy()] = g["n"].to_numpy()
    return N


def shrink(N):
    pop = N.sum(0)
    pop = pop / pop.sum(1, keepdims=True)
    return (ALPHA * pop[None] + N) / (ALPHA + N.sum(2, keepdims=True)), N.sum((1, 2)) > 0


def hit(P, start, steps):
    """Probability of reaching the last state within `steps`, start given per row."""
    k = P.shape[1]
    A = P.copy()
    A[:, k - 1, :] = 0.0
    A[:, k - 1, k - 1] = 1.0
    v = np.zeros((len(P), k))
    v[np.arange(len(P)), start] = 1.0
    for _ in range(steps):
        v = np.einsum("nk,nkj->nj", v, A)
    return v[:, k - 1]


def account_chains(apps, pos):
    a = (c.load("account_behavior").sort("customer_id", "month")
         .with_columns(med=pl.col("income_like_credits").median().over("customer_id"))
         .with_columns(lb=pl.when(pl.col("low_balance_days") == 0).then(0)
                       .when(pl.col("low_balance_days") <= 7).then(1).otherwise(2).cast(pl.Int64),
                       inc=((pl.col("income_like_credits") < 0.5 * pl.col("med")) | (pl.col("income_like_credits") == 0))
                       .cast(pl.Int64)))
    out = {}
    for name, col, k in (("ab", "lb", 3), ("ai", "inc", 2)):
        d = a.with_columns(s=pl.col(col), prev=pl.col(col).shift(1).over("customer_id")).drop_nulls("prev")
        P, has = shrink(counts(d, pos, k))
        last = a.group_by("customer_id").agg(pl.col(col).last())
        start = np.zeros(len(pos), dtype=int)
        start[np.array([pos[x] for x in last["customer_id"].to_list()])] = last[col].to_numpy()
        f = {f"{name}_entry": 1 - P[:, 0, 0], f"{name}_persist": P[:, k - 1, k - 1],
             f"{name}_hit6": hit(P, start, 6), f"{name}_hit12": hit(P, start, 12),
             f"{name}_hit6_0": hit(P, np.zeros(len(pos), dtype=int), 6)}
        if k == 3:
            f[f"{name}_escalate"] = P[:, 1, 2]
        out.update({x: np.where(has, v, np.nan) for x, v in f.items()})
    return out


def repayment_chains(apps, pos):
    cut = apps.select("customer_id", "decision_datetime")
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .join(cut, on="customer_id")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > e).cast(pl.Int8) for e in mk.EDGES]).cast(pl.Int64),
                       short=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Int64),
                       age=(pl.col("decision_datetime") - pl.col("due_date")).dt.total_days())
         .with_columns(prev=pl.col("s").shift(1).over("loan_id"), prev_short=pl.col("short").shift(1).over("loan_id"),
                       w=0.5 ** (pl.col("age") / 180.0)))
    out = {}
    d = r.drop_nulls("prev_short").with_columns(prev=pl.col("prev_short"), s=pl.col("short"))
    P, has = shrink(counts(d, pos, 2))
    out.update({"rs_entry": np.where(has, P[:, 0, 1], np.nan), "rs_persist": np.where(has, P[:, 1, 1], np.nan)})

    d = r.drop_nulls("prev")
    Nw = counts(d, pos, mk.K, "w")
    P, has = shrink(Nw * (counts(d, pos, mk.K).sum((1, 2)) / np.maximum(Nw.sum((1, 2)), 1e-9))[:, None, None])
    out.update({"rw_entry": np.where(has, 1 - P[:, 0, 0], np.nan),
                "rw_hit3": np.where(has, hit(P, np.zeros(len(pos), dtype=int), 3), np.nan),
                "rw_hit12": np.where(has, hit(P, np.zeros(len(pos), dtype=int), 12), np.nan)})

    P0, has0 = shrink(counts(d, pos, mk.K))
    last = r.group_by("customer_id").agg(pl.col("s").last(), pl.col("age").last())
    start = np.zeros(len(pos), dtype=int)
    rows = np.array([pos[x] for x in last["customer_id"].to_list()])
    start[rows] = last["s"].to_numpy()
    age = np.full(len(pos), np.nan)
    age[rows] = last["age"].to_numpy()
    out.update({"rc_hit3": np.where(has0, hit(P0, start, 3), np.nan),
                "rc_hit12": np.where(has0, hit(P0, start, 12), np.nan),
                "rc_last_state": np.where(has0, start, np.nan), "rc_last_age": age})
    return out


def within(y, ref, s, bins=10):
    dec = np.digitize(ref, np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.mean([c.ks(y[dec == i], s[dec == i]) for i in range(bins)
                    if 0 < y[dec == i].mean() < 1 and np.isfinite(s[dec == i]).sum() > 200])


def main():
    f = c.features()
    apps = f.select("application_id", "customer_id", "decision_datetime", "target", "is_train")
    pos = {cid: i for i, cid in enumerate(apps["customer_id"].to_list())}
    feats = {}
    feats.update(account_chains(apps, pos))
    feats.update(repayment_chains(apps, pos))
    out = apps.select("application_id").with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    out.write_parquet(c.PQ / "f255_process_chains.parquet")

    ids, ref = family("f251_repayment_markov")
    _, base = family("base")
    d = pl.DataFrame({"application_id": ids}).join(
        f.select("application_id", "target"), on="application_id", how="left").join(
        out, on="application_id", how="left").join(
        pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    y = d["target"].to_numpy()
    rng = np.random.default_rng(255)
    mk3 = d["mk_hit3"].to_numpy()
    print(f"reference family KS {c.ks(y, ref):.5f} (base family {c.ks(y, base):.5f})")
    print(f"anchor mk_hit3 | base deciles {within(y, base, mk3):.4f}   | markov deciles {within(y, ref, mk3):.4f}")
    floor = [within(y, ref, rng.permutation(mk3)) for _ in range(5)]
    print(f"noise floor (shuffled mk_hit3) {np.mean(floor):.4f} +/- {np.std(floor):.4f}")
    print(f"{'feature':16s} {'KS':>7s} {'|ref dec':>9s} {'coverage':>9s}")
    for k in feats:
        s = d[k].to_numpy().astype(float)
        print(f"{k:16s} {c.ks(y, s):7.4f} {within(y, ref, s):9.4f} {np.isfinite(s).mean():9.3f}")


if __name__ == "__main__":
    main()
