"""Variants of the 251 repayment chain, each written as its own block to be tested as a replacement for f251:
  a1 / a20   Dirichlet strength ALPHA 1 or 20 instead of 5
  fine       six states (0, 1-9, 10-29, 30-59, 60-89, 90+)
  term       251 features plus hit probability over the new loan's own horizon (min(term, 12) installments)
"""
import importlib

import numpy as np
import polars as pl

import common as c

mk = importlib.import_module("251_repayment_markov")


def chain(apps, edges, alpha):
    k = len(edges) + 1
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > e).cast(pl.Int8) for e in edges]).cast(pl.Int64))
         .with_columns(prev=pl.col("s").shift(1).over("loan_id")))
    tr = r.drop_nulls("prev").group_by("customer_id", "prev", "s").len()
    pos = {x: i for i, x in enumerate(apps["customer_id"].to_list())}
    N = np.zeros((len(pos), k, k))
    N[np.array([pos[x] for x in tr["customer_id"].to_list()]), tr["prev"].to_numpy(), tr["s"].to_numpy()] = tr["len"].to_numpy()
    pop = N.sum(0) / N.sum(0).sum(1, keepdims=True)
    P = (alpha * pop[None] + N) / (alpha + N.sum(2, keepdims=True))
    return P, N


def hits(P, steps):
    k = P.shape[1]
    A = P.copy()
    A[:, k - 1, :] = 0.0
    A[:, k - 1, k - 1] = 1.0
    v = np.zeros((len(P), k))
    v[:, 0] = 1.0
    out = []
    for _ in range(steps):
        v = np.einsum("nk,nkj->nj", v, A)
        out.append(v[:, k - 1].copy())
    return np.stack(out, 1)


def block(apps, P, N, name, extra=None):
    has = N.sum((1, 2)) > 0
    h = hits(P, 12)
    k = P.shape[1]
    f = {"mk_hit12": h[:, -1], "mk_hit6": h[:, 5], "mk_hit3": h[:, 2], "mk_entry": 1 - P[:, 0, 0],
         "mk_cure": P[:, 1:k - 1, 0].mean(1), "mk_stay90": P[:, k - 1, k - 1]}
    f = {x: np.where(has, v, np.nan) for x, v in f.items()}
    f.update(mk_n_trans=N.sum((1, 2)), mk_n_from0=N[:, 0].sum(1), mk_n_late=N[:, 1:].sum((1, 2)))
    if extra:
        f.update(extra)
    apps.select("application_id").with_columns(**{x: pl.Series(v) for x, v in f.items()}).write_parquet(c.PQ / f"f258_{name}.parquet")


def main():
    apps = c.features().select("application_id", "customer_id").join(pl.concat([c.load(t).select("application_id", "term_months") for t in ("train_applications", "test_applications")]), on="application_id", how="left")
    for name, edges, alpha in (("a1", mk.EDGES, 1.0), ("a20", mk.EDGES, 20.0), ("fine", [0, 9, 29, 59, 89], 5.0)):
        P, N = chain(apps, edges, alpha)
        block(apps, P, N, name)
    P, N = chain(apps, mk.EDGES, 5.0)
    h = hits(P, 12)
    term = np.clip(apps["term_months"].to_numpy(), 1, 12).astype(int)
    has = N.sum((1, 2)) > 0
    block(apps, P, N, "term", {"mk_hit_term": np.where(has, h[np.arange(len(h)), term - 1], np.nan)})
    print(apps["term_months"].value_counts().sort("term_months"))


if __name__ == "__main__":
    main()
