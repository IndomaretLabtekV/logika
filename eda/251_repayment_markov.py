"""Bet 2: per-customer repayment Markov chain and analytic P(reach 90+ DPD within 12 installments).

States per installment from max_days_past_due: 0 (on time), 1-29, 30-59, 60-89, 90+. Transitions are counted between
consecutive installments of the same historical loan (all rows are before the selected decision by construction).
Each customer's transition rows are shrunk to the population matrix with a Dirichlet prior (no label used).
The 90+ state is made absorbing to get the probability that a fresh 12-installment loan ever reaches it.

Reports KS and KS within deciles of the strongest existing repayment features.
"""
import numpy as np
import polars as pl

import common as c

EDGES = [0, 29, 59, 89]
K = 5
ALPHA = 5.0
STEPS = 12


def hit_probability(P, start=0, steps=STEPS):
    """P: (n, K, K) row-stochastic. Probability of being absorbed in state K-1 within `steps` from `start`."""
    A = P.copy()
    A[:, K - 1, :] = 0.0
    A[:, K - 1, K - 1] = 1.0
    v = np.zeros((len(P), K))
    v[:, start] = 1.0
    out = []
    for _ in range(steps):
        v = np.einsum("nk,nkj->nj", v, A)
        out.append(v[:, K - 1].copy())
    return np.stack(out, 1)


def main():
    f = c.features()
    apps = f.select("application_id", "customer_id", "target", "is_train")
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > e).cast(pl.Int8) for e in EDGES]))
         .with_columns(prev=pl.col("s").shift(1).over("loan_id")))
    tr = r.drop_nulls("prev").group_by("customer_id", "prev", "s").len()

    cust = apps["customer_id"].to_list()
    pos = {cid: i for i, cid in enumerate(cust)}
    N = np.zeros((len(cust), K, K))
    ci = np.array([pos[x] for x in tr["customer_id"].to_list()])
    N[ci, tr["prev"].to_numpy(), tr["s"].to_numpy()] = tr["len"].to_numpy()

    pop = N.sum(0)
    pop = pop / pop.sum(1, keepdims=True)
    post = (ALPHA * pop[None] + N) / (ALPHA + N.sum(2, keepdims=True))
    has = N.sum((1, 2)) > 0

    hit = hit_probability(post)
    n_from0 = N[:, 0].sum(1)
    n_late = N[:, 1:].sum((1, 2))
    feats = {
        "mk_hit12": hit[:, -1],
        "mk_hit6": hit[:, 5],
        "mk_hit3": hit[:, 2],
        "mk_entry": 1 - post[:, 0, 0],
        "mk_cure": post[:, 1:4, 0].mean(1),
        "mk_escalate": (post[:, 1, 2:].sum(1) + post[:, 2, 3:].sum(1) + post[:, 3, 4]) / 3,
        "mk_stay90": post[:, 4, 4],
        "mk_n_trans": N.sum((1, 2)),
        "mk_n_from0": n_from0,
        "mk_n_late": n_late,
    }
    for k in feats:
        if k.startswith("mk_") and not k.startswith("mk_n"):
            feats[k] = np.where(has, feats[k], np.nan)
    out = apps.select("application_id").with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    out.write_parquet(c.PQ / "f251_repayment_markov.parquet")

    print("population transition matrix (rows = from state 0,1-29,30-59,60-89,90+):")
    print(np.round(pop, 3))
    print(f"customers with transitions: {has.mean():.3f}")

    d = (f.filter(pl.col("is_train") == 1)
         .select("application_id", "target", "rp_frac_dpd1", "rp_maxdpd_mean", "rq_streak_max", "sv_share30_of_late")
         .join(out, on="application_id"))
    y = d["target"].to_numpy()
    hist = ~np.isnan(d["mk_hit12"].to_numpy())
    print(f"train rows with history {hist.sum()} of {len(d)}; bad rate with/without history "
          f"{y[hist].mean():.4f}/{y[~hist].mean():.4f}")

    def deciles(col):
        x = d[col].to_numpy().astype(float)[hist]
        qs = np.unique(np.nanquantile(x, np.linspace(.1, .9, 9)))
        return np.digitize(x, qs)

    dec_a, dec_b = deciles("rp_frac_dpd1"), deciles("rp_maxdpd_mean")
    yh = y[hist]
    print(f"{'feature (history only)':24s} {'KS':>7s} {'|frac_dpd1':>11s} {'|maxdpd_mean':>13s}")
    for col in ["rp_frac_dpd1", "rp_maxdpd_mean", "rq_streak_max", "sv_share30_of_late"] + list(feats):
        s = d[col].to_numpy().astype(float)[hist]
        within = []
        for dec in (dec_a, dec_b):
            ks = [c.ks(yh[dec == i], s[dec == i]) for i in np.unique(dec)
                  if (dec == i).sum() > 300 and 0 < yh[dec == i].mean() < 1 and np.isfinite(s[dec == i]).any()]
            within.append(np.mean(ks))
        print(f"{col:24s} {c.ks(yh, s):7.4f} {within[0]:11.4f} {within[1]:13.4f}")


if __name__ == "__main__":
    main()
