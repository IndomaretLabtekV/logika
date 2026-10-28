"""Bet 2b: two latent repayment propensities per customer on top of the population Markov chain (no label used).

States as in 251 (0, 1-29, 30-59, 60-89, 90+). Customer i shifts two logits of the population matrix:
  entry  P(0 -> late)      = sigmoid(logit(p_entry) + u_i); the late mass keeps the population split.
  cure   P(s -> 0), s=1..3 = sigmoid(logit(p_cure_s) + w_i); the non-cure mass keeps the population split.
u_i and w_i are MAP estimates under N(0, tau^2) priors, tau chosen by Laplace marginal likelihood on all customers.
Outputs latent shifts, posterior sds, and P(reach 90+ within 3/6/12 installments) from the customer matrix.
"""
import importlib

import numpy as np
import polars as pl

import common as c

mk = importlib.import_module("251_repayment_markov")
K, EDGES = mk.K, mk.EDGES
TAUS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def map_shift(base, k, n, tau, iters=30):
    """base (S,), k/n (N,S) successes/trials. MAP of a shared logit shift z per row; returns z, posterior sd, log-evidence."""
    z = np.zeros(len(k))
    for _ in range(iters):
        p = sigmoid(base[None] + z[:, None])
        g = (k - n * p).sum(1) - z / tau ** 2
        h = (n * p * (1 - p)).sum(1) + 1 / tau ** 2
        z = z + g / h
    p = np.clip(sigmoid(base[None] + z[:, None]), 1e-12, 1 - 1e-12)
    h = (n * p * (1 - p)).sum(1) + 1 / tau ** 2
    loglik = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(1)
    evidence = loglik - z ** 2 / (2 * tau ** 2) - np.log(tau) - 0.5 * np.log(h)
    return z, 1 / np.sqrt(h), evidence


def fit(base, k, n, name):
    has = n.sum(1) > 0
    ev = [map_shift(base, k[has], n[has], t)[2].sum() for t in TAUS]
    tau = TAUS[int(np.argmax(ev))]
    z, sd, _ = map_shift(base, k, n, tau)
    print(f"{name}: tau {tau} (log-evidence by tau {np.round(np.array(ev) - max(ev), 1).tolist()})")
    return z, sd


def main():
    f = c.features()
    apps = f.select("application_id", "customer_id", "target", "is_train")
    r = (c.load("repayment_history").sort("customer_id", "loan_id", "due_date")
         .with_columns(s=pl.sum_horizontal([(pl.col("max_days_past_due") > e).cast(pl.Int8) for e in EDGES]))
         .with_columns(prev=pl.col("s").shift(1).over("loan_id")))
    tr = r.drop_nulls("prev").group_by("customer_id", "prev", "s").len()
    pos = {cid: i for i, cid in enumerate(apps["customer_id"].to_list())}
    N = np.zeros((apps.height, K, K))
    N[np.array([pos[x] for x in tr["customer_id"].to_list()]), tr["prev"].to_numpy(), tr["s"].to_numpy()] = tr["len"].to_numpy()

    pop = N.sum(0) / N.sum(0).sum(1, keepdims=True)
    logit = lambda p: np.log(p / (1 - p))

    # entry: from state 0, success = any late state
    u, u_sd = fit(np.array([logit(1 - pop[0, 0])]), N[:, 0, 1:].sum(1)[:, None], N[:, 0].sum(1)[:, None], "entry u")
    # cure: from late states 1..3, success = back to 0
    w, w_sd = fit(logit(pop[1:4, 0]), N[:, 1:4, 0], N[:, 1:4].sum(2), "cure w")

    P = np.repeat(pop[None], len(N), 0)
    pe = sigmoid(logit(1 - pop[0, 0]) + u)
    P[:, 0, 0] = 1 - pe
    P[:, 0, 1:] = pe[:, None] * pop[0, 1:] / pop[0, 1:].sum()
    for s in range(1, 4):
        pc = sigmoid(logit(pop[s, 0]) + w)
        rest = pop[s, 1:] / pop[s, 1:].sum()
        P[:, s, 0] = pc
        P[:, s, 1:] = (1 - pc)[:, None] * rest[None]
    hit = mk.hit_probability(P)

    has = N.sum((1, 2)) > 0
    feats = {"lc_u": u, "lc_w": w, "lc_u_sd": u_sd, "lc_w_sd": w_sd, "lc_u_minus_w": u - w,
             "lc_hit3": hit[:, 2], "lc_hit6": hit[:, 5], "lc_hit12": hit[:, -1]}
    feats = {k: np.where(has, v, np.nan) for k, v in feats.items()}
    out = apps.select("application_id").with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    out.write_parquet(c.PQ / "f253_repayment_latent_chain.parquet")

    d = (f.filter(pl.col("is_train") == 1).select("application_id", "target", "rp_frac_dpd1", "rp_maxdpd_mean")
         .join(out, on="application_id").join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id"))
    hist = ~np.isnan(d["lc_u"].to_numpy())
    yh = d["target"].to_numpy()[hist]
    x = d["rp_frac_dpd1"].to_numpy()[hist]
    dec = np.digitize(x, np.unique(np.quantile(x, np.linspace(.1, .9, 9))))
    print(f"corr(u, w) {np.corrcoef(u[has], w[has])[0, 1]:.3f}")
    print(f"{'feature (history only)':22s} {'KS':>7s} {'|frac_dpd1':>11s}")
    for col in ["rp_frac_dpd1", "mk_hit3", "mk_entry"] + list(feats):
        s = d[col].to_numpy().astype(float)[hist]
        within = np.mean([c.ks(yh[dec == i], s[dec == i]) for i in np.unique(dec)
                          if (dec == i).sum() > 300 and 0 < yh[dec == i].mean() < 1])
        print(f"{col:22s} {c.ks(yh, s):7.4f} {within:11.4f}")


if __name__ == "__main__":
    main()
