"""Bet 1: per-customer distress onset (change point) from weekly activity before the decision.

For each customer, weekly counts over the last W weeks before decision_datetime (strictly before the cutoff):
completed e-commerce orders and previous loan applications. For every candidate onset T (recent segment = weeks
0..T-1), a two-rate Poisson likelihood ratio against a constant rate is computed; the joint version sums the
log-likelihood ratios of both series at the same T. No label is used to build the features.

Reports univariate KS vs ec_accel_30d, KS within ec_accel_30d deciles (new information), onset distribution by
class, and one figure.
"""
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as c

W = 104
TS = np.arange(2, 53)


def weekly_counts(apps, events, time_col, key):
    """(n_apps, W) counts of distinct `key` per week before decision; week 0 = the 7 days before decision."""
    cut = apps.select("customer_id", pl.col("decision_datetime").alias("cutoff"), "row")
    ev = (events.join(cut, on="customer_id")
          .filter(pl.col(time_col) < pl.col("cutoff"))
          .with_columns(week=((pl.col("cutoff") - pl.col(time_col)).dt.total_seconds() // (7 * 86400))
                        .cast(pl.Int64))
          .filter(pl.col("week") < W)
          .group_by("row", "week").agg(pl.col(key).n_unique().alias("n")))
    m = np.zeros((apps.height, W))
    m[ev["row"].to_numpy(), ev["week"].to_numpy()] = ev["n"].to_numpy()
    return m


def xlogx_rate(k, t):
    """k * log(k / t) with 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, k * np.log(k / t), 0.0)


def llr_curve(m, exposure):
    """(n, len(TS)) Poisson change-point LLR and log rate ratio (recent vs older) for each candidate onset."""
    cum = np.cumsum(m, axis=1)
    total = cum[np.arange(len(m)), exposure - 1]
    a = cum[:, TS - 1]
    b = total[:, None] - a
    t_old = exposure[:, None] - TS[None, :]
    valid = t_old >= 4
    llr = xlogx_rate(a, TS[None, :]) + xlogx_rate(b, np.maximum(t_old, 1)) - xlogx_rate(total, exposure)[:, None]
    ratio = np.log((a + 0.5) / TS[None, :]) - np.log((b + 0.5) / np.maximum(t_old, 1))
    llr = np.where(valid, llr, -np.inf)
    return llr, ratio


def summarize(llr, ratio, name):
    """Best onset by LLR, restricted to increases (ratio > 0); plus a soft posterior-weighted ratio."""
    up = np.where(ratio > 0, llr, -np.inf)
    best = up.argmax(axis=1)
    rows = np.arange(len(llr))
    best_llr = np.where(np.isfinite(up[rows, best]), up[rows, best], 0.0)
    fin = np.isfinite(llr)
    mx = np.where(fin, llr, -1e300).max(axis=1, keepdims=True)
    w = np.where(fin, np.exp(np.clip(np.where(fin, llr, 0) - mx, -50, 0)), 0.0)
    soft = (w * ratio).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    soft_t = (w * TS[None, :]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    return {f"{name}_llr": best_llr, f"{name}_onset": np.where(best_llr > 0, TS[best], np.nan),
            f"{name}_ratio": np.where(best_llr > 0, ratio[rows, best], 0.0),
            f"{name}_soft_ratio": soft, f"{name}_soft_onset": soft_t,
            f"{name}_signed": np.sign(ratio[rows, np.where(fin, llr, -1e300).argmax(axis=1)]) * mx[:, 0].clip(0)}


def main():
    f = c.features()
    apps = f.select("application_id", "customer_id", "decision_datetime", "target", "is_train").with_row_index("row")
    ec = c.load("ecommerce_transactions").filter(pl.col("order_status") == "completed")
    pv = c.load("previous_applications")

    m_ec = weekly_counts(apps, ec, "transaction_datetime", "order_id")
    m_pv = weekly_counts(apps, pv, "application_datetime", "application_id")

    first = (ec.join(apps.select("customer_id", "decision_datetime", "row"), on="customer_id")
             .filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
             .group_by("row").agg(((pl.col("decision_datetime").first() - pl.col("transaction_datetime").min())
                                   .dt.total_seconds() // (7 * 86400) + 1).alias("wk")))
    exposure = np.full(apps.height, W)
    exposure[first["row"].to_numpy()] = np.minimum(first["wk"].to_numpy(), W)
    exposure = np.maximum(exposure, TS.max() + 4)

    llr_ec, r_ec = llr_curve(m_ec, exposure)
    llr_pv, r_pv = llr_curve(m_pv, exposure)
    feats = {}
    feats.update(summarize(llr_ec, r_ec, "do_ec"))
    feats.update(summarize(llr_pv, r_pv, "do_pv"))
    # Joint evidence: both series must move up at the same onset.
    joint = np.where((r_ec > 0) & (r_pv > 0), llr_ec + llr_pv, np.where(r_ec > 0, llr_ec, -np.inf))
    feats.update(summarize(joint, r_ec, "do_joint"))
    out = apps.select("application_id").with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    out.write_parquet(c.PQ / "f250_distress_onset.parquet")

    tr = f.filter(pl.col("is_train") == 1)
    d = tr.select("application_id", "target", "ec_accel_30d", "rp_frac_dpd1", "pp_mean").join(out, on="application_id")
    y = d["target"].to_numpy()
    acc = d["ec_accel_30d"].to_numpy()
    dec = np.digitize(acc, np.nanquantile(acc, np.linspace(.1, .9, 9)))

    print(f"train rows {len(d)}; bad rate {y.mean():.4f}")
    print(f"{'feature':24s} {'KS':>7s} {'KS|accel dec':>13s} {'corr accel':>11s}")
    for col in ["ec_accel_30d", "pp_mean", "rp_frac_dpd1"] + list(feats):
        s = d[col].to_numpy().astype(float)
        k = c.ks(y, s)
        within = np.mean([c.ks(y[dec == i], s[dec == i]) for i in np.unique(dec)
                          if (dec == i).sum() > 500 and 0 < y[dec == i].mean() < 1])
        m = ~np.isnan(s) & ~np.isnan(acc)
        r = np.corrcoef(s[m], acc[m])[0, 1]
        print(f"{col:24s} {k:7.4f} {within:13.4f} {r:11.3f}")

    # Onset heterogeneity among clear increases.
    on = d["do_ec_onset"].to_numpy()
    strong = d["do_ec_llr"].to_numpy() > 4
    for lab in (0, 1):
        sel = strong & (y == lab)
        q = np.nanpercentile(on[sel], [10, 25, 50, 75, 90])
        print(f"y={lab}: strong increase share {sel.sum() / (y == lab).sum():.3f}; onset weeks p10/25/50/75/90 {q}")

    # Aligned mean weekly order rate by class.
    rows = d.join(apps.select("application_id", "row"), on="application_id")["row"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for lab, col in ((0, "tab:blue"), (1, "tab:red")):
        ax[0].plot(m_ec[rows[y == lab]].mean(0)[:60], color=col, label=f"y={lab}")
        ax[1].plot(m_pv[rows[y == lab]].mean(0)[:60], color=col, label=f"y={lab}")
        ax[2].hist(on[strong & (y == lab)], bins=TS, density=True, alpha=.5, color=col, label=f"y={lab}")
    ax[0].set_title("completed orders / week (0 = week before decision)")
    ax[1].set_title("previous applications / week")
    ax[2].set_title("estimated onset (weeks), LLR > 4")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(c.OUT / "250_distress_onset.png", dpi=110)
    print("saved", c.OUT / "250_distress_onset.png")


if __name__ == "__main__":
    main()
