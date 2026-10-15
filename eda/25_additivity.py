"""Is risk additive in log-odds across signal families? For each pair of family representatives, a 5x5 quantile grid
(+NaN bin) of bad rates is decomposed (weighted backfitting, pure arithmetic) into row + column effects; the leftover
is the interaction chi2. Also the same grid on v1 residuals (does v1 miss an interaction?). Weak interactions =>
shallow trees / additive models are the right bias; strong ones => explicit cross features."""
import itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, calibrate, features, qbin, v1_oof

f = features()
tr = f.filter(pl.col("is_train") == 1)
y = tr["target"].to_numpy().astype(float)
p = calibrate(y, v1_oof())
reps = ["rp_frac_dpd1", "ec_orders_30d", "bc_score_mean", "annual_interest_rate", "ac_out_in_delta", "rp_cur_dpd_max",
        "dti_declared", "declared_monthly_income", "loan_amount", "pv_rej_rate", "ec_aov", "age", "ac_lowdays_delta",
        "bh_score_delta", "employment_tenure_months"]
B = {c: qbin(tr[c].cast(pl.Float64).to_numpy(), 5) for c in reps}


def interaction_chi2(b1, b2, obs, exp_):
    """2-way additive fit on logit(obs/exp_) per cell, weighted; returns chi2/df of the non-additive part."""
    k1, k2 = b1.max() + 1, b2.max() + 1
    O = np.zeros((k1, k2)); E = np.zeros((k1, k2)); N = np.zeros((k1, k2))
    np.add.at(O, (b1, b2), obs); np.add.at(E, (b1, b2), exp_); np.add.at(N, (b1, b2), 1)
    m = N >= 100
    L = np.log((O + 1) / (N - O + 1)) - np.log((E + 1) / (N - E + 1))   # logit(obs) - logit(expected)
    W = np.where(m, 1 / (1 / (O + 1) + 1 / (N - O + 1)), 0)           # inverse variance of a logit
    r, c = np.zeros(k1), np.zeros(k2)
    for _ in range(50):
        r = (W * (L - c[None, :])).sum(1) / np.maximum(W.sum(1), 1e-9)
        c = (W * (L - r[:, None])).sum(0) / np.maximum(W.sum(0), 1e-9)
    res = (W * (L - r[:, None] - c[None, :]) ** 2).sum()
    df = max((np.count_nonzero(m.any(1)) - 1) * (np.count_nonzero(m.any(0)) - 1), 1)
    return res / df


rows = []
for a, b in itertools.combinations(reps, 2):
    raw = interaction_chi2(B[a], B[b], y, np.full(len(y), y.mean()))
    v1 = interaction_chi2(B[a], B[b], y, p)
    rows.append((a, b, raw, v1))
r = pl.DataFrame(rows, schema=["f1", "f2", "inter_chi2_raw", "inter_chi2_vs_v1"], orient="row")

rng = np.random.default_rng(0)
null = [interaction_chi2(rng.integers(0, 6, len(y)), rng.integers(0, 6, len(y)), y, np.full(len(y), y.mean())) for _ in range(30)]
print(f"null (random bins) chi2/df p95 = {np.percentile(null, 95):.2f}")
with pl.Config(tbl_rows=20, float_precision=3):
    print("\n=== strongest raw interactions (logit scale, chi2/df of non-additive part)")
    print(r.sort("inter_chi2_raw", descending=True).head(15))
    print("\n=== interactions left in v1 residual")
    print(r.sort("inter_chi2_vs_v1", descending=True).head(10))
print(f"\nshare of pairs with raw interaction chi2/df > 2: {(r['inter_chi2_raw'] > 2).mean():.2f}; "
      f"median raw={r['inter_chi2_raw'].median():.2f}, median vs v1={r['inter_chi2_vs_v1'].median():.2f}")

# visual: logit bad rate across bins of rp_frac_dpd1 for each bc_score quintile -> parallel lines = additive
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for i, (a, b) in enumerate([("rp_frac_dpd1", "bc_score_mean"), ("ec_orders_30d", "bc_score_mean"), ("ec_orders_30d", "rp_frac_dpd1")]):
    for k in range(6):
        m = B[b] == k
        if m.sum() < 500:
            continue
        xs, ls = [], []
        for j in range(6):
            mm = m & (B[a] == j)
            if mm.sum() >= 100:
                pr = (y[mm].sum() + 1) / (mm.sum() + 2)
                xs.append(j); ls.append(np.log(pr / (1 - pr)))
        ax[i].plot(xs, ls, "o-", label=f"{b} q{k}" if k < 5 else f"{b} NaN")
    ax[i].set_xlabel(f"{a} quintile (5=NaN)"); ax[i].set_ylabel("logit bad rate"); ax[i].legend(fontsize=6)
    ax[i].set_title("parallel lines = additive in log-odds")
plt.tight_layout(); plt.savefig(OUT / "25_additivity.png", dpi=80)
