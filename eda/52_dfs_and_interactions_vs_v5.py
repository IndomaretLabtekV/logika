"""After v5: (a) does any of the 301 generic DFS columns still explain the v5 residual? (b) is the price premium
additive with the other families in log-odds (parallel lines), and does v5 leave an interaction behind?"""
import itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, calibrate, features, ks_nan, qbin, resid_test, v5_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy().astype(float)
p = calibrate(y, v5_oof())
rng = np.random.default_rng(13)
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(30)])
CHI, DKS = np.percentile(null[:, 0], 95), np.percentile(null[:, 2], 95)
D = tr.select("application_id").join(pl.read_parquet(PQ / "dfs.parquet"), on="application_id", how="left", maintain_order="left")
num = [c for c in D.columns if D[c].dtype.is_numeric()]
rows = [(c, ks_nan(y, D[c].cast(pl.Float64).to_numpy()), *resid_test(y, p, D[c].cast(pl.Float64).to_numpy())) for c in num]
R = pl.DataFrame(rows, schema=["feature", "ks", "chi2_df", "lift", "dKS_cf"], orient="row").sort("chi2_df", descending=True)
print(f"(a) DFS vs v5 residual (noise chi2 p95={CHI:.2f}, dKS p95={DKS:+.4f}); above noise on both: "
      f"{((R['chi2_df'] > CHI) & (R['dKS_cf'] > DKS)).sum()} of {len(num)}")
with pl.Config(tbl_rows=8, float_precision=4):
    print(R.head(8))

def inter(b1, b2, obs, exp_):
    k1, k2 = b1.max() + 1, b2.max() + 1
    O = np.zeros((k1, k2)); E = np.zeros((k1, k2)); N = np.zeros((k1, k2))
    np.add.at(O, (b1, b2), obs); np.add.at(E, (b1, b2), exp_); np.add.at(N, (b1, b2), 1)
    m = N >= 100
    L = np.log((O + 1) / (N - O + 1)) - np.log((E + 1) / (N - E + 1))
    W = np.where(m, 1 / (1 / (O + 1) + 1 / (N - O + 1)), 0)
    r, c = np.zeros(k1), np.zeros(k2)
    for _ in range(50):
        r = (W * (L - c[None, :])).sum(1) / np.maximum(W.sum(1), 1e-9)
        c = (W * (L - r[:, None])).sum(0) / np.maximum(W.sum(0), 1e-9)
    return (W * (L - r[:, None] - c[None, :]) ** 2).sum() / max((np.count_nonzero(m.any(1)) - 1) * (np.count_nonzero(m.any(0)) - 1), 1)

reps = ["pp_mean", "pp_90", "rp_frac_dpd1", "ec_orders_30d", "bc_score_mean", "annual_interest_rate", "ac_out_in_delta",
        "sv_maxdpd_mean_late", "declared_monthly_income", "dec_month"]
B = {c: qbin(tr[c].cast(pl.Float64).to_numpy(), 5) for c in reps}
nul = [inter(rng.integers(0, 6, len(y)), rng.integers(0, 6, len(y)), y, p) for _ in range(30)]
rows = [(a, b, inter(B[a], B[b], y, np.full(len(y), y.mean())), inter(B[a], B[b], y, p)) for a, b in itertools.combinations(reps, 2) if "pp_" in a + b]
I = pl.DataFrame(rows, schema=["f1", "f2", "raw_interaction", "left_by_v5"], orient="row").sort("left_by_v5", descending=True)
print(f"\n(b) price-premium interactions (noise p95 {np.percentile(nul, 95):.2f})")
with pl.Config(tbl_rows=20, float_precision=3):
    print(I)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for k in range(6):
    m = B["rp_frac_dpd1"] == k
    xs, ls = [], []
    for j in range(5):
        mm = m & (B["pp_mean"] == j)
        if mm.sum() >= 100:
            pr = (y[mm].sum() + 1) / (mm.sum() + 2); xs.append(j); ls.append(np.log(pr / (1 - pr)))
    ax[0].plot(xs, ls, "o-", label=f"rp_frac_dpd1 q{k}" if k < 5 else "no repayment history")
ax[0].set_xlabel("price premium quintile"); ax[0].set_ylabel("logit bad rate"); ax[0].legend(fontsize=7); ax[0].set_title("premium x repayment")
ax[1].scatter(R["chi2_df"], R["dKS_cf"], s=8); ax[1].axvline(CHI, color="r", ls="--"); ax[1].axhline(DKS, color="r", ls="--")
ax[1].set_xlabel("chi2/df vs v5"); ax[1].set_ylabel("dKS vs v5"); ax[1].set_title("301 DFS columns vs the v5 residual")
plt.tight_layout(); plt.savefig(OUT / "52_dfs_interactions_vs_v5.png", dpi=80)
