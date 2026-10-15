"""Do the historical-loan rows teach the same feature -> risk mapping as the selected loans?

  1) per feature: univariate KS on the selected loans vs on the historical loans, both computed on rows where the
     feature is observed (so coverage differences do not drive the comparison), and the bad-rate-by-decile curves of
     the strongest features in both populations;
  2) transfer test with an additive binned log-odds score (counting only: 10 quantile bins per feature, backfitted
     log(O/E) adjustments with shrinkage, as in eda/71). Fitted on half A of the selected loans
       (a) alone, (b) A + all historical rows, (c) historical rows only,
     and scored on half B (and the mirror). If (b) beats (a), the extra rows carry usable signal for the target.
Historical rows of test customers are included in (b), as they would be in the final fit: they are dated before the
customers' cutoff and carry no competition label."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, features, ks

f = features()
M = f.filter(pl.col("is_train") == 1)
H = pl.read_parquet(PQ / "hist_feats.parquet")
y_m, y_h = M["target"].to_numpy(), H["target"].to_numpy()
num = [c for c, d in M.schema.items() if d.is_numeric() and c not in ("target", "is_train")]
num = [c for c in num if c in H.columns]


def ks_obs(y, x):
    ok = np.isfinite(x)
    return ks(y[ok], x[ok]) if ok.sum() > 500 and 0 < y[ok].mean() < 1 and np.unique(x[ok]).size > 1 else np.nan


rows = []
for c in num:
    xm, xh = M[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float), H[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    rows.append((c, ks_obs(y_m, xm), ks_obs(y_h, xh), np.isfinite(xm).mean(), np.isfinite(xh).mean()))
K = pl.DataFrame(rows, schema=["feature", "ks_main", "ks_hist", "cov_main", "cov_hist"], orient="row").sort("ks_main", descending=True, nulls_last=True)
K.write_csv(OUT / "82_feature_ks.csv")
with pl.Config(tbl_rows=40, float_precision=3):
    print(K.head(40))
fam = K.with_columns(family=pl.col("feature").str.extract(r"^([a-z]+\d?)_", 1)).group_by("family").agg(
    n=pl.len(), ks_main=pl.col("ks_main").max(), ks_hist=pl.col("ks_hist").max()).sort("ks_main", descending=True, nulls_last=True)
with pl.Config(tbl_rows=30, float_precision=3):
    print("strongest feature per family (observed rows):"); print(fam)
k_ok = K.drop_nulls()
print(f"Spearman of per-feature KS, main vs hist: {np.corrcoef(k_ok['ks_main'].rank().to_numpy(), k_ok['ks_hist'].rank().to_numpy())[0, 1]:.3f}")


def edges(x):
    q = np.nanquantile(x, np.linspace(0, 1, 11)[1:-1]) if np.isfinite(x).any() else np.array([])
    return np.unique(q)


def bins(x, e):
    b = np.searchsorted(e, x, side="right"); b[~np.isfinite(x)] = len(e) + 1
    return b


def additive(Xfit, yfit, Xev, cols, rounds=4, shrink=0.5, prior=50.0):
    """Backfitted additive log-odds from quantile bins (counts only). Returns scores for Xev."""
    base = np.log(yfit.mean() / (1 - yfit.mean()))
    E = {c: edges(Xfit[c]) for c in cols}
    Bf = {c: bins(Xfit[c], E[c]) for c in cols}
    Be = {c: bins(Xev[c], E[c]) for c in cols}
    adj = {c: np.zeros(len(E[c]) + 2) for c in cols}
    lp = np.full(len(yfit), base)
    for _ in range(rounds):
        for c in cols:
            lp_c = lp - adj[c][Bf[c]]
            p = 1 / (1 + np.exp(-lp_c))
            o = np.bincount(Bf[c], weights=yfit, minlength=len(adj[c]))
            e = np.bincount(Bf[c], weights=p, minlength=len(adj[c]))
            adj[c] = shrink * np.log((o + prior * yfit.mean()) / (e + prior * yfit.mean()))
            lp = lp_c + adj[c][Bf[c]]
    return base + sum(adj[c][Be[c]] for c in cols)


cols = K.filter(pl.col("ks_main") > 0.05)["feature"].to_list()
print(f"additive transfer test on {len(cols)} features with univariate KS > 0.05 on the selected loans")
arr = lambda d: {c: d[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) for c in cols}  # noqa: E731
XM, XH = arr(M), arr(H)
rng = np.random.default_rng(82)
res = []
for rep in range(3):
    half = rng.random(len(y_m)) < 0.5
    for a, b in ((half, ~half), (~half, half)):
        sub = lambda X, m: {c: X[c][m] for c in cols}  # noqa: E731
        s_a = additive(sub(XM, a), y_m[a], sub(XM, b), cols)
        both = {c: np.r_[XM[c][a], XH[c]] for c in cols}
        s_ab = additive(both, np.r_[y_m[a], y_h], sub(XM, b), cols)
        s_h = additive(XH, y_h, sub(XM, b), cols)
        flag = {c: np.r_[XM[c][a], XH[c]] for c in cols}
        flag["is_hist"] = np.r_[np.zeros(a.sum()), np.ones(len(y_h))]
        res.append((rep, ks(y_m[b], s_a), ks(y_m[b], s_ab), ks(y_m[b], s_h)))
R = pl.DataFrame(res, schema=["rep", "main_only", "main_plus_hist", "hist_only"], orient="row")
print(R)
d = (R["main_plus_hist"] - R["main_only"]).to_numpy()
print(f"held-out KS: main only {R['main_only'].mean():.4f}, main + hist {R['main_plus_hist'].mean():.4f} "
      f"(gain {d.mean():+.4f}, {int((d > 0).sum())}/{len(d)} halves up), hist only {R['hist_only'].mean():.4f}")
R.write_csv(OUT / "82_transfer.csv")

fig, ax = plt.subplots(2, 4, figsize=(18, 8))
ax = ax.ravel()
ax[0].scatter(K["ks_main"], K["ks_hist"], s=8); ax[0].plot([0, 0.4], [0, 0.4], "k--", lw=0.5)
ax[0].set_xlabel("KS on selected loans"); ax[0].set_ylabel("KS on historical loans"); ax[0].set_title("per-feature KS (observed rows)")
for i, c in enumerate(["bc_score_mean", "annual_interest_rate", "rp_frac_dpd1", "pp_mean", "ec_accel_30d", "dti_declared", "age"]):
    for X, yy, lab_ in ((M, y_m, "selected"), (H, y_h, "historical")):
        x = X[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float); ok = np.isfinite(x)
        if ok.sum() < 1000:
            continue
        q = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, 11)))
        bb = np.clip(np.searchsorted(q, x[ok], side="right") - 1, 0, len(q) - 2)
        rate = [yy[ok][bb == k].mean() for k in range(len(q) - 1)]
        ax[i + 1].plot(range(len(rate)), rate, "o-", label=lab_)
    ax[i + 1].set_title(f"bad rate by decile: {c}"); ax[i + 1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "82_transfer_check.png", dpi=80)
print("saved", OUT / "82_transfer_check.png")
