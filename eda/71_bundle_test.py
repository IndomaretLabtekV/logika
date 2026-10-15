"""Bundle test: do many individually sub-threshold candidates add up? (v2 gained +0.0046 from 38 small features.)
Pool: 703 log-ratios of money features + 297 native TabPFN-Rel DFS columns (eda/66a) + the 229 v6 features themselves
(the latter tests whether v6 under-fits its own inputs). Procedure, cross-fitted on two halves of the training rows:
  1) rank candidates by residual dKS vs the calibrated v6 blend on the FIT half only, keep the top K;
  2) fit an additive bin-adjustment of v6's log-odds on the FIT half (backfitting, 10 quantile bins per feature,
     log(O/E) per bin with shrinkage), i.e. counting only, no learner;
  3) apply the frozen bins and adjustments to the other half and measure the KS gain there.
The same procedure on pure-noise candidates gives the null distribution. A bundle gain above the null is a feature
set worth adding to the GPU run."""
import itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, calibrate, features, ks, resid_test, v6_oof

f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy(); n = len(y)
p = np.clip(calibrate(y, v6_oof(), bins=100), 1e-4, 1 - 1e-4)
lp0 = np.log(p / (1 - p))
num = [c for c in f.columns if f[c].dtype not in (pl.String, pl.Datetime("us")) and c not in ("target", "is_train")]
M = {c: tr[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) for c in num}
money = [c for c in num if np.nanmin(M[c]) >= 0 and np.nanmax(M[c]) > 1e5]
pool = {f"own:{c}": M[c] for c in num}
pool.update({f"ratio:{a}/{b}": np.log1p(M[a]) - np.log1p(M[b]) for a, b in itertools.combinations(money, 2)})
D = tr.select("application_id").join(pl.read_parquet(PQ / "rel_dfs_d3.parquet"), on="application_id", how="left", maintain_order="left")
pool.update({f"dfs:{c}": D[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) for c in D.columns[1:]})
pool = {k: v for k, v in pool.items() if np.nanstd(v) > 0}
print(f"candidate pool: {len(pool)}")


def edges(x, fit):
    q = np.nanquantile(x[fit], np.linspace(0, 1, 11)[1:-1]) if np.isfinite(x[fit]).any() else np.array([])
    return np.unique(q)


def bins(x, e):
    b = np.searchsorted(e, x, side="right"); b[np.isnan(x)] = len(e) + 1
    return b


def bundle_gain(cands, fit, ev, K, rounds=3, shrink=0.5):
    """Select top-K on fit, backfit additive log(O/E) bin effects on fit, return KS gain on ev."""
    pf = 1 / (1 + np.exp(-lp0))
    score = sorted(cands, key=lambda k: -resid_test(y[fit], pf[fit], cands[k][fit])[2])[:K]
    E = {k: edges(cands[k], fit) for k in score}
    B = {k: bins(cands[k], E[k]) for k in score}
    adj = {k: np.zeros(len(E[k]) + 2) for k in score}
    for _ in range(rounds):
        for k in score:
            other = sum(adj[j][B[j]] for j in score if j != k)
            pp = 1 / (1 + np.exp(-(lp0 + other)))
            o = np.bincount(B[k][fit], weights=y[fit], minlength=len(adj[k]))
            e = np.bincount(B[k][fit], weights=pp[fit], minlength=len(adj[k]))
            adj[k] = shrink * np.log((o + 5) / (e + 5))
    lp = lp0 + sum(adj[k][B[k]] for k in score)
    return ks(y[ev], lp[ev]) - ks(y[ev], lp0[ev]), score


rng = np.random.default_rng(71)
half = rng.random(n) < 0.5
rows = []
for K in (5, 20, 60):
    g = [bundle_gain(pool, a, b, K)[0] for a, b in ((half, ~half), (~half, half))]
    nulls = []
    for r in range(8):
        noise = {f"noise{i}": np.random.default_rng(1000 * r + i).random(n) for i in range(300)}
        nulls.append(np.mean([bundle_gain(noise, a, b, K)[0] for a, b in ((half, ~half), (~half, half))]))
    rows.append((K, np.mean(g), g[0], g[1], np.mean(nulls), np.max(nulls)))
    print(f"K={K:3d}: real bundle held-out gain {np.mean(g):+.4f} (halves {g[0]:+.4f} {g[1]:+.4f}); "
          f"noise bundles mean {np.mean(nulls):+.4f} max {np.max(nulls):+.4f}", flush=True)
_, top = bundle_gain(pool, half, ~half, 20)
print("top-20 chosen on half 1:", top)
R = pl.DataFrame(rows, schema=["K", "gain_mean", "gain_h1", "gain_h2", "null_mean", "null_max"], orient="row")
R.write_csv(OUT / "71_bundle_test.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(R["K"], R["gain_mean"], "o-", label="real candidates"); ax.plot(R["K"], R["null_mean"], "o--", label="noise (mean)")
ax.plot(R["K"], R["null_max"], ":", label="noise (max of 8)"); ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("K features in bundle"); ax.set_ylabel("held-out KS gain vs v6"); ax.legend(); ax.set_title("bundle of sub-threshold candidates")
plt.tight_layout(); plt.savefig(OUT / "71_bundle_test.png", dpi=80)
