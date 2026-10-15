"""Check and screen the native TabPFN-Rel DFS matrix built by 66a (run with the main .venv):
 1) preprocessing check: row alignment and the cutoff, by comparing native DFS counts with our own features
    (installments, orders, previous applications, bureau reports must agree up to the strict-< boundary);
 2) class samples of the strongest native columns, saved for inspection;
 3) every native column against the v6 residual, with a family-wise noise threshold (max over 200 random columns),
    replicated on both halves. If nothing replicates, TabPFN-Rel's representation holds no signal v6 lacks, and its
    only possible contribution is TabPFN's in-context learner, which v6 already runs on the stronger hand view."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from common import OUT, PQ, calibrate, features, ks_nan, resid_test, v6_oof

depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
f = features(); tr = f.filter(pl.col("is_train") == 1); y = tr["target"].to_numpy()
D = pl.read_parquet(PQ / f"rel_dfs_d{depth}.parquet")
names = dict(pl.read_csv(PQ / f"rel_dfs_d{depth}_names.csv").iter_rows())
Dtr = tr.select("application_id").join(D, on="application_id", how="left", maintain_order="left")
assert Dtr.height == tr.height and Dtr["application_id"].to_list() == tr["application_id"].to_list()
print(f"depth {depth}: {D.width - 1} native DFS columns, {D.height} rows")

col = {v: k for k, v in names.items()}
checks = [("d2__customers.COUNT(installments)", "rp_n_inst"), ("d2__customers.COUNT(prev_apps)", "pv_n"),
          ("d2__customers.COUNT(orders)", "ec_n_orders"), ("d2__customers.COUNT(bureau)", "bc_n_reports")]
for dn, ours in checks:
    if dn not in col:
        print(f"  (no {dn})"); continue
    a = Dtr[col[dn]].fill_null(0).to_numpy(); b = tr[ours].fill_null(0).to_numpy()
    print(f"  cutoff check {dn:40s} vs {ours:13s}: equal {np.mean(a == b):.4f}, native>ours {np.mean(a > b):.4f}, native<ours {np.mean(a < b):.4f}")

p = calibrate(y, v6_oof())
rng = np.random.default_rng(66)
half = rng.random(len(y)) < 0.5
def test(x):
    a, b = resid_test(y[half], p[half], x[half]), resid_test(y[~half], p[~half], x[~half])
    return a[0], b[0], a[2], b[2]
null = np.array([test(rng.random(len(y))) for _ in range(200)])
CHI, DKS = null[:, :2].max(), null[:, 2:].max()
print(f"family-wise noise: chi2/df {CHI:.2f}  dKS {DKS:+.4f}")
rows = []
for c in D.columns[1:]:
    x = Dtr[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    if np.nanstd(x) == 0 or np.isnan(x).all():
        continue
    rows.append((c, names[c], ks_nan(y, x), *test(x)))
R = pl.DataFrame(rows, schema=["col", "dfs_name", "ks", "chi_h1", "chi_h2", "dks_h1", "dks_h2"], orient="row").with_columns(
    dks_min=pl.min_horizontal("dks_h1", "dks_h2"),
    replicates=(pl.min_horizontal("chi_h1", "chi_h2") > CHI) & (pl.min_horizontal("dks_h1", "dks_h2") > DKS)).sort("dks_min", descending=True)
R.write_csv(OUT / f"66b_native_dfs_d{depth}_screen.csv")
with pl.Config(tbl_rows=15, float_precision=4, fmt_str_lengths=70):
    print(R.select("dfs_name", "ks", "chi_h1", "chi_h2", "dks_h1", "dks_h2", "replicates").head(15))
    print("strongest native columns by univariate KS:")
    print(R.sort("ks", descending=True).select("dfs_name", "ks").head(8))
print(f"replicated above family-wise noise: {R['replicates'].sum()} of {R.height}")
print(f"best native univariate KS {R['ks'].max():.4f} vs best hand feature (pp_mean etc.) "
      f"{max(ks_nan(y, tr[c].cast(pl.Float64).to_numpy()) for c in ('pp_mean', 'rp_frac_dpd1', 'bc_score_mean')):.4f}")

top = R.sort("ks", descending=True)["col"].head(4).to_list()
samp = pl.concat([tr.select("application_id", "target").with_columns(Dtr.select(top)).filter(pl.col("target") == k).sample(5, seed=1) for k in (0, 1)])
samp.rename({c: names[c][:60] for c in top}).write_csv(OUT / f"66b_class_samples_d{depth}.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(R["dks_h1"], R["dks_h2"], s=8); ax[0].axvline(DKS, color="r", ls="--"); ax[0].axhline(DKS, color="r", ls="--")
ax[0].set_xlabel("dKS vs v6 half 1"); ax[0].set_ylabel("half 2"); ax[0].set_title(f"native TabPFN-Rel DFS (depth {depth}) vs v6")
ax[1].hist(R["ks"], 40); ax[1].set_title("univariate KS of native DFS columns")
plt.tight_layout(); plt.savefig(OUT / f"66b_native_dfs_d{depth}.png", dpi=80)
