from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PQ = ROOT / "data_pq"
OUT = ROOT / "eda" / "out"
OUT.mkdir(parents=True, exist_ok=True)
PQ.mkdir(exist_ok=True)

TABLES = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
          "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]


def load(name: str) -> pl.DataFrame:
    p = PQ / f"{name}.parquet"
    if not p.exists():
        pl.read_csv(DATA / f"{name}.csv", try_parse_dates=True, infer_schema_length=100000).write_parquet(p)
    return pl.read_parquet(p)


def ks(y, s):
    """KS exactly as the competition scorer (ks_2samp on bad vs good)."""
    import numpy as np
    from scipy.stats import ks_2samp
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if y.min() == y.max():
        return 0.0
    return float(ks_2samp(s[y == 1], s[y == 0]).statistic)


def ks_nan(y, s):
    """Univariate KS where NaN is its own bin (placed at either extreme, best taken)."""
    import numpy as np
    s = np.asarray(s, dtype=float)
    if np.isnan(s).all():
        return 0.0
    lo, hi = np.nanmin(s) - 1, np.nanmax(s) + 1
    return max(ks(y, np.where(np.isnan(s), lo, s)), ks(y, np.where(np.isnan(s), hi, s)))


def psi(a, b, bins=10):
    """Population stability index of b vs a using a's quantile bins (NaN = own bin)."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    qs = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1))) if (~np.isnan(a)).any() else np.array([0.0])
    def dist(x):
        idx = np.where(np.isnan(x), -1, np.clip(np.searchsorted(qs[1:-1], x, side="right"), 0, None))
        cnt = np.array([(idx == k).sum() for k in range(-1, len(qs))], float) + 0.5
        return cnt / cnt.sum()
    pa, pb = dist(a), dist(b)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def features():
    p = PQ / "features.parquet"
    if not p.exists():
        from features import build_all
        build_all({t: load(t) for t in TABLES}).write_parquet(p)
    return pl.read_parquet(p)


def analyze(df: pl.DataFrame, cols, tag: str, top_plot: int = 12):
    """Prints KS / PSI / coverage table and saves decile bad-rate plots for the top features."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr, te = df.filter(pl.col("is_train") == 1), df.filter(pl.col("is_train") == 0)
    y = tr["target"].to_numpy()
    rows = []
    for c in cols:
        if tr[c].dtype == pl.String:
            rate = tr.group_by(c).agg(pl.col("target").mean()).rename({"target": "r"})
            s_tr = tr.join(rate, on=c, how="left")["r"].to_numpy()
            s_te = te.join(rate, on=c, how="left")["r"].to_numpy()
        else:
            s_tr, s_te = tr[c].cast(pl.Float64).to_numpy(), te[c].cast(pl.Float64).to_numpy()
        rows.append((c, ks_nan(y, s_tr), psi(s_tr, s_te), float(np.mean(~np.isnan(s_tr))),
                     float(np.mean(~np.isnan(s_te))), ks(np.r_[np.zeros(len(s_tr)), np.ones(len(s_te))],
                                                          np.r_[np.nan_to_num(s_tr, nan=-1e18), np.nan_to_num(s_te, nan=-1e18)])))
    res = pl.DataFrame(rows, schema=["feature", "ks_target", "psi_test", "cov_train", "cov_test", "ks_train_vs_test"],
                       orient="row").sort("ks_target", descending=True)
    with pl.Config(tbl_rows=200, fmt_str_lengths=40, float_precision=4):
        print(f"\n=== {tag}: univariate KS vs target, drift train->test")
        print(res)
    res.write_csv(OUT / f"{tag}_ks.csv")
    top = res["feature"].head(top_plot).to_list()
    n = len(top); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), top):
        s = tr.select(c, "target")
        if s[c].dtype == pl.String:
            g = s.group_by(c).agg(r=pl.col("target").mean(), n=pl.len()).sort(c)
            ax.bar(g[c].cast(str), g["r"]); ax.tick_params(axis="x", rotation=45)
        else:
            s = s.with_columns(b=pl.when(pl.col(c).is_null() | pl.col(c).is_nan()).then(-1)
                               .otherwise(pl.col(c).rank("ordinal") * 10 // (pl.col(c).is_not_nan().sum() + 1)))
            g = s.group_by("b").agg(r=pl.col("target").mean(), n=pl.len()).sort("b")
            ax.bar(g["b"].cast(str).str.replace("-1", "NA"), g["r"], color=["grey" if v == -1 else "C0" for v in g["b"]])
        ax.axhline(y.mean(), color="r", ls="--", lw=1)
        ax.set_title(f"{c}\nKS={res.filter(pl.col('feature') == c)['ks_target'][0]:.3f}", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{tag}: bad rate by decile (NA=grey, red=base rate)")
    plt.tight_layout(); plt.savefig(OUT / f"{tag}_deciles.png", dpi=80); plt.close()
    return res


# ---- residual analysis against the v1 model (results/v1 OOF; no model is trained here) ----
def v1_oof():
    """Rank-average of v1 GBDT OOF predictions, aligned to the train order of features()."""
    import numpy as np
    from scipy.stats import rankdata
    s = np.mean([rankdata(np.load(ROOT / "results" / "v1" / f"oof_{n}.npy")) for n in ("lgb", "xgb", "cat")], 0) / 70000
    # align by application_id: the row order of a rebuilt feature matrix is not guaranteed to match the v1 run
    v1 = pl.read_parquet(PQ / "v1_train_order.parquet").with_columns(s=pl.Series(s))
    cur = features().filter(pl.col("is_train") == 1).select("application_id")
    return cur.join(v1, on="application_id", how="left", maintain_order="left")["s"].to_numpy()


def calibrate(y, s, bins=50):
    """Score -> probability via bad rate in quantile bins of s (monotone enough for residual tests)."""
    import numpy as np
    b = np.minimum((np.argsort(np.argsort(s)) * bins) // len(s), bins - 1)
    rate = np.bincount(b, weights=y, minlength=bins) / np.bincount(b, minlength=bins)
    return rate[b]


def qbin(x, bins=10):
    """Quantile bin id (0..bins-1); NaN -> bins (own bin). Ties stay together."""
    import numpy as np
    x = np.asarray(x, float)
    out = np.full(len(x), bins)
    m = ~np.isnan(x)
    if m.sum() == 0:
        return out
    qs = np.unique(np.quantile(x[m], np.linspace(0, 1, bins + 1))[1:-1])
    out[m] = np.searchsorted(qs, x[m], side="right")
    return out


def resid_test(y, p, x, bins=10):
    """Does x explain what p (calibrated v1 OOF) missed? Returns (chi2/df, max |O/E-1| over bins >=500 rows, cross-fit dKS).
    chi2 uses binomial variance per bin; dKS = KS gain of logit(p)+log(O/E)_bin with bin effects fit on the other half."""
    import numpy as np
    b = qbin(x, bins)
    o = np.bincount(b, weights=y, minlength=bins + 1)
    e = np.bincount(b, weights=p, minlength=bins + 1)
    v = np.bincount(b, weights=p * (1 - p), minlength=bins + 1)
    n = np.bincount(b, minlength=bins + 1)
    k = n > 0
    chi = float((((o - e) ** 2)[k] / np.maximum(v[k], 1e-9)).sum() / max(k.sum() - 1, 1))
    big = n >= 500
    lift = float(np.abs(o[big] / np.maximum(e[big], 1e-9) - 1).max()) if big.any() else 0.0
    rng = np.random.default_rng(0)
    half = rng.random(len(y)) < 0.5
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    adj = np.zeros(len(y))
    for fit in (half, ~half):
        of = np.bincount(b[fit], weights=y[fit], minlength=bins + 1) + 5
        ef = np.bincount(b[fit], weights=p[fit], minlength=bins + 1) + 5
        adj[~fit] = np.log(of / ef)[b[~fit]]
    return chi, lift, ks(y, lp + adj) - ks(y, lp)


def screen(cands: pl.DataFrame, tag: str, show: int = 40):
    """Screen candidate features (customer_id + columns) against v1: univariate KS, residual chi2/dKS vs v1 OOF,
    PSI train->test, and max |Spearman| with any v1 feature (redundancy). Writes out/{tag}_screen.csv."""
    import numpy as np
    f = features()
    tr = f.filter(pl.col("is_train") == 1)
    te = f.filter(pl.col("is_train") == 0)
    y = tr["target"].to_numpy()
    p = calibrate(y, v1_oof())
    ctr = tr.select("customer_id").join(cands, on="customer_id", how="left", maintain_order="left")
    cte = te.select("customer_id").join(cands, on="customer_id", how="left", maintain_order="left")
    ID = {"application_id", "customer_id", "decision_datetime", "target", "is_train"}
    v1 = [c for c in f.columns if c not in ID and f[c].dtype != pl.String]
    sub = np.random.default_rng(0).choice(len(y), 15000, replace=False)
    V = tr[sub].select(v1).with_columns(pl.all().cast(pl.Float64).rank()).to_numpy()
    rows = []
    for c in cands.columns:
        if c == "customer_id":
            continue
        x = ctr[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
        xt = cte[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
        chi, lift, dks = resid_test(y, p, x)
        xs = pl.Series(x[sub]).rank().to_numpy()
        m = ~np.isnan(xs)
        red = 0.0
        if m.sum() > 100:
            Vm = V[m]; xm = xs[m]
            ok = ~np.isnan(Vm)
            # pairwise-complete spearman via ranks (approx: drop rows with NaN per column)
            cors = []
            for j in range(V.shape[1]):
                k = ok[:, j]
                if k.sum() > 100 and np.std(Vm[k, j]) > 0 and np.std(xm[k]) > 0:
                    cors.append(abs(np.corrcoef(xm[k], Vm[k, j])[0, 1]))
            red = max(cors) if cors else 0.0
        rows.append((c, ks_nan(y, x), chi, dks, float(np.mean(~np.isnan(x))), psi(x, xt), red))
    r = pl.DataFrame(rows, schema=["feature", "ks", "chi2_df", "dKS_cf", "cov", "psi", "max_rho_v1"], orient="row")
    r = r.sort("chi2_df", descending=True)
    with pl.Config(tbl_rows=show, float_precision=4, fmt_str_lengths=45):
        print(f"\n=== {tag}: candidates vs v1 residual (noise: chi2/df p95~1.2, dKS p95~+0.002)")
        print(r.head(show))
    r.write_csv(OUT / f"{tag}_screen.csv")
    return r


def v2_oof():
    """Equal rank-blend of the six v2 GBDT OOF predictions (results/v2). The v2 run used the same build_all code
    as the local features(), and the row order was verified (lgb OOF KS 0.5118 matches the notebook)."""
    import numpy as np
    from scipy.stats import rankdata
    R = ROOT / "results" / "v2"
    return np.mean([rankdata(np.load(R / f"oof_{n}.npy")) for n in ("lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s")], 0) / 70000


def v5_oof():
    """The submitted v5 blend (weights [4,0,0,1,9,3,0,13]/30 reconstructed from the run) as OOF ranks in features() order."""
    import numpy as np
    from scipy.stats import rankdata
    R = ROOT / "results" / "v5"
    w = {"lgb": 4, "xgb_s": 1, "cat": 9, "cat_s": 3, "tabm": 13}
    return sum(v * rankdata(np.load(R / f"oof_{n}.npy")) for n, v in w.items()) / 30 / 70000


def v6_oof():
    """The submitted v6 blend (greedy weights [3,1,0,0,2,1,6,10,7]/30) as OOF ranks in features() order."""
    import numpy as np
    from scipy.stats import rankdata
    R = ROOT / "results" / "v6"
    w = {"lgb": 3, "lgb_s": 1, "cat": 2, "cat_s": 1, "tabpfn": 6, "tabm": 10, "tabm_p": 7}
    return sum(v * rankdata(np.load(R / f"oof_{n}.npy")) for n, v in w.items()) / 30 / 70000
