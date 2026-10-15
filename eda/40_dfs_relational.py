"""TabPFN-Rel recipe, feature half: build the DFS view (eda/dfs.py) and check it like any preprocessing.
(a) shape, coverage, inf; (b) independent pandas recomputation on random customers; (c) future-row injection on all
dated tables; (d) train->test drift; (e) how much of the hand-made signal DFS recovers, family by family;
(f) residual test against the v3 blend: does generic DFS know anything v3 does not?"""
import time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from common import OUT, PQ, ROOT, TABLES, calibrate, features, ks, ks_nan, load, psi, resid_test
from dfs import build_dfs

T = {t: load(t) for t in TABLES}
t0 = time.time()
D = build_dfs(T)
print(f"(a) DFS view {D.shape} built in {time.time() - t0:.1f}s; one row per application: {D['application_id'].n_unique() == D.height}")
D.write_parquet(PQ / "dfs.parquet")
CAT = ["loan_purpose", "region", "employment_type", "housing_status"]
num = [c for c in D.columns if c not in ("application_id", "customer_id", *CAT)]
inf = [c for c in num if D[c].dtype.is_float() and D[c].is_infinite().any()]
allnull = [c for c in num if D[c].null_count() == D.height]
const = [c for c in num if D[c].drop_nulls().n_unique() <= 1]
print(f"    {len(num)} numeric + {len(CAT)} categorical; inf columns {inf}; all-null {allnull}; constant {const}")

# (b) independent recomputation
f = features()
meta = f.select("application_id", "customer_id", "decision_datetime", "is_train", "target", "has_history")
fp = D.join(meta, on=["application_id", "customer_id"]).to_pandas().set_index("customer_id")
rng = np.random.default_rng(3)
cust = rng.choice(fp.index[fp["has_history"] == 1], 200, replace=False)
pd_t = {t: T[t].filter(pl.col("customer_id").is_in(list(cust))).to_pandas() for t in
        ["repayment_history", "ecommerce_transactions", "bureau_observations", "previous_applications", "account_behavior"]}
ok = {}
def chk(name, exp):
    ok.setdefault(name, []).append(bool(np.isclose(exp, fp.loc[c, name], equal_nan=True, rtol=1e-9)))
for c in cust:
    cut = fp.loc[c, "decision_datetime"]
    r = pd_t["repayment_history"]; r = r[(r.customer_id == c) & (r.due_date < cut)]
    chk("dfs_inst__max_days_past_due__mean", r.max_days_past_due.mean() if len(r) else np.nan)
    chk("dfs_loan__dpd_max__mean", r.groupby("loan_id").max_days_past_due.max().mean() if len(r) else np.nan)
    e = pd_t["ecommerce_transactions"]; e = e[(e.customer_id == c) & (e.transaction_datetime < cut)]
    o = e.drop_duplicates("order_id")
    chk("dfs_order__count", len(o) if len(o) else np.nan)
    chk("dfs_order__order_total__sum", o.order_total.sum() if len(o) else np.nan)
    chk("dfs_item__product_category__share_health", (e.product_category == "health").mean() if len(e) else np.nan)
    b = pd_t["bureau_observations"]; b = b[(b.customer_id == c) & (b.available_datetime < cut)]
    chk("dfs_bureau__score__max", b.score.max() if len(b) else np.nan)
    chk("dfs_pull__score_mean__std", b.groupby("application_id").score.mean().std() if b.application_id.nunique() > 1 else np.nan)
    p = pd_t["previous_applications"]; p = p[(p.customer_id == c) & (p.decision_datetime < cut)].sort_values("decision_datetime")
    chk("dfs_prev__decision__share_rejected", (p.decision == "rejected").mean() if len(p) else np.nan)
    chk("dfs_prev__requested_amount__last", p.requested_amount.iloc[-1] if len(p) else np.nan)
    a = pd_t["account_behavior"]; a = a[(a.customer_id == c) & (a.month.apply(lambda m: pd.Timestamp(m) + pd.DateOffset(months=1)) < cut)]
    chk("dfs_acct__low_balance_days__sum", a.low_balance_days.sum() if len(a) else np.nan)
print("(b) independent recomputation on 200 customers with history:")
for k, v in ok.items():
    print(f"    {k:<45} match {np.mean(v):.3f}")
    assert np.mean(v) == 1.0, k

# (c) future-row injection
apps = pl.concat([T["train_applications"], T["test_applications"]], how="diagonal_relaxed").select("customer_id", cut="decision_datetime")
fut = apps.sample(2000, seed=1)
first = lambda t: T[t].join(fut, on="customer_id").group_by("customer_id").first()  # noqa: E731
d5 = pl.duration(days=5)
inj = {
    "repayment_history": first("repayment_history").with_columns(due_date=pl.col("cut") + d5, max_days_past_due=pl.lit(999), loan_id=pl.lit("L_FAKE")),
    "ecommerce_transactions": first("ecommerce_transactions").with_columns(transaction_datetime=pl.col("cut") + pl.duration(hours=1),
                                                                           order_id=pl.lit("O_FAKE") + pl.col("customer_id"), order_total=pl.lit(10**9)),
    "bureau_observations": first("bureau_observations").with_columns(available_datetime=pl.col("cut") + d5, score=pl.lit(300.0),
                                                                       application_id=pl.lit("A_FAKE")),
    "account_behavior": first("account_behavior").with_columns(month=pl.col("cut").dt.truncate("1mo").cast(pl.Date), outflows=pl.lit(10**10)),
    "previous_applications": first("previous_applications").with_columns(
        application_datetime=pl.col("cut") - pl.duration(hours=1), decision_datetime=pl.col("cut") + d5, decision=pl.lit("rejected"),
        requested_amount=pl.lit(10**9), application_id=pl.lit("A_FAKE") + pl.col("customer_id"), loan_id=pl.lit(None, pl.String)),
}
T2 = dict(T)
for t, extra in inj.items():
    T2[t] = pl.concat([T[t], extra.select(T[t].columns)], how="vertical_relaxed")
D2 = build_dfs(T2).sort("application_id"); D1 = D.sort("application_id")
moved = [c for c in num if not np.allclose(D1[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(),
                                           D2[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(), rtol=1e-9, atol=1e-9)]
print(f"(c) features moved by injected future rows: {moved or 'none'} (of {len(num)})")
assert not moved

# (d)-(f): drift, recovered signal, residual vs v3
X = meta.join(D, on=["application_id", "customer_id"], how="left", maintain_order="left")
tr, te = X.filter(pl.col("is_train") == 1), X.filter(pl.col("is_train") == 0)
assert (tr["application_id"] == f.filter(pl.col("is_train") == 1)["application_id"]).all()
y = tr["target"].to_numpy()
W = {"lgb": .27, "lgb_s": .10, "xgb": .03, "xgb_s": .10, "cat": .07, "tabm": .43}   # chosen v3 weights
s3 = sum(w * rankdata(np.load(ROOT / "results" / "v3" / f"oof_{n}.npy")) for n, w in W.items()) / 70000
print(f"v3 chosen-blend OOF KS reproduced locally: {ks(y, s3):.4f} (notebook 0.5142)")
p = calibrate(y, s3)
hand = [c for c in f.columns if c not in ("application_id", "customer_id", "decision_datetime", "target", "is_train") and f[c].dtype != pl.String]
sub = rng.choice(len(y), 15000, replace=False)
H = f.filter(pl.col("is_train") == 1)[sub].select(hand).with_columns(pl.all().cast(pl.Float64).fill_nan(None).rank()).to_numpy()
null = np.array([resid_test(y, p, rng.random(len(y))) for _ in range(20)])
rows = []
for c in num:
    x = tr[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    xt = te[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    xs = pl.Series(x[sub]).rank().to_numpy()
    cors = []
    for j in range(H.shape[1]):
        m = ~np.isnan(xs) & ~np.isnan(H[:, j])
        if m.sum() > 200 and np.std(xs[m]) > 0 and np.std(H[m, j]) > 0:
            cors.append(abs(np.corrcoef(xs[m], H[m, j])[0, 1]))
    chi, lift, dks = resid_test(y, p, x)
    rows.append((c, c.split("__")[0], ks_nan(y, x), psi(x, xt), max(cors) if cors else 0.0, chi, dks))
R = pl.DataFrame(rows, schema=["feature", "family", "ks", "psi", "max_rho_hand", "chi2_df", "dKS_cf"], orient="row")
R.write_csv(OUT / "40_dfs_screen.csv")
print(f"(d) max PSI train->test over DFS view: {R['psi'].max():.5f}")
print(f"(e) DFS view: {(R['max_rho_hand'] > 0.95).mean():.0%} of features are near-copies (|rho|>0.95) of a hand-made feature")
with pl.Config(tbl_rows=20, float_precision=4, fmt_str_lengths=50):
    print("    best univariate KS per DFS family vs best hand-made in the same table:")
    print(R.group_by("family").agg(n=pl.len(), best_ks=pl.col("ks").max(), best=pl.col("feature").sort_by("ks").last()).sort("best_ks", descending=True))
    print(f"(f) residual vs v3 blend (noise: chi2/df p95={np.percentile(null[:, 0], 95):.2f}, dKS p95={np.percentile(null[:, 2], 95):+.4f})")
    print(R.sort("chi2_df", descending=True).head(15))
print(f"    DFS features above noise on both chi2 and dKS: "
      f"{((R['chi2_df'] > np.percentile(null[:, 0], 95)) & (R['dKS_cf'] > np.percentile(null[:, 2], 95))).sum()} of {len(num)}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].scatter(R["max_rho_hand"], R["ks"], s=8); ax[0].set_xlabel("max |Spearman| with a hand-made feature"); ax[0].set_ylabel("univariate KS")
ax[0].set_title("DFS features: signal vs redundancy")
ax[1].scatter(R["chi2_df"], R["dKS_cf"], s=8); ax[1].axvline(np.percentile(null[:, 0], 95), color="r", ls="--"); ax[1].axhline(np.percentile(null[:, 2], 95), color="r", ls="--")
ax[1].set_xlabel("residual chi2/df vs v3"); ax[1].set_ylabel("cross-fit dKS"); ax[1].set_title("what DFS knows beyond v3 (red = noise p95)")
fam = R.group_by("family").agg(pl.col("ks").max()).sort("ks")
ax[2].barh(fam["family"], fam["ks"]); ax[2].set_title("best univariate KS per DFS family")
plt.tight_layout(); plt.savefig(OUT / "40_dfs.png", dpi=80)
