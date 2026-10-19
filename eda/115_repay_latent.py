"""Customer repayment-discipline latent from a per-installment model, tested against the clean v31 OOF.
Installments due <= cutoff. Fixed effects (logistic, fitted on train customers' installments only; no target used):
late (max_dpd > 0) ~ installment index in loan, loan rate, installment / declared income, loan term, due year-month.
Customer latent = one-step empirical-Bayes random intercept: sum(y - p) / (sum p(1-p) + tau), plus a recency-weighted
version (half-life 180 days) and the same for severity (max_dpd >= 30 among all installments).
Residual tests as eda/103/114 (O/E chi2/df per safe bin, cross-fit dKS, permuted null).
Usage: cd eda && ../.venv/bin/python 115_repay_latent.py"""
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

import valid2 as v
from common import OUT, ROOT, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins

R = ROOT / "results"
f = features()
cut = f.select("customer_id", "is_train", "declared_monthly_income", cutoff="decision_datetime")
prev = load("previous_applications").select("loan_id", "term_months", "annual_interest_rate")
r = (load("repayment_history").join(cut, on="customer_id").filter(pl.col("due_date") <= pl.col("cutoff"))
     .join(prev, on="loan_id", how="left").sort("customer_id", "loan_id", "due_date")
     .with_columns(k=pl.col("due_date").rank("ordinal").over("loan_id"),
                   age_d=(pl.col("cutoff") - pl.col("due_date")).dt.total_seconds() / 86400,
                   burden=(pl.col("amount_due") / pl.col("declared_monthly_income")).log(),
                   ym=pl.col("due_date").dt.year() * 12 + pl.col("due_date").dt.month(),
                   late=(pl.col("max_days_past_due") > 0).cast(pl.Int8),
                   sev=(pl.col("max_days_past_due") >= 30).cast(pl.Int8)))
print(f"installments {r.height}, late rate {r['late'].mean():.4f}, sev rate {r['sev'].mean():.4f}")


def design(d):
    k = d["k"].to_numpy().clip(max=18)
    ym = d["ym"].to_numpy()
    cols = [np.eye(19)[k][:, 2:], d["annual_interest_rate"].fill_null(0.24).to_numpy()[:, None],
            d["burden"].fill_nan(None).fill_null(-3).to_numpy()[:, None], (d["term_months"].fill_null(12).to_numpy()[:, None] / 12),
            np.eye(ym.max() - ym.min() + 1)[ym - ym.min()][:, 1:]]
    return np.hstack(cols).astype(np.float32)


Z = design(r)
trn = r["is_train"].to_numpy() == 1
out = r.select("customer_id")
for tgt in ("late", "sev"):
    yy = r[tgt].to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[trn], yy[trn])
    p = lr.predict_proba(Z)[:, 1]
    print(f"{tgt}: fixed-effect AUC on train installments {np.round(__import__('sklearn.metrics', fromlist=['x']).roc_auc_score(yy[trn], p[trn]), 4)}")
    w = 0.5 ** (r["age_d"].to_numpy() / 180)
    out = out.with_columns(**{f"{tgt}_res": yy - p, f"{tgt}_info": p * (1 - p), f"{tgt}_wres": w * (yy - p),
                              f"{tgt}_winfo": w * p * (1 - p)})
agg = out.group_by("customer_id").agg(pl.all().sum())
cand = f.select("customer_id", "application_id", "is_train").join(agg, on="customer_id", how="left")
new = {}
for tgt in ("late", "sev"):
    for tau in (1.0, 5.0):
        new[f"re_{tgt}_t{tau:g}"] = pl.col(f"{tgt}_res") / (pl.col(f"{tgt}_info") + tau)
        new[f"re_{tgt}_w_t{tau:g}"] = pl.col(f"{tgt}_wres") / (pl.col(f"{tgt}_winfo") + tau)
cand = cand.with_columns(**new).select("application_id", "is_train", *new)
cand.drop("is_train").write_parquet(OUT / "115_repay_latent.parquet")
tr = cand.filter(pl.col("is_train") == 1).sort("application_id")

X, Xt, y, ids, tids = v.frames(f)
assert (tr["application_id"] == ids).all()


def load_oof(path, col):
    return pl.read_parquet(path).to_pandas().set_index("application_id")[col].loc[ids].to_numpy()


trees = np.mean([rankdata(load_oof(R / f"v22/v22/gbdt/oof_{m}.parquet", m)) for m in
                 ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
ref = (rankdata(trees) + rankdata(load_oof(R / "v22/v22/fm/oof_causilo.parquet", "causilo"))
       + rankdata(load_oof(R / "v28m/oof_kumo_large_c56000.parquet", "kumo_large_c56000"))) / 3
pc = calibrate(y, ref)
rng = np.random.default_rng(0)
null = [(oe_test(y, pc, safe_bins(rng.permutation(X["rp_frac_dpd1"].to_numpy()), 10))[0],
         crossfit_dks(y, pc, safe_bins(rng.permutation(X["rp_frac_dpd1"].to_numpy()), 10))) for _ in range(30)]
c95, d95 = np.quantile([a for a, _ in null], .95), np.quantile([b for _, b in null], .95)
print(f"null (permuted feature): chi2/df p95 {c95:.2f}, dKS p95 {d95:+.5f}")
rows = []
for col in [*new, "rp_frac_dpd1", "rp_frac_dpd30"]:
    x = (tr[col] if col in tr.columns else pl.Series(X[col].to_numpy())).cast(pl.Float64).to_numpy()
    bb = safe_bins(x, 10)
    chi, tab = oe_test(y, pc, bb)
    rows.append(dict(feature=col, ks_uni=v.ks_nan(y, x), chi2=chi, dks=crossfit_dks(y, pc, bb),
                     rho_frac=float(pl.DataFrame({"a": x, "b": X["rp_frac_dpd1"].to_numpy()}).drop_nans().drop_nulls()
                                    .select(pl.corr("a", "b", method="spearman")).item()),
                     oe=" ".join(f"{t[4]:.2f}" for t in tab)))
with pl.Config(tbl_rows=40, tbl_cols=10, fmt_str_lengths=80, tbl_width_chars=200):
    print(pl.DataFrame(rows).sort("chi2", descending=True))
pl.DataFrame(rows).write_csv(OUT / "115_repay_latent.csv")
