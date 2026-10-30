"""Error analysis of the 8-seed trees+Markov OOF (v72 + v73 kernels): is anything left to fix, or is the rest label noise?

1  per segment: rows, bad rate, mean calibrated prediction, their ratio, KS inside the segment
2  residual learnability: a second LightGBM boosts from init_score = logit(calibrated OOF) with all current features,
   cross-fitted on new folds (seed 99). If it raises KS on folds it never saw, the trees left structure; its gain
   importance says where. If not, the remaining error is noise for these features.
3  confident errors: inside the lowest-scored half and the top 10%, which features still separate bads from goods
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

import common as c
import valid2 as v

R = Path(__file__).resolve().parent.parent / "results"
SIX = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]


def oof8(ids):
    cols = []
    for m in SIX:
        a = pl.read_parquet(R / f"v72k_out/v72/preds/oof_{m}.parquet")
        b = pl.read_parquet(R / f"v73k_out/v72/preds/oof_{m}.parquet")
        assert (a["application_id"] == ids).all() and (b["application_id"] == ids).all()
        cols.append(rankdata((3 * a[m].to_numpy() + 5 * b[m].to_numpy()) / 8))
    return np.mean(cols, 0) / len(ids)


def calibrate(s, y):
    """Cross-fitted isotonic map from blend rank to probability (monotone, so KS is unchanged up to ties)."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=270).split(s, y):
        p[va] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s[tr], y[tr]).predict(s[va])
    return p


def segments(d):
    print("== 1: segments (ratio = mean prediction / bad rate; KS inside the segment)")
    q = lambda k, n=5: pl.col(k).qcut(n, allow_duplicates=True).cast(pl.String)
    segs = {
        "has_history": pl.col("has_history").cast(pl.String), "dec_month": pl.col("dec_month").cast(pl.String),
        "loan_purpose": pl.col("loan_purpose"), "region": pl.col("region"), "employment_type": pl.col("employment_type"),
        "housing_status": pl.col("housing_status"), "loan_amount q5": q("loan_amount"), "age q5": q("age"),
        "interest q5": q("annual_interest_rate"), "dti_total q5": q("dti_total"), "ec_orders_90d q5": q("ec_orders_90d"),
        "income missing": pl.col("declared_monthly_income").is_null().cast(pl.String),
    }
    rows = []
    for name, ex in segs.items():
        for (g,), part in d.with_columns(seg=ex).group_by(["seg"]):
            y, p = part["target"].to_numpy(), part["p"].to_numpy()
            if len(y) < 500 or y.sum() < 30:
                continue
            rows.append(dict(segment=name, value=str(g)[:22], n=len(y), bad=y.mean(), pred=p.mean(), ratio=p.mean() / y.mean(),
                             ks=ks_2samp(p[y == 1], p[y == 0]).statistic))
    t = pl.DataFrame(rows).sort("segment", "value")
    with pl.Config(tbl_rows=200, tbl_width_chars=140, float_precision=4):
        print(t)
    print("largest calibration gaps:")
    with pl.Config(tbl_rows=12, float_precision=4):
        print(t.with_columns(gap=(pl.col("pred") - pl.col("bad")) * pl.col("n").sqrt()).sort(pl.col("gap").abs(), descending=True).head(10))


def residual(X, y, p):
    print("== 2: residual learnability (boost from the calibrated OOF; new folds, seed 99)")
    base = np.log(p / (1 - p))
    out, imp = np.zeros(len(y)), np.zeros(X.shape[1])
    params = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=200, feature_fraction=0.5,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, n_jobs=8)
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=99).split(X, y)):
        fi, st = next(StratifiedKFold(7, shuffle=True, random_state=k).split(tr, y[tr]))
        a, b = tr[fi], tr[st]
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a], init_score=base[a]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[b], y[b], init_score=base[b])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        out[va] = base[va] + m.predict(X.iloc[va], raw_score=True)
        imp += m.feature_importance("gain")
        print(f"   fold {k}: {m.best_iteration:4d} trees, KS ref {v.ks(y[va], base[va]):.5f} -> {v.ks(y[va], out[va]):.5f}"
              f" ({v.ks(y[va], out[va]) - v.ks(y[va], base[va]):+.5f})", flush=True)
    print(f"   pooled KS ref {v.ks(y, base):.5f} -> {v.ks(y, out):.5f}")
    o = np.argsort(-imp)[:15]
    print("   top residual-model gain:", ", ".join(f"{X.columns[i]} {imp[i] / imp.sum():.2f}" for i in o))


def confident(X, y, s):
    print("== 3: features that still separate bads from goods inside score bands (KS between y=1 and y=0)")
    for name, m in (("lowest 50%", s <= np.quantile(s, 0.5)), ("top 10%", s >= np.quantile(s, 0.9))):
        yy, ref = y[m], s[m]
        res = []
        for k in X.columns:
            if str(X[k].dtype) == "category":
                continue
            x = X[k].to_numpy()[m].astype(float)
            f = np.isfinite(x)
            if f.sum() > 300 and yy[f].sum() > 30 and np.unique(x[f]).size > 1:
                res.append((ks_2samp(x[f][yy[f] == 1], x[f][yy[f] == 0]).statistic, k))
        res.sort(reverse=True)
        print(f"   {name}: n {m.sum()}, bad {yy.mean():.4f}, KS of the score itself {v.ks(yy, ref):.4f}")
        print("      " + ", ".join(f"{k} {ks:.3f}" for ks, k in res[:12]))


def main():
    feats = pl.read_parquet(c.PQ / "features.parquet").join(
        pl.read_parquet(c.PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, _, y, ids, _ = v.frames(feats)
    s = oof8(ids)
    p = calibrate(s, y)
    print(f"8-seed OOF KS {v.ks(y, s):.5f}, calibrated {v.ks(y, p):.5f}, mean p {p.mean():.4f} vs bad {y.mean():.4f}")
    d = pl.DataFrame({"application_id": ids, "s": s, "p": p}).join(
        feats.select("application_id", "target", "has_history", "dec_month", "loan_purpose", "region", "employment_type",
                     "housing_status", "loan_amount", "age", "annual_interest_rate", "dti_total", "ec_orders_90d",
                     "declared_monthly_income"), on="application_id", how="left", maintain_order="left")
    assert (d["target"].to_numpy() == y).all()
    segments(d)
    residual(X, y, p)
    confident(X, y, s)


if __name__ == "__main__":
    main()
