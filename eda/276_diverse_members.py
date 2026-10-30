"""Diversity members for the trees+Markov ensemble (270 showed no learnable leftover, so what is left is variance).
Each member runs on features + f251 with the seed-42 outer folds, then is judged as (a) a single model and (b) an extra
rank member next to the 8-seed six-model blend (paired bootstrap of the KS change).

dart     LightGBM DART, fixed rounds (DART has no reliable early stopping)
mono     v22 LightGBM with monotone constraints on features whose train Spearman with the target exceeds 0.05
rf       sklearn RandomForest (native NaN), categoricals ordinal-coded
et       sklearn ExtraTrees (native NaN)
seg      two v22 LightGBMs, one per has_history segment; their logloss probabilities are merged as they are
Usage: cd eda && ../.venv/bin/python 276_diverse_members.py member [threads]
"""
import importlib
import math
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v73/diverse"
ea = importlib.import_module("270_error_analysis")
bs = importlib.import_module("275_blend_scheme")


def numeric(X):
    Z = X.copy()
    for k in Z.columns:
        if str(Z[k].dtype) == "category":
            Z[k] = Z[k].cat.codes.replace(-1, np.nan).astype(float)
    return Z


def dart_fit(th):
    p = dict(v.LGB, boosting="dart", learning_rate=0.05, drop_rate=0.1, skip_drop=0.5, n_jobs=th)

    def fit(X, y, tr, fi, st, rest, seed):
        m = lgb.train(dict(p, seed=seed), lgb.Dataset(X.iloc[tr], y[tr]), 1200)
        return [m.predict(r) for r in rest], 1200
    return fit


def mono_fit(th, X, y):
    cons = []
    for k in X.columns:
        if str(X[k].dtype) == "category":
            cons.append(0)
            continue
        x = X[k].to_numpy()
        f = np.isfinite(x)
        r = spearmanr(x[f], y[f])[0] if f.sum() > 1000 and np.unique(x[f]).size > 2 else 0.0
        cons.append(int(np.sign(r)) if abs(r) > 0.05 else 0)
    print(f"monotone constraints on {sum(c != 0 for c in cons)} of {len(cons)} features", flush=True)
    return v.lgb_fit(dict(v.LGB, n_jobs=th, monotone_constraints=cons, monotone_constraints_method="advanced"))


def sk_fit(cls, th):
    def fit(X, y, tr, fi, st, rest, seed):
        m = cls(n_estimators=500, min_samples_leaf=50, max_features=0.3, n_jobs=th, random_state=seed).fit(numeric(X.iloc[tr]), y[tr])
        return [m.predict_proba(numeric(r))[:, 1] for r in rest], 500
    return fit


def seg_fit(th, X):
    base = v.lgb_fit(dict(v.LGB, n_jobs=th))
    seg = X["has_history"].to_numpy() if "has_history" in X.columns else None
    assert seg is not None

    def fit(X, y, tr, fi, st, rest, seed):
        out = [np.zeros(len(r)) for r in rest]
        for g in (0, 1):
            ts = tr[seg[tr] == g]
            pos = {j: i for i, j in enumerate(ts)}
            fi_g = np.array([pos[j] for j in tr[fi] if j in pos])
            st_g = np.array([pos[j] for j in tr[st] if j in pos])
            masks = [r["has_history"].to_numpy() == g for r in rest]
            preds, _ = base(X, y, ts, fi_g, st_g, [r[mk] for r, mk in zip(rest, masks)], seed)
            for o, pr, mk in zip(out, preds, masks):
                o[mk] = pr
        return out, 0
    return fit


def main():
    member = sys.argv[1]
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    fit = {"dart": lambda: dart_fit(th), "mono": lambda: mono_fit(th, X, y),
           "rf": lambda: sk_fit(RandomForestClassifier, th), "et": lambda: sk_fit(ExtraTreesClassifier, th),
           "seg": lambda: seg_fit(th, X)}[member]()
    oof, _, _ = v.run(f"{member}_s42", fit, X, Xt, y, ids, test_ids, OUT, seeds=[42], test=False)
    yy, cols = bs.load()
    six = np.mean([cols[m] for m in bs.SIX], 0)
    r = rankdata(oof) / len(oof)
    print(f"RESULT {member}: single KS {v.ks(y, oof):.5f}, spearman vs six {np.corrcoef(rankdata(six), r)[0, 1]:.4f}")
    for w in (1 / 7, 0.25):
        b = (1 - w) * six + w * r
        mu, pp = bs.boot(yy, b, six)
        print(f"RESULT {member}: w {w:.2f} blend KS {v.ks(yy, b):.5f} vs six {v.ks(yy, six):.5f} boot delta {mu:+.5f} P>0 {pp:.2f}", flush=True)


if __name__ == "__main__":
    main()
