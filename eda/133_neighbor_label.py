"""Is there a 'neighbour' signal among training customers that memorising models (ICL, kNN) exploit in OOF?
Public: offsets public - CV order by memorisation capacity (LR +0.037 ~ trees +0.039 > Kumo +0.034 > Causilo +0.029).
Per outer fold (seed 42): features = rank-gauss of the 40 top fm_select columns fitted on the training fold (NaN -> 0
+ missing indicator for the history block); for each validation row the label mean of its k nearest training-fold rows
(k = 10, 50, 200). Then: does the neighbour label mean add information beyond the clean v31 OOF (LR chi2, cross-fit dKS),
and how does a pure kNN score compare with trees on OOF. Writes eda/out/133_oof_knn{k}.parquet (for a public probe).
Usage: cd eda && ../.venv/bin/python 133_neighbor_label.py"""
import numpy as np
import polars as pl
import torch
from scipy.special import logit
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression

import valid2 as v
from common import OUT, ROOT, features
from common9 import calibrate

R = ROOT / "results"
X, Xt, y, ids, tids = v.frames(features())
KS_ = (10, 50, 200)
oof = {k: np.full(len(y), np.nan) for k in KS_}
dev = torch.device("cuda")


def gauss(train_col, col):
    """ECDF of the training rows -> normal scores; NaN -> 0."""
    t = np.sort(train_col[~np.isnan(train_col)])
    out = np.zeros(len(col))
    m = ~np.isnan(col)
    if len(t):
        q = (np.searchsorted(t, col[m], side="left") + np.searchsorted(t, col[m], side="right")) / 2 / len(t)
        out[m] = norm.ppf(np.clip(q, 0.5 / len(t), 1 - 0.5 / len(t)))
    return out


for fold, (tr, va) in enumerate(v.outer(y, 42)):
    cols = [c for c in v.fm_select(X, y, tr, 40) if c not in v.CAT_COLS]
    A = np.column_stack([gauss(X[c].to_numpy()[tr], X[c].to_numpy()) for c in cols] + [X["has_history"].to_numpy() * 2.0])
    a_tr = torch.tensor(A[tr], dtype=torch.float32, device=dev)
    y_tr = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    res = {k: [] for k in KS_}
    for i in range(0, len(va), 2000):
        q = torch.tensor(A[va[i:i + 2000]], dtype=torch.float32, device=dev)
        idx = torch.cdist(q, a_tr).topk(max(KS_), largest=False).indices
        lab = y_tr[idx]
        for k in KS_:
            res[k].append(lab[:, :k].mean(1).cpu().numpy())
    for k in KS_:
        oof[k][va] = np.concatenate(res[k])
    print(f"fold {fold}: " + " ".join(f"knn{k} KS {v.ks(y[va], oof[k][va]):.4f}" for k in KS_), flush=True)

f = pl.read_parquet(R / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
assert (f["application_id"] == ids).all()


def raw(p):
    d = pl.read_parquet(p).sort("application_id")
    return d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(raw(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
v31 = (trees + rk(raw(R / "v22/v22/fm/oof_causilo.parquet")) + rk(raw(R / "v28m/oof_kumo_large_c56000.parquet"))) / 3
for ref_name, ref in [("trees", trees), ("v31", v31)]:
    lp = logit(np.clip(calibrate(y, ref), 1e-4, 1 - 1e-4))
    base = LogisticRegression(C=1e6, max_iter=500).fit(lp[:, None], y)
    dev0 = -2 * np.sum(np.log(np.where(y == 1, base.predict_proba(lp[:, None])[:, 1], base.predict_proba(lp[:, None])[:, 0])))
    for k in KS_:
        F = np.column_stack([lp, logit(np.clip(oof[k], 0.5 / k, 1 - 0.5 / k))])
        m = LogisticRegression(C=1e6, max_iter=500).fit(F, y)
        pr = m.predict_proba(F)
        dev1 = -2 * np.sum(np.log(np.where(y == 1, pr[:, 1], pr[:, 0])))
        print(f"vs {ref_name}: knn{k} alone KS {v.ks(y, oof[k]):.5f}; LR chi2 adding knn {dev0 - dev1:.1f} (df 1), coef {m.coef_[0][1]:+.3f}")
for k in KS_:
    pl.DataFrame({"application_id": ids, "target": y, f"knn{k}": oof[k]}).write_parquet(OUT / f"133_oof_knn{k}.parquet")
