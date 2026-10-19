"""Does a forward-in-time validation reproduce the public ranking of model families (trees > LR > Kumo > Causilo,
eda/131) that random-split CV gets wrong? For decision months m = 6..11: train on decision months <= m, score month
m+1 (one block each). Same fit functions as validation_v2 (inner 1/7 stop split inside the training block). Models:
LGB, LGB_s (trees), lr_q (LR rank-gauss), Causilo. Compares the mean forward KS with the random-CV KS of the same
model (clean OOF). Saves eda/out/140_forward_transfer.png and the per-block predictions.
Usage: cd eda && ../.venv/bin/python 140_forward_transfer.py [--models lgb,lgb_s,lr_q,causilo]"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import run_valid2
import valid2 as v
from common import OUT, ROOT, features

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="lgb,lgb_s,lr_q,causilo")
a = ap.parse_args()
R = ROOT / "results"
X, Xt, y, ids, tids = v.frames(features())
month = X["dec_month"].to_numpy().astype(int)
cv_ref = {"lgb": R / "v22/v22/gbdt/oof_lgb.parquet", "lgb_s": R / "v22/v22/gbdt/oof_lgb_s.parquet",
          "lr_q": R / "screen/lin_5f/oof_lr_q.parquet", "causilo": R / "v22/v22/fm/oof_causilo.parquet"}
res = {}
for name in a.models.split(","):
    fit = run_valid2.model(name)
    ks_f, pred = [], np.full(len(y), np.nan)
    for m in range(6, 12):
        tr, va = np.where(month <= m)[0], np.where(month == m + 1)[0]
        fi, st = v.inner(y[tr], 42, m)
        (p,), _ = fit(X, y, tr, fi, st, [X.iloc[va]], 42)
        pred[va] = p
        ks_f.append(v.ks(y[va], p))
    d = pl.read_parquet(cv_ref[name]).sort("application_id")
    oof = d[[c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()
    blocks = month >= 7
    ks_cv = [v.ks(y[month == m + 1], oof[month == m + 1]) for m in range(6, 12)]
    res[name] = (np.mean(ks_f), np.mean(ks_cv))
    print(f"{name:8s} forward KS per month {[round(k, 4) for k in ks_f]} mean {np.mean(ks_f):.5f} | random-CV OOF on the same "
          f"months {np.mean(ks_cv):.5f} | drop {np.mean(ks_f) - np.mean(ks_cv):+.5f}; pooled forward {v.ks(y[blocks], pred[blocks]):.5f}",
          flush=True)
    pl.DataFrame({"application_id": ids, "target": y, f"fwd_{name}": pred}).write_parquet(OUT / f"140_fwd_{name}.parquet")
fig, ax = plt.subplots(figsize=(5, 3.5))
for n, (f_, c_) in res.items():
    ax.scatter(c_, f_)
    ax.annotate(n, (c_, f_))
ax.set(xlabel="random-CV KS (months 7-12)", ylabel="forward KS (months 7-12)", title="Forward-time transfer by model")
fig.tight_layout()
fig.savefig(OUT / "140_forward_transfer.png", dpi=110)
