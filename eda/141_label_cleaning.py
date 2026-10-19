"""Nested label cleaning for LGB (validation_v2-clean, seed 42, 5 outer folds). Inside each outer training fold an
inner 5-fold LGB gives OOF scores for the training rows; the most confident disagreements (y=1 with inner score
below its q-quantile among bads... i.e. the lowest-scored bads, and the highest-scored goods) are dropped or
down-weighted, then LGB (stop on the inner 1/7 split, refit x REFIT) predicts the outer fold. Outer labels never touch
the selection. Compared paired with plain LGB seed 42 (results/screen/seeds/s42).
Usage: cd eda && ../.venv/bin/python 141_label_cleaning.py [--fracs 0.02,0.05] [--mode drop|weight]"""
import argparse
import math

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import OUT, features

ap = argparse.ArgumentParser()
ap.add_argument("--fracs", default="0.02,0.05")
ap.add_argument("--mode", default="drop")
a = ap.parse_args()
X, Xt, y, ids, tids = v.frames(features())
fracs = [float(s) for s in a.fracs.split(",")]
oof = {fr: np.full(len(y), np.nan) for fr in fracs}
P = dict(v.LGB, seed=42)
for k, (tr, va) in enumerate(v.outer(y, 42)):
    inner = np.full(len(y), np.nan)
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=500 + k).split(tr, y[tr]):
        inner[tr[b_]] = lgb.train(P, lgb.Dataset(X.iloc[tr[a_]], y[tr[a_]]), 600).predict(X.iloc[tr[b_]])
    s = inner[tr]
    for fr in fracs:
        bad, good = y[tr] == 1, y[tr] == 0
        lo = np.quantile(s[bad], fr)          # lowest-scored bads
        hi = np.quantile(s[good], 1 - fr)     # highest-scored goods
        flag = (bad & (s <= lo)) | (good & (s >= hi))
        w = np.where(flag, 0.0 if a.mode == "drop" else 0.3, 1.0)
        fi, st = v.inner(y[tr], 42, k)
        keep_fi = fi[w[fi] > 0] if a.mode == "drop" else fi
        m = lgb.train(P, lgb.Dataset(X.iloc[tr[keep_fi]], y[tr[keep_fi]], weight=w[keep_fi]), v.N_ROUNDS,
                      valid_sets=[lgb.Dataset(X.iloc[tr[st]], y[tr[st]])], callbacks=[lgb.early_stopping(v.EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * v.REFIT)
        keep = np.arange(len(tr)) if a.mode != "drop" else np.where(w > 0)[0]
        oof[fr][va] = lgb.train(P, lgb.Dataset(X.iloc[tr[keep]], y[tr[keep]], weight=w[keep]), n).predict(X.iloc[va])
        print(f"fold {k} frac {fr}: flagged {flag.sum()} rows, outer KS {v.ks(y[va], oof[fr][va]):.5f}", flush=True)
base = pl.read_parquet("../results/screen/seeds/s42/oof_lgb.parquet").sort("application_id")["lgb"].to_numpy()
for fr, o in oof.items():
    r = np.random.default_rng(0)
    d = np.array([(lambda i: v.ks(y[i], o[i]) - v.ks(y[i], base[i]))(r.choice(len(y), len(y))) for _ in range(500)])
    w_ = sum(v.ks(y[va], o[va]) > v.ks(y[va], base[va]) for _, va in v.outer(y, 42))
    print(f"{a.mode} frac {fr}: KS {v.ks(y, o):.5f} vs plain LGB s42 {v.ks(y, base):.5f} d {d.mean():+.5f} sd {d.std():.5f} "
          f"P>0 {(d > 0).mean():.2f} folds {w_}/5")
    pl.DataFrame({"application_id": ids, "target": y, f"clean_{a.mode}_{fr}": o}).write_parquet(OUT / f"141_oof_clean_{a.mode}_{fr}.parquet")
