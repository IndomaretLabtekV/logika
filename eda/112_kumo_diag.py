"""Why does kumo_finetune.py score 0.298 on fold 0 when the proven v28m Kumo (context = whole training fold) scores
0.5256? Same 2,000 outer-fold-0 query rows, one factor changed at a time, each compared row by row with the saved
v28m OOF (Spearman) and by KS on those rows:
  A  sdm_screen/valid2 style: table = context + query (query target "0"), context = whole training fold
  B  kumo_finetune style: one 70k TableTensor (all train rows, true targets), rows taken by index
  C  B with query target column set to "0" before building the table
  D  B but the table holds only context + query rows (no other outer-fold rows)
Usage: cd eda && ../.venv/bin/python 112_kumo_diag.py [--n 2000] [--ctx 0]"""
import argparse
import time

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.stats import spearmanr

import sdm
import valid2 as v
from common import OUT, ROOT, features

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--ctx", type=int, default=0, help="0 = whole training fold")
ap.add_argument("--cases", default="ABCD")
a = ap.parse_args()

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
cols = v.fm_select(X, y, tr, 100)
ctx = tr if not a.ctx else np.sort(np.random.default_rng(0).choice(tr, a.ctx, replace=False))
q = va[: a.n]
ref = pl.read_parquet(ROOT / "results/v28m/oof_kumo_large_c56000.parquet").to_pandas().set_index("application_id")
ref = ref.loc[ids[q], "kumo_large_c56000"].to_numpy()
dev = torch.device("cuda")
model = sdm.models.KumoTabular(task="classification", size="large", device=dev)
print(f"ctx {len(ctx)} query {len(q)} pos {y[q].sum()}  v28m on these rows KS {v.ks(y[q], ref):.5f}", flush=True)


def frame(rows):
    d = X.iloc[rows][cols].copy()
    for c in v.CAT_COLS:
        d[c] = d[c].cat.codes.astype(float).replace(-1, np.nan)
    return d.astype(float).reset_index(drop=True)


def run(T, ci, qi, seed=4200):
    xc, yc = T[ci].drop_columns("target"), T[ci][:, "target"]
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(qi), 1000):
            r = model(x_context=xc, y_context=yc, x_query=T[qi[i:i + 1000]].drop_columns("target"), num_estimators=4, generator=g)
            out.append(r.to_pandas()["1"].to_numpy())
    return np.concatenate(out)


def table(df):
    st = sdm.infer_stypes(df, overrides={"target": "categorical", **{c: "numerical" for c in cols}})
    return sdm.TableTensor.from_pandas(df=df, stypes=st, device=dev), st


res = {}
for case in a.cases:
    t0 = time.time()
    if case in "AD":
        lab = np.r_[y[ctx].astype(str), (np.full(len(q), "0") if case == "A" else y[q].astype(str))]
        T, st = table(pd.concat([frame(ctx), frame(q)], ignore_index=True).assign(target=lab))
        ci, qi = torch.arange(len(ctx), device=dev), torch.arange(len(ctx), len(ctx) + len(q), device=dev)
    else:
        lab = y.astype(str).astype(object)
        if case == "C":
            lab[q] = "0"
        T, st = table(frame(np.arange(len(y))).assign(target=lab))
        ci, qi = torch.as_tensor(ctx, device=dev), torch.as_tensor(q, device=dev)
    p = run(T, ci, qi)
    res[case] = p
    print(f"{case}: KS {v.ks(y[q], p):.5f}  spearman vs v28m {spearmanr(p, ref)[0]:.4f}  mean p {p.mean():.4f}  "
          f"target stype {st['target']}  {time.time() - t0:.0f} s", flush=True)
    del T
    torch.cuda.empty_cache()

pl.DataFrame({"application_id": ids[q], "target": y[q], "v28m": ref, **res}).write_parquet(OUT / "112_kumo_diag.parquet")
