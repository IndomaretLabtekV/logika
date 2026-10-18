"""Why did valid2.kumo_fit score 0.5182 on fold 0 when eda/sdm_screen.py scored 0.5276 with the same model?
Fold 0 (seed 42), Kumo large, 100 fm_select features, 4 estimators, 20k context. One factor changed at a time:
A  sdm_screen setup: context subsample rng(0) floor, no generator, table = context + fold rows
B  A + generator seeded 42
C  A with context subsample rng(1) / rng(2)          (context-draw variance)
D  A with the 30k test rows appended to the table    (does the table content change the preprocessing?)
Prints KS per variant and the rank correlation with A. Saves eda/out/106_kumo_variance.csv."""
import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.stats import spearmanr
import sdm
import valid2 as v
from common import OUT, features

X, Xt, y, ids, tids = v.frames(features())
tr, va = v.outer(y, 42)[0]
cols = v.fm_select(X, y, tr, 100)
dev = torch.device("cuda")
model = sdm.models.KumoTabular(task="classification", size="large", device=dev)


def frame(d):
    d = d[cols].copy()
    for c in v.CAT_COLS:
        d[c] = d[c].cat.codes.astype(float).replace(-1, np.nan)
    return d.astype(float).reset_index(drop=True)


def ctx_rows(seed, rounding):
    rng = np.random.default_rng(seed)
    k = {c: (int if rounding == "floor" else lambda z: int(round(z)))(20000 * (y[tr] == c).mean()) for c in (0, 1)}
    return np.sort(np.concatenate([rng.choice(tr[y[tr] == c], k[c], replace=False) for c in (0, 1)]))


def run(ctx, gen_seed=None, with_test=False, chunk=2000):
    q = pd.concat([frame(X.iloc[va])] + ([frame(Xt)] if with_test else []), ignore_index=True)
    cdf = frame(X.iloc[ctx])
    full = pd.concat([cdf.assign(target=y[ctx].astype(str)), q.assign(target="0")], ignore_index=True)
    st = sdm.infer_stypes(full, overrides={"target": "categorical", **{c: "numerical" for c in cols}})
    t = sdm.TableTensor.from_pandas(df=full, stypes=st, device=dev)
    n = len(cdf)
    xc, yc = t[:n].drop_columns("target"), t[:n, "target"]
    g = torch.Generator(device=dev).manual_seed(gen_seed) if gen_seed is not None else None
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(n, n + len(va), chunk):
            r = model(x_context=xc, y_context=yc, x_query=t[i:min(i + chunk, n + len(va))].drop_columns("target"),
                      num_estimators=4, generator=g)
            out.append(r.to_pandas()["1"].to_numpy())
    return np.concatenate(out)


base_ctx = ctx_rows(0, "floor")
res = {"A": run(base_ctx)}
res["A_repeat"] = run(base_ctx)
res["B_gen42"] = run(base_ctx, gen_seed=42)
res["C_ctx1"] = run(ctx_rows(1, "floor"))
res["C_ctx2"] = run(ctx_rows(2, "floor"))
res["D_with_test"] = run(base_ctx, with_test=True)
rows = []
for k, p in res.items():
    rows.append(dict(variant=k, ks=v.ks(y[va], p), spearman_vs_A=spearmanr(p, res["A"]).statistic))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "106_kumo_variance.csv", index=False)
