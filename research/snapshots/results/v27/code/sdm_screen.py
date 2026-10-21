"""NVIDIA structured-data-models (sdm) foundation models on the validation_v2 fold-0 screen (seed 42, outer fold 0):
KumoTabular (small/medium/large), TabFM, TabICLv2. In-context learners, so no early stopping: the context is the
whole outer training fold (or a stratified subsample of `--ctx` rows), features = valid2.fm_select (100 numeric by
training-fold KS + the 4 categoricals as integer codes). Runs in .venv-zoo.
Usage: .venv-zoo/bin/python eda/sdm_screen.py kumo_large [tabfm ...] [--ctx 20000] [--est 4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
import valid2 as v  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "screen" / "sdm_f0"


def make(name, dev):
    import sdm
    if name.startswith("kumo_"):
        return sdm.models.KumoTabular(task="classification", size=name.split("_")[1], device=dev)
    if name == "tabfm":  # non-commercial licence, accepted for this student-competition research use (journal 17 Okt)
        return sdm.models.TabFM(task="classification", device=dev, accept_license=True)
    return sdm.models.TabICLv2(task="classification", device=dev)


def frame(X, cols):
    d = X[cols].copy()
    for c in v.CAT_COLS:
        d[c] = d[c].cat.codes.astype(float).replace(-1, np.nan)
    return d.astype(float).reset_index(drop=True)


def predict(model, ctx_df, y_ctx, q_df, est, chunk, dev):
    import sdm
    full = pd.concat([ctx_df.assign(target=y_ctx.astype(str)), q_df.assign(target="0")], ignore_index=True)
    st = sdm.infer_stypes(full, overrides={"target": "categorical", **{c: "numerical" for c in ctx_df.columns}})
    t = sdm.TableTensor.from_pandas(df=full, stypes=st, device=dev)
    n = len(ctx_df)
    xc, yc = t[:n].drop_columns("target"), t[:n, "target"]
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=not FP32):
        for i in range(n, len(full), chunk):
            p = model(x_context=xc, y_context=yc, x_query=t[i:min(i + chunk, len(full))].drop_columns("target"),
                      num_estimators=est)
            out.append(p)
    return out


FP32 = False


def main(names, ctx, est, chunk, nfeat=100, features=None, out=None):
    global OUT
    OUT = Path(out) if out else OUT
    X, Xt, y, ids, tids = v.frames(pl.read_parquet(features or ROOT / "data_pq" / "features.parquet"))
    tr, va = v.outer(y, 42)[0]
    cols = v.fm_select(X, y, tr, nfeat)
    if ctx and len(tr) > ctx:
        rng = np.random.default_rng(0)
        tr_c = np.sort(np.concatenate([rng.choice(tr[y[tr] == c], int(ctx * (y[tr] == c).mean()), replace=False) for c in (0, 1)]))
    else:
        tr_c = tr
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    for name in names:
        t0 = time.time()
        model = make(name, dev)
        res = predict(model, frame(X.iloc[tr_c], cols), y[tr_c], frame(X.iloc[va], cols), est, chunk, dev)
        print(name, type(res[0]), getattr(res[0], "columns", None), flush=True)
        p = np.concatenate([r.to_pandas()["1"].to_numpy() if hasattr(r, "to_pandas") else r[..., 1].float().cpu().numpy()
                            for r in res])
        oof = np.full(len(y), np.nan)
        oof[va] = p
        tag = f"{name}_c{len(tr_c)}_e{est}_f{len(cols)}"
        pl.DataFrame({"application_id": ids, "target": y, tag: oof}).write_parquet(OUT / f"oof_{tag}.parquet")
        print(dict(model=tag, ks=v.ks(y[va], p), sec=round(time.time() - t0)), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--est", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--nfeat", type=int, default=100)
    ap.add_argument("--features")
    ap.add_argument("--out")
    ap.add_argument("--fp32", action="store_true", help="no fp16 autocast (TabFM gave NaN under fp16 on T4)")
    a = ap.parse_args()
    FP32 = a.fp32
    main(a.names, a.ctx, a.est, a.chunk, a.nfeat, a.features, a.out)
