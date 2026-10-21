"""Kumo-Tabular large with the whole training set (70k rows) as context, predicting the 30k test rows once.
Cheaper alternative to averaging the 5 fold models for the final notebook. Prints wall time and peak VRAM.
Usage: python kumo_fulltrain.py OUT_DIR [--features PATH] [--g_off K]"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # v30 v1 OOM: 7 GB reserved-unallocated
import numpy as np
import polars as pl
import torch

import valid2 as v

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--features", default=None)
ap.add_argument("--chunk", type=int, default=500)
ap.add_argument("--g_off", type=int, default=0, help="other estimator draw, same context (output name gets _g<k>)")
a = ap.parse_args()
if a.features:
    feats = pl.read_parquet(a.features)
else:
    from common import features
    feats = features()
X, Xt, y, ids, tids = v.frames(feats)
t0 = time.time()
(p,), n = v.kumo_fit(size="large", max_context=10 ** 6, chunk=a.chunk, g_off=a.g_off)(X, y, np.arange(len(y)), None, None, [Xt], 42)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"kumo_full70k{'_g%d' % a.g_off if a.g_off else ''}"
pl.DataFrame({"application_id": tids, tag: p}).write_parquet(out / f"test_{tag}.parquet")
print(f"context {n}, test rows {len(p)}, {time.time() - t0:.0f} s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
