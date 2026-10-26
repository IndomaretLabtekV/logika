"""Build strict outer-fold feature pools (fit = outer-train IDs only); CPU, resumable via build_pool."""
import os
import sys
import time

import numpy as np
import polars as pl

import valid2 as v
from common import ROOT
from strict_feature_pool import build_pool

train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
for fold in map(int, sys.argv[1:] or range(5)):
    tr, _ = v.outer(y, 42)[fold]
    t = time.time()
    pool = build_pool(ids[tr].tolist(), ROOT / f"results/v60/strict_tfm/pool_outer_f{fold}")
    os.sync()  # crash 24 Okt 00:35 left a 0-byte manifest; flush before moving on
    m = pool["manifest"]
    print(fold, m["fitting_rows"], len(m["union_columns"]), f"{time.time() - t:.0f}s", flush=True)
