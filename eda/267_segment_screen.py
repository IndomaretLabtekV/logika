"""256 restricted to one segment: the 30% of applications with has_history == 0 (no account, repayment, previous data;
only application, bureau, e-commerce). Reference = trees+Markov family OOF ranked inside the segment.
A block that is flat on all rows can still move this segment, where the base KS is lowest (0.498 vs 0.536).
Usage: cd eda && ../.venv/bin/python 267_segment_screen.py block[,block] [0|1]
"""
import importlib
import sys

import numpy as np
import polars as pl

import common as c

lr = importlib.import_module("256_lr_screen")
scr = importlib.import_module("255_residual_screen")


def main():
    seg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ids, ref = scr.family("f251_repayment_markov")
    d = pl.DataFrame({"application_id": ids, "ref": ref}).join(
        c.features().select("application_id", "target", "has_history"), on="application_id", how="left", maintain_order="left")
    cand = []
    for b in sys.argv[1].split(","):
        e = pl.read_parquet(c.PQ / f"{b}.parquet")
        d = d.join(e, on="application_id", how="left", maintain_order="left")
        cand += [x for x in e.columns if x != "application_id"]
    d = d.filter(pl.col("has_history") == seg)
    y = d["target"].to_numpy()
    ref = d["ref"].to_numpy()
    d0, k0 = lr.dev(lr.design(ref), y), lr.cv_ks(lr.design(ref), y)
    print(f"segment has_history={seg}: {len(y)} train rows, bad {y.mean():.4f}, ref KS {c.ks(y, ref):.5f}, ref cvKS {k0:.5f}")
    print(f"{'feature':24s} {'chi2':>8s} {'cvKS delta':>11s} {'KS alone':>9s}")
    for k in cand:
        s = d[k].to_numpy().astype(float)
        if np.isfinite(s).sum() < 500:
            continue
        X = lr.design(ref, s)
        fin = np.isfinite(s)
        print(f"{k:24s} {d0 - lr.dev(X, y):8.1f} {lr.cv_ks(X, y) - k0:+11.5f} {c.ks(y[fin], s[fin]):9.4f}", flush=True)


if __name__ == "__main__":
    main()
