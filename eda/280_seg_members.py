"""Segment members with test predictions: one model per has_history segment (276 'seg': single seed KS 0.5229,
Spearman 0.971 with the six-model blend, +0.0006 as a 7th rank member). Same 8 seeds as v72 + v73.
Usage: cd eda && ../.venv/bin/python 280_seg_members.py lgb_seg|xgb_seg|xgb5_seg|xgbd8_seg|lgb_s_seg|cat_s_seg [threads] [b]
"""
import importlib
import sys

import polars as pl

import valid2 as v
from common import PQ, ROOT

dm = importlib.import_module("276_diverse_members")
OUT = ROOT / "results/v73/seg_members"
SEEDS = [42, 7, 2026, 1, 11, 101, 777, 31337]
OUT_B, SEEDS_B = ROOT / "results/v73/seg_members_b", [3, 5, 13, 17, 23, 29, 31, 37]  # "b": 8 more seeds


def segmented(base, X):
    seg = X["has_history"].to_numpy()

    def fit(X, y, tr, fi, st, rest, seed):
        import numpy as np
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
    name = sys.argv[1]
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    base = {"lgb_seg": lambda: v.lgb_fit(dict(v.LGB, n_jobs=th)),
            "xgb_seg": lambda: v.xgb_fit(dict(v.XGB_S, device="cpu", nthread=th)),
            "xgb5_seg": lambda: v.xgb_fit(dict(v.XGB, device="cpu", nthread=th)),
            "xgbd8_seg": lambda: v.xgb_fit(dict(v.XGB_D8, device="cpu", nthread=th)),
            "lgb_s_seg": lambda: v.lgb_fit(dict(v.LGB_S, n_jobs=th)),
            "cat_s_seg": lambda: v.cat_fit(dict(v.CAT_S, task_type="CPU", thread_count=th))}[name]()
    extra = len(sys.argv) > 3 and sys.argv[3] == "b"
    v.run(name, segmented(base, X), X, Xt, y, ids, test_ids, OUT_B if extra else OUT, seeds=SEEDS_B if extra else SEEDS)


if __name__ == "__main__":
    main()
