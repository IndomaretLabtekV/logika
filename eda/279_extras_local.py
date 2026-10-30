"""The three v72 extra models (lgb_x, xgb_d8, cat_d8) over the five v73 seeds, locally on CPU, with test predictions,
so that the nine-model blend can be compared with the six-model blend at 8 seeds each.
Usage: cd eda && ../.venv/bin/python 279_extras_local.py lgb_x|xgb_d8|cat_d8 [threads]
"""
import sys

import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v73/extras_local"
SEEDS = [1, 11, 101, 777, 31337]


def main():
    name = sys.argv[1]
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fit = {"lgb_x": lambda: v.lgb_fit(dict(v.LGB_X, n_jobs=th)),
           "xgb_d8": lambda: v.xgb_fit(dict(v.XGB_D8, device="cpu", nthread=th)),
           "cat_d8": lambda: v.cat_fit(dict(v.CAT_D8, task_type="CPU", thread_count=th))}[name]()
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    v.run(name, fit, X, Xt, y, ids, test_ids, OUT, seeds=SEEDS)


if __name__ == "__main__":
    main()
