"""LightGBM "v10": per outer fold, one of valid2.LGB_GRID (10 configs) chosen on the inner stop split only, then refit.
Features + f251, seeds 42/7/2026, with test predictions. Member for the 4-model recipe (seg LGB, seg XGB, MLP, v10).
Usage: cd eda && ../.venv/bin/python 293_lgb_nested.py
"""
import polars as pl

import valid2 as v
from common import PQ, ROOT


def main():
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    v.run("lgb_nested", v.lgb_nested_fit(), X, Xt, y, ids, test_ids, ROOT / "results/v73/lgb_nested", seeds=[42, 7, 2026])


if __name__ == "__main__":
    main()
