"""v72 Kaggle run: 9 GBDT (6 v22 family + lgb_x, xgb_d8, cat_d8) on main features + f251 Markov, seeds 42/7/2026.
One process per device group so LightGBM (CPU), XGBoost (GPU 0) and CatBoost (GPU 1) run side by side.
Usage: python kaggle_v72_run.py lgb|xgb|cat   (features.parquet and f251 must already exist in data_pq/)
"""
import sys

import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "v72"
GROUPS = {
    "lgb": {"lgb": lambda: v.lgb_fit(dict(v.LGB, n_jobs=2)), "lgb_s": lambda: v.lgb_fit(dict(v.LGB_S, n_jobs=2)),
            "lgb_x": lambda: v.lgb_fit(dict(v.LGB_X, n_jobs=2))},
    "xgb": {"xgb": lambda: v.xgb_fit(v.XGB), "xgb_s": lambda: v.xgb_fit(v.XGB_S), "xgb_d8": lambda: v.xgb_fit(v.XGB_D8)},
    "cat": {"cat": lambda: v.cat_fit(v.CAT), "cat_s": lambda: v.cat_fit(v.CAT_S), "cat_d8": lambda: v.cat_fit(v.CAT_D8)},
}


def main():
    feats = pl.read_parquet(PQ / "features.parquet").join(
        pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    print(f"{X.shape[1]} features, {len(y)} train rows, {len(Xt)} test rows", flush=True)
    for name, make in GROUPS[sys.argv[1]].items():
        v.run(name, make(), X, Xt, y, ids, test_ids, OUT)


if __name__ == "__main__":
    main()
