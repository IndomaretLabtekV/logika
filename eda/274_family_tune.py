"""Hyperparameter screen for XGBoost and CatBoost on features + f251 (CPU, local), same protocol as 273.
Reference: results/v71/family_add_eval/oof_{xgb|cat}__f251_repayment_markov_s{seed} (v22 params, same features).
Usage: cd eda && ../.venv/bin/python 274_family_tune.py xgb|cat name[,name...] [seed] [threads]
"""
import sys

import numpy as np
import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v73/family_tune"
CONFIGS = {
    "xgb": {
        "lr01": dict(learning_rate=0.01), "d4": dict(max_depth=4), "d6": dict(max_depth=6, min_child_weight=50, reg_lambda=10.0),
        "cs03": dict(colsample_bytree=0.3), "gamma1": dict(gamma=1.0),
        "lossguide": dict(grow_policy="lossguide", max_depth=0, max_leaves=31, min_child_weight=20),
        "mcw50": dict(min_child_weight=50), "bin64": dict(max_bin=64),
    },
    "cat": {
        "d5": dict(depth=5), "d7": dict(depth=7, l2_leaf_reg=10), "l2_20": dict(l2_leaf_reg=20), "rsm05": dict(rsm=0.5),
        "lossguide": dict(grow_policy="Lossguide", max_leaves=31, min_data_in_leaf=100),
        "border254": dict(border_count=254), "lr05": dict(learning_rate=0.05), "rs1": dict(random_strength=3, bagging_temperature=1),
    },
}


def main():
    fam, names = sys.argv[1], sys.argv[2].split(",")
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    th = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    feats = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    X, Xt, y, ids, test_ids = v.frames(feats)
    ref = pl.read_parquet(ROOT / f"results/v71/family_add_eval/oof_{fam}__f251_repayment_markov_s{seed}.parquet")
    assert (ref["application_id"] == ids).all()
    rv = ref[ref.columns[-1]].to_numpy()
    for name in names:
        if fam == "xgb":
            fit = v.xgb_fit(dict(v.XGB, device="cpu", nthread=th, **CONFIGS[fam][name]))
        else:
            fit = v.cat_fit(dict(v.CAT, task_type="CPU", thread_count=th, **CONFIGS[fam][name]))
        oof, _, log = v.run(f"{fam}_{name}_s{seed}", fit, X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
        d = [v.ks(y[va], oof[va]) - v.ks(y[va], rv[va]) for _, va in v.outer(y, seed)]
        print(f"RESULT {fam} {name:10s} KS {v.ks(y, oof):.5f} vs ref {v.ks(y, rv):.5f} delta {v.ks(y, oof) - v.ks(y, rv):+.5f} "
              f"folds won {sum(x > 0 for x in d)}/5 per fold {np.round(d, 4)} trees {[r['budget'] for r in log]}", flush=True)


if __name__ == "__main__":
    main()
