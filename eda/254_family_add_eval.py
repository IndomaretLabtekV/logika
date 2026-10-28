"""Does a label-free feature block survive the 6-GBDT family? v22 configs (CPU here), seed-42 outer folds, screening.
Runs lgb, lgb_s, xgb, xgb_s, cat, cat_s on main features and on main + data_pq/<block>.parquet, then compares each
member and the equal-weight rank mean of the family (the trees part of v31/v59). The block variant also writes
test predictions (fold-averaged) so a confirmed block can feed a submission candidate without rerunning.
Usage: cd eda && ../.venv/bin/python 254_family_add_eval.py f251_repayment_markov [seed]
"""
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v71/family_add_eval"
T = 8
MODELS = {
    "lgb": lambda: v.lgb_fit(dict(v.LGB, n_jobs=T)),
    "lgb_s": lambda: v.lgb_fit(dict(v.LGB_S, n_jobs=T)),
    "xgb": lambda: v.xgb_fit(dict(v.XGB, device="cpu", nthread=T)),
    "xgb_s": lambda: v.xgb_fit(dict(v.XGB_S, device="cpu", nthread=T)),
    "cat": lambda: v.cat_fit(dict(v.CAT, task_type="CPU", thread_count=T)),
    "cat_s": lambda: v.cat_fit(dict(v.CAT_S, task_type="CPU", thread_count=T)),
}


def main():
    block = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    feats = pl.read_parquet(PQ / "features.parquet")
    aug = feats.join(pl.read_parquet(PQ / f"{block}.parquet"), on="application_id", how="left")
    frames = {"base": v.frames(feats), block: v.frames(aug)}
    y, ids = frames["base"][2], frames["base"][3]
    assert frames[block][3].to_list() == ids.to_list()

    oof = {}
    for name, make in MODELS.items():
        for tag, (X, Xt, _, _, test_ids) in frames.items():
            run = f"{name}__{tag}_s{seed}"
            path = OUT / f"oof_{run}.parquet"
            if not path.exists():
                v.run(run, make(), X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=tag == block)
            oof[(name, tag)] = pl.read_parquet(path)[run].to_numpy()
        print(f"{name}: base {v.ks(y, oof[(name, 'base')]):.5f}  +{block} {v.ks(y, oof[(name, block)]):.5f}", flush=True)

    fam = {tag: np.mean([rankdata(oof[(m, tag)]) for m in MODELS], 0) for tag in frames}
    b, c = fam["base"], fam[block]
    for k, (_, va) in enumerate(v.outer(y, seed)):
        print(f"family fold {k}: base {v.ks(y[va], b[va]):.5f}  new {v.ks(y[va], c[va]):.5f}  "
              f"delta {v.ks(y[va], c[va]) - v.ks(y[va], b[va]):+.5f}")
    rng = np.random.default_rng(254)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.array([v.ks(y[r], c[r]) - v.ks(y[r], b[r])
                  for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))])
    print(f"family pooled: base {v.ks(y, b):.5f} new {v.ks(y, c):.5f} delta {v.ks(y, c) - v.ks(y, b):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {np.mean(d > 0):.3f}")


if __name__ == "__main__":
    main()
