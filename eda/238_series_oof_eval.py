"""Customer-level evaluation of series-OOF aggregates (237): v22 LGB on main232 + so_<table>_* features vs
main232 only, same seed-42 outer folds and inner stop split. The aggregates for outer fold k come from
results/v66/series_oof/<table>_f<k>.parquet (row models never saw fold-k labels).
Usage: cd eda && ../.venv/bin/python 238_series_oof_eval.py <tables e.g. ecom,repay> <folds e.g. 0>
"""
import sys

import numpy as np
import pandas as pd
import polars as pl

import valid2 as v
from common import ROOT

OUT = ROOT / "results/v66/series_oof_eval"


def main():
    tables, folds = sys.argv[1].split(","), [int(f) for f in sys.argv[2].split(",")]
    feats = pl.read_parquet(ROOT / "data_pq/features.parquet")
    X, Xt, y, ids, test_ids = v.frames(feats)
    tag = "+".join(tables)
    base = v.lgb_fit(dict(v.LGB, n_jobs=4))

    def fit(Xf, yy, tr, fi, st, rest, seed):
        k = next(j for j, (a, _) in enumerate(v.outer(yy, 42)) if np.array_equal(a, tr))
        extra = None
        for t in tables:
            e = pl.read_parquet(ROOT / f"results/v66/series_oof/{t}_f{k}.parquet")
            extra = e if extra is None else extra.join(e, on="application_id")
        e = extra.filter(pl.col("application_id").is_in(ids.to_list())).sort("application_id")
        assert e["application_id"].to_list() == ids.to_list()
        Xa = pd.concat([Xf.reset_index(drop=True), e.drop("application_id").to_pandas()], axis=1)
        return base(Xa, yy, tr, fi, st, [Xa.iloc[np.asarray(r.index)] for r in rest], seed)  # rest = X.iloc[va] (RangeIndex)

    v.run(f"lgb_so_{tag}", fit, X, Xt, y, ids, test_ids, OUT, seeds=[42], folds=folds, test=False)
    bp = OUT / "oof_lgb_base4.parquet"
    if not bp.exists() or not np.isfinite(pl.read_parquet(bp)["lgb_base4"].to_numpy()[np.concatenate([v.outer(y, 42)[k][1] for k in folds])]).all():
        v.run("lgb_base4", base, X, Xt, y, ids, test_ids, OUT, seeds=[42], folds=folds, test=False)  # same threads/params
    b = pl.read_parquet(bp)["lgb_base4"].to_numpy()
    c = pl.read_parquet(OUT / f"oof_lgb_so_{tag}.parquet")[f"lgb_so_{tag}"].to_numpy()
    for k in folds:
        _, va = v.outer(y, 42)[k]
        print(f"fold {k}: base {v.ks(y[va], b[va]):.5f}  so {v.ks(y[va], c[va]):.5f}")
    rows = np.concatenate([v.outer(y, 42)[k][1] for k in folds])
    if len(folds) == 5:
        rng = np.random.default_rng(238)
        pos, neg = rows[y[rows] == 1], rows[y[rows] == 0]
        d = [v.ks(y[r], c[r]) - v.ks(y[r], b[r]) for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))]
        print(f"pooled: base {v.ks(y, b):.5f} so {v.ks(y, c):.5f} delta {v.ks(y, c) - v.ks(y, b):+.5f} "
              f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {np.mean(np.array(d) > 0):.3f}")


if __name__ == "__main__":
    main()
