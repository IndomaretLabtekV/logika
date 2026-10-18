"""Merge partial validation_v2 runs of one model (disjoint folds, same seed) into one oof/test/log set.
OOF: take the non-NaN value per row. Test: fold-count-weighted mean of each run's averaged test predictions.
Usage: python merge_runs.py OUT NAME DIR [DIR ...]"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

out, name, dirs = Path(sys.argv[1]), sys.argv[2], [Path(d) for d in sys.argv[3:]]
oof, test, logs = None, 0, []
for d in dirs:
    log = pl.read_csv(d / f"log_{name}.csv")
    o = pl.read_parquet(d / f"oof_{name}.parquet").sort("application_id")
    t = pl.read_parquet(d / f"test_{name}.parquet").sort("application_id")
    if oof is None:
        oof, ids = o, t["application_id"]
    else:
        assert (o["application_id"] == oof["application_id"]).all() and (t["application_id"] == ids).all()
        a, b = oof[name].to_numpy(), o[name].to_numpy()
        assert not (~np.isnan(a) & ~np.isnan(b)).any(), "runs overlap"
        oof = oof.with_columns(pl.Series(name, np.where(np.isnan(a), b, a)))
    test = test + t[name].to_numpy() * log.height
    logs.append(log)
log = pl.concat(logs).sort("seed", "fold")
assert log.height == log.unique(["seed", "fold"]).height
out.mkdir(parents=True, exist_ok=True)
oof.write_parquet(out / f"oof_{name}.parquet")
pl.DataFrame({"application_id": ids, name: test / log.height}).write_parquet(out / f"test_{name}.parquet")
log.write_csv(out / f"log_{name}.csv")
print(log, "\nNaN OOF rows:", int(np.isnan(oof[name].to_numpy()).sum()))
