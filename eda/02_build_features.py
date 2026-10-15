"""Preprocessing: build the leak-safe customer-level feature matrix and sanity-check it."""
import time
import polars as pl
from common import PQ, features

t = time.time()
(PQ / "features.parquet").unlink(missing_ok=True)
f = features()
print(f"built {f.shape} in {time.time() - t:.0f}s")
assert f["application_id"].n_unique() == f.height == 100_000, "one row per selected application"
assert f.filter(pl.col("is_train") == 1)["target"].null_count() == 0
nul = f.select(pl.all().null_count() / pl.len()).transpose(include_header=True, column_names=["null_rate"])
with pl.Config(tbl_rows=300):
    print(nul.filter(pl.col("null_rate") > 0).sort("null_rate"))
num = [c for c, d in f.schema.items() if d.is_numeric()]
inf = {c: f[c].is_infinite().sum() for c in num if f[c].dtype.is_float() and f[c].is_infinite().sum()}
print("columns with inf:", inf)
