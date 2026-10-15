"""Previous applications: credit hunger, rejections, historical pricing."""
import polars as pl
from common import analyze, features

f = features()
cols = [c for c in f.columns if c.startswith("pv_")] + ["has_history"]
analyze(f, cols, "08_prev", top_plot=16)
tr = f.filter(pl.col("is_train") == 1)
print(tr.group_by("pv_rate_last", "annual_interest_rate").agg(n=pl.len(), r=pl.col("target").mean())
      .pivot(on="annual_interest_rate", index="pv_rate_last", values="r").sort("pv_rate_last"))
