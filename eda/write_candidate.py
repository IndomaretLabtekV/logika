"""Write a rank-average candidate from saved clean test predictions and validate the submission format.
Usage: python write_candidate.py OUT.csv name=weight [...] --dirs results/v22/v22 [more]"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

import blend2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("weights", nargs="+")
ap.add_argument("--dirs", nargs="+", required=True)
a = ap.parse_args()
w = {k: float(x) for k, x in (s.split("=") for s in a.weights)}
_, test = blend2.load(a.dirs)
s = sum(wt * rankdata(test[n].to_numpy()) / test.height for n, wt in w.items()) / sum(w.values())
sub = pl.DataFrame({"application_id": test["application_id"], "default_probability": s})
ref = pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "sample_submission.csv")
assert sub.height == 30000 == ref.height and sub["application_id"].n_unique() == 30000
assert set(sub["application_id"]) == set(ref["application_id"])
assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
sub.write_csv(a.out)
v8 = pl.read_csv(Path(__file__).resolve().parent.parent / "results" / "v8" / "submission.csv").sort("application_id")
from scipy.stats import spearmanr
print(f"wrote {a.out}: rows {sub.height}, weights {w}, Spearman vs v8 submission "
      f"{spearmanr(sub.sort('application_id')['default_probability'], v8['default_probability']).statistic:.5f}")
