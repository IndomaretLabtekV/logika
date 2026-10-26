"""Inner fold-0 screen: cap300 @ 28672 on the deduplicated/affine-pruned 777-column pool vs the raw
896 pool (cached 226 arm). Random 300-column subsets per estimator waste slots on the 104 duplicate/
constant and 15 affine columns; the 777 recipe was fitted on the same 48k inner-fit rows
(results/eda_23okt/affine_view/preprocessing.json, hash checked here).
Usage: cd eda && ../.venv/bin/python 228_dedup_pool_cap300.py A_cap300_ctx28672_777
"""
import hashlib
import importlib.util
import json
import sys

import numpy as np
import polars as pl

import valid2 as v
from common import ROOT

spec = importlib.util.spec_from_file_location("screen222", ROOT / "eda/222_context_ssl_screen.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
recipe = json.loads((ROOT / "results/eda_23okt/affine_view/preprocessing.json").read_text())
train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
tr, _ = v.outer(y, 42)[0]
fi, _ = v.inner(y[tr], 42, 0)
assert recipe["fitting_ids_sha256"] == hashlib.sha256("\n".join(sorted(ids[tr[fi]])).encode()).hexdigest()
KEPT = set(recipe["kept_columns"])
assert len(KEPT) == 777
s.OUT = ROOT / "results/v61/dedup_pool_f0"
s.ARMS = {"W_cap300_ctx28672": ("wide", 28672, .85, "fit"), "A_cap300_ctx28672_777": ("wide", 28672, .85, "fit")}
_predict = s.predict


def predict(X, context, query, labels, columns, seed, **kw):
    kw["column_cap"] = 300
    if sys.argv[1].startswith("A_"):
        columns = [c for c in columns if c in KEPT]
        assert len(columns) == 777
    return _predict(X, context, query, labels, columns, seed, **kw)


s.predict = predict
s.__file__ = __file__
if __name__ == "__main__":
    s.main()
