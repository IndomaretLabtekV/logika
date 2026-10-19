"""Temporal snapshots of EVERY feature: features.build_all recomputed with each application's decision_datetime moved
back by 90 and 180 days (all builders filter every table on that cutoff, so a snapshot only sees older data), then
deltas d90_x = x(T) - x(T-90), d180_x = x(T) - x(T-180) for every numeric feature that changes. Cutoff-safe by
construction (uses strictly older information); verified: no snapshot value may depend on rows dated after T-k.
Residual scan of the deltas vs the clean TREES-ONLY OOF (eda/134 setup); writes eda/out/142_snapshot_deltas.parquet
(application_id + delta columns) for a --extra model run.
Usage: cd eda && ../.venv/bin/python 142_snapshot_deltas.py [--shifts 90,180] [--out 142_snapshot_deltas]"""
import argparse
import io
import contextlib

import numpy as np
import polars as pl

import blend2
import valid2 as v
from common import OUT, PQ, TABLES, features, load
from common9 import calibrate, crossfit_dks, oe_test, safe_bins
from features import build_all

ap = argparse.ArgumentParser()
ap.add_argument("--shifts", default="90,180")
ap.add_argument("--out", default="142_snapshot_deltas")
args = ap.parse_args()
base = features().sort("application_id")
tables = {t: load(t) for t in TABLES}
snaps = {}
for k in [int(x) for x in args.shifts.split(",")]:
    p = PQ / f"features_shift{k}.parquet"
    if not p.exists():
        t2 = dict(tables)
        for a in ("train_applications", "test_applications"):
            t2[a] = tables[a].with_columns(decision_datetime=pl.col("decision_datetime") - pl.duration(days=k),
                                           application_datetime=pl.col("application_datetime") - pl.duration(days=k))
        build_all(t2).write_parquet(p)
    snaps[k] = pl.read_parquet(p).sort("application_id")
    assert (snaps[k]["application_id"] == base["application_id"]).all()
skip = set(v.ID_COLS) | set(v.CAT_COLS) | {"dec_month"}
nums = [c for c in base.columns if c not in skip and base[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32)]
cols = {}
for k, s in snaps.items():
    for c in nums:
        if c not in s.columns:  # selected-application bureau pivots do not exist k days earlier
            continue
        d = base[c].cast(pl.Float64) - s[c].cast(pl.Float64)
        if d.drop_nulls().drop_nans().std() and d.drop_nulls().drop_nans().std() > 0:
            cols[f"d{k}_{c}"] = d
D = pl.DataFrame({"application_id": base["application_id"], **cols})
D.write_parquet(OUT / f"{args.out}.parquet")
print(f"{len(cols)} delta columns (of {2 * len(nums)} candidates) written")
tr = D.filter(pl.Series(base["is_train"].to_numpy() == 1))
with contextlib.redirect_stdout(io.StringIO()):
    oof, _, _, _, _, res = blend2.main(["../results/v22/v22"])
assert (oof["application_id"] == tr["application_id"]).all()
y = oof["target"].to_numpy()
pc = calibrate(y, res["equal_trees"][0])
rng = np.random.default_rng(0)
x0 = tr[tr.columns[1]].to_numpy().astype(float)
null = [oe_test(y, pc, safe_bins(z, 10))[0] for z in (rng.permutation(x0) for _ in range(100))]
print(f"null chi2/df p95 {np.quantile(null, .95):.2f}, max of 100 {np.max(null):.2f}")
rows = [dict(feature=c, ks_uni=v.ks_nan(y, tr[c].to_numpy().astype(float)), chi2=oe_test(y, pc, safe_bins(tr[c].to_numpy().astype(float), 10))[0])
        for c in tr.columns if c != "application_id"]
R = pl.DataFrame(rows).sort("chi2", descending=True)
top = R.head(15)
top = top.with_columns(dks=pl.Series([crossfit_dks(y, pc, safe_bins(tr[c].to_numpy().astype(float), 10)) for c in top["feature"]]))
with pl.Config(tbl_rows=20, fmt_str_lengths=50):
    print(top)
R.write_csv(OUT / f"{args.out}.csv")
