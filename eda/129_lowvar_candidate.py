"""Low-variance v31 candidate from saved clean test predictions: trees = rank mean of the 6 v22 GBDT (each the mean of
15 fold x seed models), Causilo = probability mean over every seed run available (v22 seed 42, v37 seeds 7/2026,
v41 seeds 1/2; weights = number of seeds per file), Kumo = mean of the 70k-context draws available (v34 + v41 g1/g2).
Families rank-averaged 1/3 each. Validates the submission format and prints Spearman vs v31 / v34 submissions.
CV analogue (eda/123, v31 with averaged components): 0.52626, +0.00185 vs clean v8 (P 0.88, 5/5).
Usage: cd eda && ../.venv/bin/python 129_lowvar_candidate.py ../results/v41/v31_lowvar.csv"""
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

from common import DATA, ROOT

R = ROOT / "results"
tids = pl.read_csv(DATA / "sample_submission.csv")["application_id"].sort()


def col(p):
    d = pl.read_parquet(p).sort("application_id")
    assert (d["application_id"] == tids).all(), p
    x = d[[c for c in d.columns if c != "application_id"][0]].to_numpy()
    assert np.isfinite(x).all(), p
    return x


rk = lambda x: rankdata(x) / len(x)  # noqa: E731
trees = rk(np.mean([rk(col(R / f"v22/v22/gbdt/test_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0))
cau = [(1, R / "v22/v22/fm/test_causilo.parquet"), (2, R / "v37/test_causilo.parquet"), (2, R / "v41/causilo_s12/test_causilo.parquet")]
cau = [(w, p) for w, p in cau if p.exists()]
kum = [p for p in [R / "v34/test_kumo_full70k.parquet", R / "v41/test_kumo_full70k_g1.parquet", R / "v41/test_kumo_full70k_g2.parquet"] if p.exists()]
print("causilo files:", [(w, p.parent.name) for w, p in cau], " kumo files:", [p.name for p in kum])
c = rk(sum(w * col(p) for w, p in cau) / sum(w for w, _ in cau))
k = rk(np.mean([col(p) for p in kum], axis=0))
s = (trees + c + k) / 3
sub = pl.DataFrame({"application_id": tids, "default_probability": s})
assert sub.height == 30000 and sub["application_id"].n_unique() == 30000 and np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
sub.write_csv(sys.argv[1])
for ref in ("v31", "v34"):
    r = pl.read_csv(R / ref / "submission.csv").sort("application_id")
    assert (r["application_id"] == tids).all()
    print(f"spearman vs {ref}: {spearmanr(s, r['default_probability']).statistic:.5f}")
if len(kum) > 1:
    print("kumo draw spearman:", [round(spearmanr(col(kum[0]), col(p)).statistic, 5) for p in kum[1:]])
print("wrote", sys.argv[1])
