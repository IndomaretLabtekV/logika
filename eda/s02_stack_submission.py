"""Write a logit-stacking submission: v8 9-model base + extra candidates (saved predictions only, no training).
Usage: python eda/s02_stack_submission.py out.csv name=oof.parquet:col:test.parquet:col [...]"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s01_stack_bench import BASE, ROOT, fit_lr, ks, load_base, logit  # noqa: E402

df = load_base()
folder = ROOT / "results/v8"
te = pl.DataFrame({"application_id": np.load(folder / "test_ids.npy", allow_pickle=True),
                   **{n: np.load(folder / f"test_{n}.npy") for n in BASE}})
names = list(BASE)
for arg in sys.argv[2:]:
    name, spec = arg.split("=", 1)
    op, oc, tp, tc = spec.split(":")
    df = df.join(pl.read_parquet(ROOT / op).select("application_id", pl.col(oc).alias(name)), on="application_id", validate="1:1")
    te = te.join(pl.read_parquet(ROOT / tp).select("application_id", pl.col(tc).alias(name)), on="application_id", validate="1:1")
    names.append(name)
assert df.height == 70000 and te.height == 30000 and not df.null_count().sum_horizontal()[0] and not te.null_count().sum_horizontal()[0]
y = df["target"].to_numpy()
X = np.column_stack([logit(df[n].to_numpy()) for n in names])
Xt = np.column_stack([logit(te[n].to_numpy()) for n in names])
mu, sd = X.mean(0), X.std(0)
w = fit_lr((X - mu) / sd, y)
s = (X - mu) / sd @ w[1:] + w[0]
st = (Xt - mu) / sd @ w[1:] + w[0]
print("weights", dict(zip(names, np.round(w[1:], 3))), "in-sample KS", round(ks(y, s), 5))
p = 1 / (1 + np.exp(-st))
assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
ref = pl.read_csv(ROOT / "results/v8/submission_v9_stack.csv")
out = te.select("application_id").with_columns(default_probability=pl.Series(p))
chk = ref.join(out, on="application_id", validate="1:1")
from scipy.stats import spearmanr  # noqa: E402
print("Spearman vs v8-stack submission", round(spearmanr(chk[:, 1], chk[:, 2]).statistic, 5))
out.write_csv(sys.argv[1])
print("wrote", sys.argv[1])
