"""Held-out logit stacking bench on saved OOF predictions (arithmetic only, no model training).
Usage: python eda/s01_stack_bench.py name=path/to/oof.parquet:col [...]
Base = v8 9-model stack (the submission_v9_stack set). Each candidate is added on top; 20 x 2-fold stratified halves;
reports mean held-out KS, paired delta vs base, wins out of 40."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
BASE = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabm", "tabpfn", "causilo"]


def ks(y, s):
    return float(ks_2samp(s[y == 1], s[y == 0]).statistic)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_lr(X, y, l2=1.0, iters=30):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w


def load_base():
    folder = ROOT / "results/v8"
    ids = np.load(folder / "train_ids.npy", allow_pickle=True)
    df = pl.DataFrame({"application_id": ids, **{n: np.load(folder / f"oof_{n}.npy") for n in BASE}})
    lab = pl.read_csv(ROOT / "data/train_applications.csv").select("application_id", "target")
    return df.join(lab, on="application_id", validate="1:1").sort("application_id")


def bench(df, base_cols, cands, reps=20):
    y = df["target"].to_numpy()
    cols = {c: logit(df[c].to_numpy()) for c in df.columns if c not in ("application_id", "target")}
    sets = {"base": base_cols, **{k: base_cols + v for k, v in cands.items()}}
    res = {k: [] for k in sets}
    rng = np.random.default_rng(0)
    for r in range(reps):
        half = np.zeros(len(y), bool)
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            half[rng.choice(idx, len(idx) // 2, replace=False)] = True
        for a, b in ((half, ~half), (~half, half)):
            for k, cs in sets.items():
                X = np.column_stack([cols[c] for c in cs])
                mu, sd = X[a].mean(0), X[a].std(0)
                X = (X - mu) / sd
                w = fit_lr(X[a], y[a])
                res[k].append(ks(y[b], X[b] @ w[1:]))
    base = np.array(res["base"])
    print(f"base ({len(base_cols)} models): held-out KS {base.mean():.5f}")
    for k in cands:
        v = np.array(res[k])
        d = v - base
        print(f"+{k:28s} {v.mean():.5f}  delta {d.mean():+.5f}  sd {d.std():.5f}  wins {int((d > 0).sum())}/{len(d)}")
    return res


if __name__ == "__main__":
    df = load_base()
    cands = {}
    for arg in sys.argv[1:]:
        name, spec = arg.split("=", 1)
        path, col = spec.rsplit(":", 1)
        c = pl.read_parquet(ROOT / path).select("application_id", pl.col(col).alias(name))
        df = df.join(c, on="application_id", how="left", validate="1:1")
        assert df[name].null_count() == 0, name
        cands[name] = [name]
    y = df["target"].to_numpy()
    for c in df.columns[1:]:
        if c != "target":
            print(f"{c:12s} OOF KS {ks(y, df[c].to_numpy()):.5f}")
    bench(df, BASE, cands)
