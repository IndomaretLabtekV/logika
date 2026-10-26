"""Inner fold-0 screen (frozen Kumo large, s2026, e8): full fitting context (48k) with
(a) wide 896 pool capped to 128 random columns per estimator, and (b) main232 selection with a
100 vs 150 numeric budget. Pairs: X_wide_cap128_full vs cached W_cap300_ctx28672 (226);
X_main150_full vs X_main100_full. Only inner-fit labels used; 8k inner-stop scored.
Usage: cd eda && ../.venv/bin/python 229_full_context_budget.py <arm>  (one uncached arm per process)
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT
from nested_input_pools import select_columns

OUT = ROOT / "results/v61/full_context_budget_f0"
POOL = ROOT / "results/v60/strict_tfm/pool_inner_f0"
ARMS = {"X_wide_cap128_full": ("wide", 128, None), "X_main100_full": ("main", 500, 100),
        "X_main150_full": ("main", 500, 150)}
PAIRS = [("X_wide_cap128_full", "W_cap300_ctx28672"), ("X_main150_full", "X_main100_full")]


def atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cached = ROOT / "results/v61/width_context_f0/W_cap300_ctx28672.parquet"
    if not (OUT / cached.name).exists():
        shutil.copyfile(cached, OUT / cached.name)
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    tr, _ = v.outer(y, 42)[0]
    fi, st = v.inner(y[tr], 42, 0)
    manifest = json.loads((POOL / "manifest.json").read_text())
    assert manifest["inputs"]["fit_application_ids"] == sorted(ids[tr[fi]].tolist())
    frame = pl.read_parquet(POOL / "strict_features.parquet")
    index = {k: i for i, k in enumerate(frame["application_id"].to_numpy())}
    fit = np.array([index[k] for k in ids[tr[fi]]])
    stop = np.array([index[k] for k in ids[tr[st]]])
    labels = np.full(len(frame), np.nan)
    labels[fit] = y[tr[fi]]
    ystop = y[tr[st]]
    columns, main_cols = manifest["union_columns"], manifest["main_columns"]
    X = frame.select(columns).to_pandas()
    for c in columns:
        if c in v.CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=sorted(X.iloc[fit][c].dropna().unique()))
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    for arm in sys.argv[1:]:
        pp = OUT / f"{arm}.parquet"
        if not pp.exists():
            view, cap, budget = ARMS[arm]
            cols = columns if view == "wide" else select_columns(X[main_cols + [c for c in v.CAT_COLS if c not in main_cols]],
                                                                 labels, fit, n_numeric=budget, extend_shortlist=True)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            from strict_kumo_engine import predict
            p, info = predict(X, fit, stop, labels, cols, 2026, n_estimators=8, max_context=None,
                              column_cap=cap, chunk=128, memory_fraction=.85)
            atomic(pp, lambda t: pl.DataFrame({"application_id": ids[tr[st]], "prediction": p}).write_parquet(t))
            info.pop("context_rows", None)
            info.update(prediction_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(), arm=arm, columns_used=cols)
            atomic(OUT / f"{arm}.json", lambda t: t.write_text(json.dumps(info, indent=2, default=str)))
        p = pl.read_parquet(pp)["prediction"].to_numpy()
        print(arm, {"ks": round(v.ks(ystop, p), 5), "auc": round(roc_auc_score(ystop, p), 5)}, flush=True)
    rng = np.random.default_rng(229)
    pos, neg = np.flatnonzero(ystop == 1), np.flatnonzero(ystop == 0)
    boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500)]
    for a, b in PAIRS:
        if (OUT / f"{a}.parquet").exists() and (OUT / f"{b}.parquet").exists():
            pa, pb = (pl.read_parquet(OUT / f"{n}.parquet")["prediction"].to_numpy() for n in (a, b))
            d = np.array([v.ks(ystop[r], pa[r]) - v.ks(ystop[r], pb[r]) for r in boots])
            print(a, "-", b, json.dumps({"delta": v.ks(ystop, pa) - v.ks(ystop, pb),
                                         "ci95": np.quantile(d, [.025, .975]).tolist(), "p_gt0": float((d > 0).mean())}), flush=True)


if __name__ == "__main__":
    main()
