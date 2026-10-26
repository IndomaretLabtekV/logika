"""Inner fold-0 screen: anchored union view. Random cap300 subsets drop most strong main features per
estimator; here every member keeps the 104 main_small columns (main232 selector, fitting labels only)
plus 196 random union columns (member-specific seed), context 28672, 8 sequential e1 members.
Pair: vs cached e8 W_cap300_ctx28672 (226). Only inner-fit labels; 8k inner-stop scored.
Usage: cd eda && ../.venv/bin/python 232_anchored_union_view.py
"""
import importlib.util
import json
import os
import shutil

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT
from nested_input_pools import select_columns

spec = importlib.util.spec_from_file_location("seq231", ROOT / "eda/231_sequential_estimators.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
OUT = ROOT / "results/v61/anchored_union_f0"
ARM, BASE, CTX, EXTRA, MEMBERS = "Y_anchor104_rand196_ctx28k_8x1", "W_cap300_ctx28672", 28672, 196, 8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / f"results/v61/width_context_f0/{BASE}.parquet"
    if not (OUT / src.name).exists():
        shutil.copyfile(src, OUT / src.name)
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
    tr, _ = v.outer(y, 42)[0]
    fi, st = v.inner(y[tr], 42, 0)
    manifest = json.loads((s.POOL / "manifest.json").read_text())
    assert manifest["inputs"]["fit_application_ids"] == sorted(ids[tr[fi]].tolist())
    frame = pl.read_parquet(s.POOL / "strict_features.parquet")
    index = {k: i for i, k in enumerate(frame["application_id"].to_numpy())}
    fit = np.array([index[k] for k in ids[tr[fi]]])
    stop = np.array([index[k] for k in ids[tr[st]]])
    labels = np.full(len(frame), np.nan)
    labels[fit] = y[tr[fi]]
    ystop = y[tr[st]]
    columns, main_cols = manifest["union_columns"], manifest["main_columns"]
    import pandas as pd
    X = frame.select(columns).to_pandas()
    for c in columns:
        if c in v.CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=sorted(X.iloc[fit][c].dropna().unique()))
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    anchor = select_columns(X[main_cols], labels, fit, extend_shortlist=True)
    assert len(anchor) == 104
    rest = [c for c in columns if c not in set(anchor) and c not in v.CAT_COLS]
    pp = OUT / f"{ARM}.parquet"
    if not pp.exists():
        parts = []
        for b in range(MEMBERS):
            mp = OUT / f"{ARM}_m{b}.npy"
            if not mp.exists():
                extra = list(np.random.default_rng(232_000 + b).choice(rest, EXTRA, replace=False))
                p, info = s.predict(X, fit, stop, labels, anchor + extra, 2026 * 1000 + b, n_estimators=1,
                                    max_context=CTX, column_cap=500, chunk=128, memory_fraction=.85)
                s.atomic(mp, lambda t: t.write_bytes(s._npy(p)))
                print(ARM, "member", b, "peak GiB", round(info["max_allocated_gib"], 2), flush=True)
            parts.append(np.load(mp))
        p = np.mean(parts, axis=0)
        s.atomic(pp, lambda t: pl.DataFrame({"application_id": ids[tr[st]], "prediction": p}).write_parquet(t))
    p = pl.read_parquet(pp)["prediction"].to_numpy()
    pb = pl.read_parquet(OUT / f"{BASE}.parquet")
    assert pb["application_id"].to_list() == ids[tr[st]].tolist()
    pb = pb["prediction"].to_numpy()
    print(ARM, {"ks": round(v.ks(ystop, p), 5), "auc": round(roc_auc_score(ystop, p), 5)}, flush=True)
    rng = np.random.default_rng(232)
    pos, neg = np.flatnonzero(ystop == 1), np.flatnonzero(ystop == 0)
    d = []
    for _ in range(500):
        r = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        d.append(v.ks(ystop[r], p[r]) - v.ks(ystop[r], pb[r]))
    print(ARM, "-", BASE, json.dumps({"delta": v.ks(ystop, p) - v.ks(ystop, pb),
                                      "ci95": np.quantile(d, [.025, .975]).tolist(), "p_gt0": float(np.mean(np.array(d) > 0))}), flush=True)


if __name__ == "__main__":
    main()
