"""Inner fold-0 screen: memory-bounded ensembling. sdm's fit(num_estimators=E) keeps one KV cache per
estimator until predict, so peak memory grows with E and context rows. Here E sequential e1 fits
(generator seed = 2026*1000 + b), probabilities averaged; peak = one estimator. With max_context
set, each member also draws its own class-proportional context subset (declared deviation).
Pairs: B_cap300_ctx28k_8x1 vs cached e8 W_cap300_ctx28672 (226); B_cap128_full_8x1 (48k context,
OOM as e8 in 229) vs the same e8 arm. Only inner-fit labels; 8k inner-stop scored.
Usage: cd eda && ../.venv/bin/python 231_sequential_estimators.py <arm>
"""
import hashlib
import json
import os
import shutil
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT
from strict_kumo_engine import predict

OUT = ROOT / "results/v61/sequential_estimators_f0"
POOL = ROOT / "results/v60/strict_tfm/pool_inner_f0"
ARMS = {"B_cap300_ctx28k_8x1": (300, 28672, 8), "B_cap128_full_8x1": (128, None, 8),
        "B_cap300_ctx28k_24x1": (300, 28672, 24)}  # members 0-7 identical to the 8x1 arm (same seeds), copied
BASE = "W_cap300_ctx28672"


def atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def _npy(array):
    import io
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / f"results/v61/width_context_f0/{BASE}.parquet"
    if not (OUT / src.name).exists():
        shutil.copyfile(src, OUT / src.name)
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
    columns = manifest["union_columns"]
    X = frame.select(columns).to_pandas()
    for c in columns:
        if c in v.CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=sorted(X.iloc[fit][c].dropna().unique()))
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    for arm in sys.argv[1:]:
        pp = OUT / f"{arm}.parquet"
        if not pp.exists():
            cap, ctx, members = ARMS[arm]
            parts, peaks = [], []
            for b in range(members):
                mp = OUT / f"{arm}_m{b}.npy"  # per-member cache: resumable after a crash
                if not mp.exists():
                    p, info = predict(X, fit, stop, labels, columns, 2026 * 1000 + b, n_estimators=1,
                                      max_context=ctx, column_cap=cap, chunk=128, memory_fraction=.85)
                    atomic(mp, lambda t: t.write_bytes(_npy(p)))
                    peaks.append(info["max_allocated_gib"])
                    print(arm, "member", b, "peak GiB", round(info["max_allocated_gib"], 2), flush=True)
                parts.append(np.load(mp))
            p = np.mean(parts, axis=0)
            atomic(pp, lambda t: pl.DataFrame({"application_id": ids[tr[st]], "prediction": p}).write_parquet(t))
            atomic(OUT / f"{arm}.json", lambda t: t.write_text(json.dumps(
                {"arm": arm, "cap": cap, "max_context": ctx, "members": members, "peak_gib": peaks,
                 "prediction_sha256": hashlib.sha256(pp.read_bytes()).hexdigest()}, indent=2)))
        p = pl.read_parquet(pp)["prediction"].to_numpy()
        print(arm, {"ks": round(v.ks(ystop, p), 5), "auc": round(roc_auc_score(ystop, p), 5)}, flush=True)
    base = pl.read_parquet(OUT / f"{BASE}.parquet")
    assert base["application_id"].to_list() == ids[tr[st]].tolist()
    pb = base["prediction"].to_numpy()
    rng = np.random.default_rng(231)
    pos, neg = np.flatnonzero(ystop == 1), np.flatnonzero(ystop == 0)
    boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500)]
    for arm in ARMS:
        if (OUT / f"{arm}.parquet").exists():
            pa = pl.read_parquet(OUT / f"{arm}.parquet")["prediction"].to_numpy()
            d = np.array([v.ks(ystop[r], pa[r]) - v.ks(ystop[r], pb[r]) for r in boots])
            print(arm, "-", BASE, json.dumps({"delta": v.ks(ystop, pa) - v.ks(ystop, pb),
                                              "ci95": np.quantile(d, [.025, .975]).tolist(),
                                              "p_gt0": float((d > 0).mean())}), flush=True)


if __name__ == "__main__":
    main()
