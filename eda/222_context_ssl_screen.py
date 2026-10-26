"""Inner fold-0 screening of C2 (max context for the wide cap500 view) and S3 (sampled pseudo-label
context), per docs/integration/protokol_context_ssl_24okt.md. Frozen Kumo inference only.

Only inner-fit labels (48k) are used; the 8k inner-stop rows are scored; outer rows are never
predicted. Test covariates are used transductively by S3 (declared). Cached per arm, fsync+rename.
Usage: cd eda && ../.venv/bin/python 222_context_ssl_screen.py [arm ...]
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT
from nested_input_pools import select_columns
from strict_kumo_engine import predict

OUT = ROOT / "results/v61/context_ssl_screen_f0"
POOL = ROOT / "results/v60/strict_tfm/pool_inner_f0"
SEED, E = 2026, 8
ARMS = {  # name: (view, max_context, memory_fraction, context kind); M_* = fit-only median impute
    "C2_ctx8192": ("wide", 8192, .7, "fit"),
    "C2_ctx16384": ("wide", 16384, .85, "fit"),  # 24576 needs ~8 GB row buffer: infeasible on 8 GB
    "M_ctx8192": ("wide_median", 8192, .7, "fit"),
    "M_ctx16384": ("wide_median", 16384, .85, "fit"),
    "M_small": ("small_median", None, .7, "fit"),
    "S3_base_teacher": ("small", None, .7, "fit"),
    "S3_pseudo_d0": ("small", None, .85, "pseudo0"),
    "S3_pseudo_d1": ("small", None, .85, "pseudo1"),
    "S3_permuted_d0": ("small", None, .85, "perm0"),
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    tids = pl.read_csv(ROOT / "data/sample_submission.csv")["application_id"].to_numpy()
    test = np.array([index[k] for k in tids])
    labels = np.full(len(frame), np.nan)
    labels[fit] = y[tr[fi]]
    assert np.isnan(frame["target"].to_numpy()[stop].astype(float)).all()
    columns = manifest["union_columns"]
    X = frame.select(columns).to_pandas()
    for c in columns:
        if c in v.CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=sorted(X.iloc[fit][c].dropna().unique()))
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    small = select_columns(X, labels, fit, extend_shortlist=True)
    assert len(small) == 104
    run = {"pool_manifest_sha256": sha(POOL / "manifest.json"), "seed": SEED, "estimators": E,
           "source_sha256": sha(__file__), "engine_sha256": sha(ROOT / "eda/strict_kumo_engine.py"),
           "small_columns": small, "arms": ARMS}
    mp = OUT / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == json.loads(json.dumps(run)), "changed run; use a new folder"
    else:
        atomic(mp, lambda t: t.write_text(json.dumps(run, indent=2)))
    ystop = y[tr[st]]
    results = {}
    for arm in sys.argv[1:] or ARMS:
        view, max_context, fraction, kind = ARMS[arm]
        pp = OUT / f"{arm}.parquet"
        if not pp.exists():
            arm_labels, context, query = labels.copy(), fit, stop
            if arm == "S3_base_teacher":
                query = np.r_[stop, test]  # same fit also serves as the teacher for test rows
            if kind != "fit":
                teacher = pl.read_parquet(OUT / "S3_base_teacher.parquet")["prediction"].to_numpy()[len(stop):]
                draw = int(kind[-1])
                pseudo = (np.random.default_rng(222 + draw).random(len(test)) < teacher).astype(float)
                if kind.startswith("perm"):
                    pseudo = np.random.default_rng(999).permutation(pseudo)
                arm_labels[test] = pseudo
                context = np.r_[fit, test]
            # read once at CUDA init: run one uncached arm per process (see run_222 order)
            if fraction > .7:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            Xa = X
            if view.endswith("_median"):  # medians from the actual labelled fitting rows only
                Xa = X.copy()
                for c in columns:
                    if c not in v.CAT_COLS:
                        Xa[c] = X[c].fillna(X[c].iloc[fit].median())
            p, info = predict(Xa, context, query, arm_labels, small if view.startswith("small") else columns, SEED,
                              n_estimators=E, max_context=max_context, column_cap=500,
                              chunk=128, memory_fraction=fraction)
            atomic(pp, lambda t: pl.DataFrame({"application_id": frame["application_id"].to_numpy()[query],
                                               "prediction": p}).write_parquet(t))
            info.pop("context_rows", None)
            info.update(prediction_sha256=sha(pp), manifest_sha256=sha(mp))
            atomic(OUT / f"{arm}.json", lambda t: t.write_text(json.dumps(info, indent=2, default=str)))
        p = pl.read_parquet(pp)["prediction"].to_numpy()[:len(stop)]
        results[arm] = p
        print(arm, {"ks": round(v.ks(ystop, p), 5), "auc": round(roc_auc_score(ystop, p), 5)}, flush=True)
    rng = np.random.default_rng(222)
    boots = [np.r_[rng.choice(np.flatnonzero(ystop == 1), int(ystop.sum())),
                   rng.choice(np.flatnonzero(ystop == 0), int((1 - ystop).sum()))] for _ in range(1000)]
    pairs = [("C2_ctx16384", "C2_ctx8192"), ("M_ctx8192", "C2_ctx8192"), ("M_ctx16384", "C2_ctx16384"),
             ("M_small", "S3_base_teacher"), ("S3_pseudo_mean", "S3_base_teacher"), ("S3_pseudo_mean", "S3_permuted_d0")]
    if "S3_pseudo_d0" in results and "S3_pseudo_d1" in results:
        from scipy.stats import rankdata
        results["S3_pseudo_mean"] = (rankdata(results["S3_pseudo_d0"]) + rankdata(results["S3_pseudo_d1"])) / 2
        print("S3_pseudo_mean", {"ks": round(v.ks(ystop, results["S3_pseudo_mean"]), 5)})
    summary = {}
    for a, b in pairs:
        if a in results and b in results:
            d = np.array([v.ks(ystop[r], results[a][r]) - v.ks(ystop[r], results[b][r]) for r in boots])
            summary[f"{a}-{b}"] = {"delta": v.ks(ystop, results[a]) - v.ks(ystop, results[b]),
                                   "ci95": np.quantile(d, [.025, .975]).tolist(), "p_gt0": float((d > 0).mean())}
            print(a, "-", b, json.dumps(summary[f"{a}-{b}"]), flush=True)
    atomic(OUT / "summary.json", lambda t: t.write_text(json.dumps(summary, indent=2)))


if __name__ == "__main__":
    main()
