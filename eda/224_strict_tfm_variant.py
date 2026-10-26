"""Variant of 212 for amended views (fit-only median impute, per-view context/memory); never submits.

Protocol JSON views may add "impute": "median" and "memory_fraction". No baseline member."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT
from nested_input_pools import select_columns
from strict_feature_pool import build_pool
from strict_kumo_engine import predict, technical_preflight

PROTOCOL = ROOT / "artifacts/optimization/continuation_23okt/protocol.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    fsync_replace(temp, path)


def fsync_replace(temp, path):
    with open(temp, "rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metrics(y, p):
    assert len(y) == len(p) and np.isfinite(p).all()
    return {"ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
            "auc": float(roc_auc_score(y, p))}


def rank(p):
    return rankdata(p, method="average") / len(p)


def run(args):
    protocol = json.loads(args.protocol.read_text())
    out = args.out.resolve()
    folder = out / f"{args.stage}_f{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)
    train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
    ids = train["application_id"].to_numpy()
    y = train["target"].to_numpy().astype(int)
    tr, va = v.outer(y, protocol["outer_seed"])[args.fold]
    fi, st = v.inner(y[tr], protocol["outer_seed"], args.fold)
    fit_indices = tr if args.stage == "outer" else tr[fi]
    query_indices = va if args.stage == "outer" else tr[st]
    pool_root = (args.pool_root or out).resolve()
    pool = build_pool(ids[fit_indices].tolist(), pool_root / f"pool_{'outer' if args.stage == 'outer' else 'inner'}_f{args.fold}")
    frame = pl.read_parquet(pool["parquet"])
    all_ids = frame["application_id"].to_numpy()
    index = {key: i for i, key in enumerate(all_ids)}
    fit_rows = np.array([index[key] for key in ids[fit_indices]])
    query_rows = np.array([index[key] for key in ids[query_indices]])
    test_ids = pl.read_csv(ROOT / "data/sample_submission.csv")["application_id"].to_numpy()
    test_rows = np.array([index[key] for key in test_ids]) if args.stage == "outer" else np.array([], dtype=int)
    labels = np.full(len(frame), np.nan)
    labels[fit_rows] = y[fit_indices]
    assert frame.filter(~pl.col("application_id").is_in(ids[fit_indices].tolist()))["target"].null_count() == len(frame) - len(fit_rows)
    columns = pool["manifest"]["union_columns"]
    main = pool["manifest"]["main_columns"]
    X = frame.select(columns).to_pandas()
    for c in columns:
        if c in v.CAT_COLS:
            categories = sorted(X.iloc[fit_rows][c].dropna().unique())
            X[c] = pd.Categorical(X[c], categories=categories)
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    small = select_columns(X, labels, fit_rows, extend_shortlist=True)
    base_columns = select_columns(X[main], labels, fit_rows, extend_shortlist=True)
    assert len(small) == len(base_columns) == 104
    specs = []
    for view in protocol["candidate_views"]:
        for seed in protocol["model_seeds"]:
            specs.append({"name": f"{view['name']}_s{seed}",
                          "columns": {"small": small, "main_small": base_columns, "all": columns,
                                      "main_150": select_columns(X[main], labels, fit_rows, n_numeric=150, extend_shortlist=True) if view["columns"] == "main_150" else None}[view["columns"]],
                          "model_seed": seed, "n_estimators": protocol["internal_estimators"],
                          "column_cap": view["column_cap"], "max_context": view["max_context"],
                          "impute": view.get("impute"), "memory_fraction": view.get("memory_fraction", .7)})
    sources = [Path(__file__), ROOT / "eda/212_strict_tfm_views.py", ROOT / "eda/strict_feature_pool.py", ROOT / "eda/strict_kumo_engine.py",
               ROOT / "eda/features.py", ROOT / "eda/valid2.py", ROOT / "eda/nested_input_pools.py"]
    request = {"protocol_sha256": sha(args.protocol), "protocol_path": str(args.protocol.resolve()),
               "pool_root": str(pool_root), "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
               "pool_manifest_sha256": sha(pool["manifest_path"]), "stage": args.stage, "fold": args.fold,
               "outer_seed": protocol["outer_seed"], "fit_ids": ids[fit_indices].tolist(),
               "query_ids": ids[query_indices].tolist(), "test_ids": test_ids.tolist() if args.stage == "outer" else [],
               "specs": specs, "smoke_queries": args.smoke_queries if args.stage == "preflight" else None}
    mp = folder / "manifest.json"
    if mp.exists():
        assert json.loads(mp.read_text()) == request, "Changed recipe/source/data; use a new output directory"
    else:
        save(mp, request)
    all_queries = np.r_[query_rows, test_rows]
    results = {}
    for spec in specs:
        name = spec["name"]
        if args.member and name != args.member:
            continue
        kwargs = {k: value for k, value in spec.items() if k not in ("name", "impute")}
        kwargs.update(size="large", chunk=128)
        if kwargs["memory_fraction"] > .7:  # read once at CUDA init; one member per process
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        Xm = X
        if spec["impute"] == "median":  # medians from labelled fitting rows of this partition only
            Xm = X.copy()
            for c in spec["columns"]:
                if c not in v.CAT_COLS:
                    Xm[c] = X[c].fillna(X[c].iloc[fit_rows].median())
        pp = folder / f"{name}.parquet"
        cp = folder / f"{name}.json"
        if args.stage == "preflight":
            if cp.exists():
                saved = json.loads(cp.read_text())
                assert saved["manifest_sha256"] == sha(mp)
                print("CACHED PREFLIGHT", name, flush=True)
                continue
            try:
                info = technical_preflight(Xm, fit_rows, query_rows, labels,
                                           query_count=args.smoke_queries, **kwargs)
            except Exception as exc:
                save(folder / f"{name}_failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                      "metadata": getattr(exc, "metadata", None),
                                                      "manifest_sha256": sha(mp), "quality_measured": False})
                raise
            save(cp, {"manifest_sha256": sha(mp), "metadata": info, "quality_measured": False})
            print("PREFLIGHT", name, json.dumps({k: info.get(k) for k in
                  ("status", "context_count", "input_column_count", "n_estimators",
                   "max_allocated_gib", "max_reserved_gib")}), flush=True)
            continue
        if cp.exists():
            saved = json.loads(cp.read_text())
            assert saved["manifest_sha256"] == sha(mp) and saved["prediction_sha256"] == sha(pp)
            d = pl.read_parquet(pp)
            assert d["application_id"].to_list() == all_ids[all_queries].tolist()
            p = d["prediction"].to_numpy()
            print("CACHED", name, flush=True)
        else:
            p, info = predict(Xm, fit_rows, all_queries, labels, **kwargs)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            temp = pp.with_suffix(".parquet.tmp")
            pl.DataFrame({"application_id": all_ids[all_queries], "prediction": p}).write_parquet(temp)
            fsync_replace(temp, pp)
            save(cp, {"manifest_sha256": sha(mp), "prediction_sha256": sha(pp), "metadata": info})
        results[name] = p
        print(name, metrics(y[query_indices], p[:len(query_rows)]), flush=True)
        gc.collect()
    return  # evaluation by eda/225 (no baseline member here)
    n = len(query_rows)
    members = [spec["name"] for spec in specs if spec["name"] != "baseline"]
    candidate = np.mean([rank(results[name][:n]) for name in members], axis=0)
    baseline = rank(results["baseline"][:n])
    pred = pl.DataFrame({"application_id": ids[query_indices], "target": y[query_indices],
                         "fold": np.full(n, args.fold), "candidate": candidate, "baseline": baseline})
    pred.write_parquet(folder / "evaluation.parquet")
    if args.stage == "outer":
        test_prediction = np.mean([rank(results[name][n:]) for name in members], axis=0)
        pl.DataFrame({"application_id": test_ids, "candidate": test_prediction,
                      "baseline": rank(results["baseline"][n:])}).write_parquet(folder / "test.parquet")
    report = {"stage": args.stage, "rows": n, "candidate": metrics(y[query_indices], candidate),
              "baseline": metrics(y[query_indices], baseline), "model_members": members,
              "manifest_sha256": sha(mp), "evaluation_sha256": sha(folder / "evaluation.parquet"),
              "submission_eligible": False}
    save(folder / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--pool-root", type=Path, help="Reuse separately fingerprinted fitting-partition feature pools")
    ap.add_argument("--stage", choices=["preflight", "inner", "outer"], required=True)
    ap.add_argument("--fold", type=int, choices=range(5), default=0)
    ap.add_argument("--member")
    ap.add_argument("--smoke-queries", type=int, default=8)
    args = ap.parse_args()
    if args.smoke_queries < 1:
        ap.error("smoke query count must be positive")
    run(args)


if __name__ == "__main__":
    main()
