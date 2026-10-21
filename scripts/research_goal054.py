"""Reproducible five-fold foundation-model research toward pooled KS >= 0.54.

Run in a separate Python 3.11 environment with tabpfn==9.2.0. All data stays
local. No submission is automatically sent to Kaggle. Model/feature/weight
selection reuses the training labels, which introduces selection optimism.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from improve_kumo import definitions, fingerprint, load_frame
from kumo_tabular import EXCLUDED, ROOT, context_indices
from hybrid_kumo import baseline_directory, ids_hash

DEST = ROOT / "artifacts/optimization/goal054"
FOLD_SEED = 31027
SPLITS = 5
WEIGHT_REPO = "Prior-Labs/tabpfn_3_5"
WEIGHT_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
PUBLIC_CHECKPOINT = "submission_kumo_spline_exploratory.csv"
PUBLIC_SHA256 = "3de98287976a7137ad63c946e9932b6ac3ba3a7ee6df607d74cde1aa6a8fc417"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def specifications():
    old = {s["name"]: s for s in definitions()}
    result = []
    for size, features in [("fast", "v5"), ("large", "v5"), ("large", "denoised")]:
        columns = old["baseline_e8" if features == "v5" else "denoised_e8"]["columns"]
        assert not set(columns) & EXCLUDED and len(set(columns)) == len(columns)
        result.append(
            dict(
                name=f"tabpfn35_{size}_{features}_cache_e8_ctx50000",
                family="tabpfn35",
                size=size,
                feature_set=features,
                columns=columns,
                estimators=8,
                context_limit=50000,
                checkpoint=(
                    "tabpfn-v3.5-fast-20260909.safetensors"
                    if size == "fast"
                    else "tabpfn-v3.5-20260909.safetensors"
                ),
            )
        )
    return result


def scores(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return dict(
        ks=float(ks_2samp(p[y == 1], p[y == 0]).statistic),
        auc=float(roc_auc_score(y, p)),
    )


def inputs(specs):
    # Reuse deterministic, label-free feature formulas already documented in 09.
    frame_specs = [dict(columns=s["columns"]) for s in specs]
    data, test = load_frame(frame_specs)
    assert len(data) == 70000 and int(data.target.sum()) == 9507 and len(test) == 30000
    assert data.application_id.is_unique and test.application_id.is_unique
    assert set(data.customer_id).isdisjoint(test.customer_id)
    return data, test


def model_features(frame, columns):
    result = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].fillna("MISSING").astype(str)
    return result


def checkpoint_path(spec):
    from huggingface_hub import hf_hub_download

    source = Path(
        hf_hub_download(
            WEIGHT_REPO, spec["checkpoint"], revision=WEIGHT_REVISION, token=False
        )
    )
    # TabPFN 9.2 resolves symlinks before detecting the checkpoint format.
    # Hub snapshot symlinks resolve to extensionless blob names. Materialize
    # the same immutable bytes with the correct suffix; do not use pickle.
    target = ROOT / "data/weights/tabpfn35" / WEIGHT_REVISION / spec["checkpoint"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(source.resolve(), target)
        except OSError:
            shutil.copyfile(source, target)
    assert not target.is_symlink() and checksum(target) == checksum(source)
    return target


def classifier(spec, path, fold, estimators=None):
    from tabpfn import TabPFNClassifier

    # Explicitly disable automatic ensemble scaling to keep the recipe fixed.
    return TabPFNClassifier(
        model_path=path,
        n_estimators=estimators or spec["estimators"],
        auto_scale_n_estimators=False,
        device="cuda",
        random_state=2026 + fold,
        fit_mode="fit_with_cache",
        keep_cache_on_device=False,
        kv_cache_precision="auto",
        memory_saving_mode=True,
        n_preprocessing_jobs=4,
        inference_precision="autocast",
        eval_metric="roc_auc",
        show_progress_bar=False,
    )


def predict(model, frame, batch_size):
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    assert len(positive) == 1
    parts = []
    for start in range(0, len(frame), batch_size):
        p = model.predict_proba(frame.iloc[start : start + batch_size])[:, positive[0]]
        parts.append(np.asarray(p, dtype=float))
        print(
            "PREDICT", min(start + batch_size, len(frame)), "/", len(frame), flush=True
        )
    result = np.concatenate(parts)
    assert len(result) == len(frame) and np.isfinite(result).all()
    return result


def run(args):
    import torch

    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert checksum(ROOT / PUBLIC_CHECKPOINT) == PUBLIC_SHA256
    specs = [s for s in specifications() if s["name"] in args.candidates]
    assert len(specs) == len(set(args.candidates))
    data, test = inputs(specs)
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(SPLITS, shuffle=True, random_state=FOLD_SEED).split(data, y)
    )
    packages = {
        name: importlib.metadata.version(name)
        for name in [
            "tabpfn",
            "torch",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "skrub",
        ]
    }
    for spec in specs:
        path = checkpoint_path(spec)
        recipe = dict(
            spec=spec,
            splits=SPLITS,
            fold_seed=FOLD_SEED,
            weights_repo=WEIGHT_REPO,
            weights_revision=WEIGHT_REVISION,
            weights_sha256=checksum(path),
            packages=packages,
            training_sha256=fingerprint(
                data, ["application_id", "target"] + spec["columns"]
            ),
            test_sha256=fingerprint(test, ["application_id"] + spec["columns"]),
            batch_size=args.batch_size,
            auto_scale_n_estimators=False,
            fit_mode="fit_with_cache",
            keep_cache_on_device=False,
            kv_cache_precision="auto",
            memory_saving_mode=True,
            inference_precision="autocast",
            eval_metric="roc_auc",
        )
        directory = DEST / f"cv{SPLITS}_seed{FOLD_SEED}" / spec["name"]
        definition = directory / "definition.json"
        if definition.exists() and any(directory.glob("fold*_oof.parquet")):
            assert (
                json.loads(definition.read_text()) == recipe
            ), "Use a new experiment name for a changed recipe."
        write_json(definition, recipe)
        x, xt = model_features(data, spec["columns"]), model_features(
            test, spec["columns"]
        )
        for fold in args.folds if args.folds is not None else range(SPLITS):
            ti, vi = folds[fold]
            oof_file, test_file = (
                directory / f"fold{fold}_oof.parquet",
                directory / f"fold{fold}_test.parquet",
            )
            meta_file = directory / f"fold{fold}_metadata.json"
            if (
                oof_file.exists()
                and meta_file.exists()
                and (not args.test or test_file.exists())
            ):
                meta = json.loads(meta_file.read_text())
                assert (
                    meta["recipe"] == recipe
                    and checksum(oof_file) == meta["oof_sha256"]
                )
                if args.test:
                    assert checksum(test_file) == meta["test_sha256"]
                print("CACHED", spec["name"], fold, meta["metrics"], flush=True)
                continue
            ci = context_indices(ti, y, spec["context_limit"], 2026 + fold)
            assert not len(np.intersect1d(ci, vi)) and set(ci).issubset(set(ti))
            start = time.monotonic()
            print("FIT", spec["name"], "fold", fold, "context", len(ci), flush=True)
            torch.manual_seed(2026 + fold)
            torch.cuda.reset_peak_memory_stats()
            model = classifier(spec, path, fold)
            model.fit(x.iloc[ci], y.iloc[ci])
            print(
                "FIT DONE",
                spec["name"],
                fold,
                "seconds",
                time.monotonic() - start,
                flush=True,
            )
            pv = predict(model, x.iloc[vi], args.batch_size)
            pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[vi].to_numpy(),
                    target=y.iloc[vi].to_numpy(),
                    prediction=pv,
                )
            ).to_parquet(oof_file, index=False)
            pt = predict(model, xt, args.batch_size) if args.test else None
            if pt is not None:
                pd.DataFrame(
                    dict(application_id=test.application_id, prediction=pt)
                ).to_parquet(test_file, index=False)
            from pydantic import TypeAdapter

            fitted_config = model.get_inference_config()
            config = TypeAdapter(type(fitted_config)).dump_python(
                fitted_config, mode="json"
            )
            meta = dict(
                recipe=recipe,
                fold=fold,
                metrics=scores(y.iloc[vi], pv),
                training_rows=len(ti),
                context_rows=len(ci),
                validation_rows=len(vi),
                training_ids_sha256=ids_hash(data, ti),
                validation_ids_sha256=ids_hash(data, vi),
                context_ids_sha256=ids_hash(data, ci),
                oof_sha256=checksum(oof_file),
                test_sha256=checksum(test_file) if pt is not None else None,
                inference_config=config,
                seconds=time.monotonic() - start,
                peak_allocated_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
            write_json(meta_file, meta)
            print(
                "RESULT",
                spec["name"],
                fold,
                json.dumps(meta["metrics"]),
                "seconds",
                meta["seconds"],
                flush=True,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            assert checksum(ROOT / PUBLIC_CHECKPOINT) == PUBLIC_SHA256
    compare()


def read_oof(directory, anchor, folds):
    values = np.full(len(anchor), np.nan)
    for fold, (_, vi) in enumerate(folds):
        p = pd.read_parquet(directory / f"fold{fold}_oof.parquet")
        assert p.application_id.tolist() == anchor.application_id.iloc[vi].tolist()
        assert np.array_equal(p.target.astype(int), anchor.target.iloc[vi].astype(int))
        assert not np.isfinite(values[vi]).any()
        values[vi] = p.prediction.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    return values


def compare():
    specs = specifications()
    data, _ = inputs(specs)
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(SPLITS, shuffle=True, random_state=FOLD_SEED).split(data, y)
    )
    frozen = read_oof(baseline_directory(SPLITS, FOLD_SEED), data, folds)
    spline = read_oof(
        ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
        data,
        folds,
    )
    baseline = 0.75 * frozen + 0.25 * spline

    def row(name, p, **extra):
        m = scores(y, p)
        m.update(
            name=name,
            fold_ks=[scores(y.iloc[vi], p[vi])["ks"] for _, vi in folds],
            **extra,
        )
        m["mean_fold_ks"] = float(np.mean(m["fold_ks"]))
        return m

    base = row("public_056393_recipe", baseline)
    result = [base]
    incomplete = []
    for spec in specs:
        directory = DEST / f"cv{SPLITS}_seed{FOLD_SEED}" / spec["name"]
        if not all(
            (directory / f"fold{i}_metadata.json").exists() for i in range(SPLITS)
        ):
            incomplete.append(spec["name"])
            continue
        p = read_oof(directory, data, folds)
        for weight in [1.0, 0.25, 0.5, 0.75]:
            values = (1 - weight) * baseline + weight * p
            m = row(
                spec["name"] + f"_w{weight}",
                values,
                candidate=spec["name"],
                new_model_weight=weight,
            )
            m.update(
                delta_ks=m["ks"] - base["ks"],
                delta_mean_fold_ks=m["mean_fold_ks"] - base["mean_fold_ks"],
                fold_wins=sum(a > b for a, b in zip(m["fold_ks"], base["fold_ks"])),
                target_reached=m["ks"] >= 0.54,
            )
            result.append(m)
    report = dict(
        protocol=dict(splits=SPLITS, seed=FOLD_SEED, rows=len(data)),
        comparisons=result,
        incomplete=incomplete,
        target_reached=any(r.get("target_reached", False) for r in result),
        caveat="Repeated model selection on previously used training labels and splits introduces optimism. Public score of every changed candidate is unknown.",
    )
    write_json(DEST / "comparison.json", report)
    print("COMPARISON", json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", nargs="+", default=[specifications()[0]["name"]]
    )
    parser.add_argument("--folds", nargs="+", type=int, choices=range(SPLITS))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    compare() if args.compare_only else run(args)


if __name__ == "__main__":
    main()
