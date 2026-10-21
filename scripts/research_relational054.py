"""Evaluate frozen Kumo Relational on disjoint, past-only customer graphs.

Use --preflight for a real-data API/memory check without computing any score.
Full evaluation uses the same five outer folds as the public-checkpoint recipe.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r
from hybrid_kumo import read_data
from preflight_relational054 import REVISION, related, table
from relational_readout054 import install

FEATURES = r.ROOT / "data/processed/relational_goal054"
NAME = "kumo_relational_v5_e8_ctx50000_h1_readout_b8192"


def add_decision(data, test):
    result = []
    for split, frame in [("train", data), ("test", test)]:
        frame = frame.drop(columns="decision_datetime", errors="ignore")
        dates = pd.read_csv(
            r.ROOT / f"data/{split}_applications.csv",
            usecols=["application_id", "customer_id", "decision_datetime"],
            parse_dates=["decision_datetime"],
        )
        joined = frame.merge(
            dates,
            on=["application_id", "customer_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        assert joined.application_id.equals(frame.application_id)
        assert joined.decision_datetime.notna().all()
        result.append(joined)
    return result


def load_inputs():
    data, test, _, columns = read_data()
    data, test = add_decision(data, test)
    manifest = json.loads((FEATURES / "manifest.json").read_text())
    assert not manifest["targets_in_related_tables"]
    assert manifest["ids_are_graph_links_only"]
    frames = {}
    for filename, digest in manifest["output_sha256"].items():
        assert r.checksum(FEATURES / filename) == digest
        frame = pd.read_parquet(FEATURES / filename)
        assert not set(frame) & {"target", "split", "application_id"}
        frames[filename.removesuffix(".parquet")] = frame
    return data, test, columns, frames, manifest


def recipe_for(data, test, columns, manifest):
    from huggingface_hub import hf_hub_download

    from pathlib import Path

    weight = Path(
        hf_hub_download(
            "nvidia/Kumo-Relational", "classifier.pt", revision=REVISION, token=False
        )
    )
    return dict(
        spec=dict(
            name=NAME,
            family="kumo_relational",
            columns=columns,
            context_limit=50000,
            estimators=8,
            estimator_batch_size=1,
            num_hops=1,
            query_batch_size=8192,
        ),
        splits=r.SPLITS,
        fold_seed=r.FOLD_SEED,
        weights_repo="nvidia/Kumo-Relational",
        weights_revision=REVISION,
        weights_sha256=r.checksum(weight),
        training_sha256=r.fingerprint(data, ["application_id", "target"] + columns),
        test_sha256=r.fingerprint(test, ["application_id"] + columns),
        decision_training_sha256=r.fingerprint(
            data, ["application_id", "decision_datetime"]
        ),
        decision_test_sha256=r.fingerprint(
            test, ["application_id", "decision_datetime"]
        ),
        related_manifest_sha256=r.checksum(FEATURES / "manifest.json"),
        related_parquet_sha256=manifest["output_sha256"],
        task_link="customer_id typed id; never a numeric or categorical predictor",
        related_targets=False,
        customer_subgraphs_disjoint=True,
        gnn_backend="Single-hop destination-only readout; all incoming messages retained",
        gnn_backend_sha256=r.checksum(r.ROOT / "scripts/relational_readout054.py"),
        precision="float16 autocast",
        seed="2026+fold",
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "structured-data-models",
                "numpy",
                "pandas",
                "scikit-learn",
            ]
        },
    )


def task_features(frame, columns):
    result = frame[["customer_id", "decision_datetime"] + columns].copy()
    for name in columns:
        if pd.api.types.is_numeric_dtype(result[name]):
            result[name] = result[name].replace([np.inf, -np.inf], np.nan)
        else:
            result[name] = result[name].fillna("MISSING").astype(str)
    assert not set(result) & {"target", "split", "application_id"}
    return result


def predictions(model, frame, columns, frames, batch_size):
    import torch

    parts = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        xq = table(task_features(batch, columns))
        rq = related(frames, batch.customer_id)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model.predict(xq, rq)
            parts.append(output[:, "1"].numerical.reshape(-1).float().cpu().numpy())
        del xq, rq, output
        print(
            "RELATIONAL PREDICT",
            min(start + batch_size, len(frame)),
            "/",
            len(frame),
            flush=True,
        )
    result = np.concatenate(parts)
    assert len(result) == len(frame) and np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    return result


def fit_context(model, data, columns, frames, indices, estimators, seed):
    import sdm
    import torch

    xc = table(task_features(data.iloc[indices], columns))
    yc = sdm.TableTensor.from_pandas(
        pd.DataFrame({"target": data.target.iloc[indices].to_numpy(dtype=int)}),
        stypes={"target": "categorical"},
        device="cuda",
    )
    rc = related(frames, data.customer_id.iloc[indices])
    print("RELATIONAL FIT", len(indices), "estimators", estimators, flush=True)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        model.fit(
            xc,
            yc,
            rc,
            num_estimators=estimators,
            estimator_batch_size=1,
            num_hops=1,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
    del xc, yc, rc


def audit_relational(data, test, fold, ti, vi, meta):
    recipe = meta["recipe"]
    assert recipe["gnn_backend_sha256"] == r.checksum(
        r.ROOT / "scripts/relational_readout054.py"
    )
    parity = json.loads((r.DEST / "relational_readout_parity.json").read_text())
    assert parity["passed"] and parity["backend_sha256"] == recipe["gnn_backend_sha256"]
    data, test = add_decision(data, test)
    for frame, key in [(data, "training"), (test, "test")]:
        assert recipe[f"decision_{key}_sha256"] == r.fingerprint(
            frame, ["application_id", "decision_datetime"]
        )
    assert recipe["related_manifest_sha256"] == r.checksum(FEATURES / "manifest.json")
    manifest = json.loads((FEATURES / "manifest.json").read_text())
    assert recipe["related_parquet_sha256"] == manifest["output_sha256"]
    for filename, digest in recipe["related_parquet_sha256"].items():
        assert r.checksum(FEATURES / filename) == digest
    ci = r.context_indices(ti, data.target.astype(int), 50000, 2026 + fold)
    assert meta["context_ids_sha256"] == r.ids_hash(data, ci)
    assert meta["context_customer_ids_sha256"] == r.fingerprint(
        data.iloc[ci], ["customer_id"]
    )
    assert set(data.customer_id.iloc[ci]).isdisjoint(data.customer_id.iloc[vi])
    assert set(data.customer_id.iloc[ci]).isdisjoint(test.customer_id)
    assert not recipe["related_targets"] and recipe["customer_subgraphs_disjoint"]
    return dict(
        context_rows=len(ci),
        customer_subgraphs_disjoint=True,
        related_targets_absent=True,
        related_file_hashes_verified=True,
    )


def run(preflight=False, selected_folds=None):
    import sdm
    import torch

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.3)
    assert torch.cuda.is_available()
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    data, test, columns, frames, manifest = load_inputs()
    recipe = recipe_for(data, test, columns, manifest)
    directory = r.DEST / "cv5_seed31027" / NAME
    if not preflight:
        definition = directory / "definition.json"
        if definition.exists():
            assert json.loads(definition.read_text()) == recipe
        else:
            r.write_json(definition, recipe)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(
            data, data.target
        )
    )
    model = None
    for fold, (ti, vi) in enumerate(folds):
        if selected_folds is not None and fold not in selected_folds:
            continue
        meta_path = directory / f"fold{fold}_metadata.json"
        if not preflight and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta["recipe"] == recipe
            for kind in ["oof", "test"]:
                assert (
                    r.checksum(directory / f"fold{fold}_{kind}.parquet")
                    == meta[f"{kind}_sha256"]
                )
            continue
        ci = r.context_indices(
            ti, data.target.astype(int), 512 if preflight else 50000, 2026 + fold
        )
        if preflight:
            vi = vi[:8192]
        assert not np.intersect1d(ci, vi).size
        minimum_free_gib = 8 if preflight else 26
        while torch.cuda.mem_get_info()[0] < minimum_free_gib * 1024**3:
            print(f"WAIT RELATIONAL GPU: need {minimum_free_gib} GiB free", flush=True)
            time.sleep(30)
        started = time.monotonic()
        if model is None:
            model = sdm.models.KumoRelational(task="classification", device="cuda")
            parity = json.loads((r.DEST / "relational_readout_parity.json").read_text())
            assert (
                parity["passed"]
                and parity["backend_sha256"] == recipe["gnn_backend_sha256"]
            )
            install(model)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        torch.manual_seed(2026 + fold)
        torch.cuda.reset_peak_memory_stats()
        fit_context(
            model, data, columns, frames, ci, 1 if preflight else 8, 2026 + fold
        )
        pv = predictions(model, data.iloc[vi], columns, frames, 8192)
        if preflight:
            result = dict(
                passed=True,
                context_rows=len(ci),
                query_rows=len(vi),
                estimators=1,
                query_batch_size=8192,
                full_evaluation_recipe_is_separate=True,
                recipe=recipe,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                seconds=time.monotonic() - started,
                validation_scores_computed=False,
            )
            r.write_json(r.DEST / "relational_real_preflight.json", result)
            print("REAL DATA PREFLIGHT PASS", result["seconds"], flush=True)
            return
        pt = predictions(model, test, columns, frames, 8192)
        outputs = dict(
            oof=pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[vi].to_numpy(),
                    target=data.target.iloc[vi].to_numpy(dtype=int),
                    prediction=pv,
                )
            ),
            test=pd.DataFrame(dict(application_id=test.application_id, prediction=pt)),
        )
        for kind, frame in outputs.items():
            frame.to_parquet(directory / f"fold{fold}_{kind}.parquet", index=False)
        meta = dict(
            recipe=recipe,
            fold=fold,
            training_ids_sha256=r.ids_hash(data, ti),
            context_ids_sha256=r.ids_hash(data, ci),
            context_customer_ids_sha256=r.fingerprint(data.iloc[ci], ["customer_id"]),
            validation_ids_sha256=r.ids_hash(data, vi),
            training_rows=len(ti),
            context_rows=len(ci),
            validation_rows=len(vi),
            metrics=r.scores(data.target.iloc[vi], pv),
            seconds=time.monotonic() - started,
            peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
            **{
                kind + "_sha256": r.checksum(directory / f"fold{fold}_{kind}.parquet")
                for kind in outputs
            },
        )
        audit_relational(data, test, fold, ti, vi, meta)
        r.write_json(meta_path, meta)
        print(
            "RELATIONAL RESULT",
            fold,
            json.dumps(meta["metrics"]),
            meta["seconds"],
            flush=True,
        )
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5))
    args = parser.parse_args()
    run(args.preflight, args.folds)
