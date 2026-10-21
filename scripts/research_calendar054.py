"""Training-only calendar effects transferred from spline to hybrid log odds.

Calendar hypotheses came from descriptive OOF diagnostics, so research selection
is optimistic. Every coefficient here is nevertheless fitted using outer-training
labels only. The existing spline used as training offset also excludes outer val.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

import research_goal054 as r
from diagnose_calendar054 import calendar_features
from hybrid_kumo import read_data, predict_spline
from research_context_correction054 import objective

PLAN = r.DEST / "calendar_transfer_plan.json"
BASE = r.DEST / "calendar_transfer_base/cv5_seed31027"
STAGE = r.DEST / "cv5_seed31027"


def add_calendar(data, test):
    frames = []
    for frame, split in [(data, "train"), (test, "test")]:
        apps = pd.read_csv(r.ROOT / f"data/{split}_applications.csv")
        features = calendar_features(apps)
        result = frame.merge(features, on="application_id", validate="one_to_one")
        assert result.application_id.equals(frame.application_id)
        frames.append(result)
    return tuple(frames)


def specifications():
    plan = json.loads(PLAN.read_text())
    return [
        dict(
            name=f"calendar_transfer_l2{str(penalty).replace('.','p')}_{balance}_d{str(strength).replace('.','p')}",
            family="calendar_transfer",
            penalty=penalty,
            balance=balance,
            correction_strength=strength,
            calendar_columns=plan["calendar_columns"],
        )
        for penalty in plan["penalties"]
        for balance in ["unweighted", "balanced"]
        for strength in plan["correction_strengths"]
    ]


def fit(spec, calendar, y, spline_probability):
    encoder = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float64
    )
    x = np.column_stack(
        [
            np.ones(len(calendar)),
            encoder.fit_transform(calendar[spec["calendar_columns"]]),
        ]
    )
    offset = logit(np.clip(np.asarray(spline_probability, dtype=float), 1e-6, 1 - 1e-6))
    y = np.asarray(y, dtype=int)
    ratio = (
        float((y == 0).sum() / (y == 1).sum()) if spec["balance"] == "balanced" else 1.0
    )
    weight = np.where(y == 1, ratio, 1.0)
    with threadpool_limits(limits=4):
        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            args=(x, offset, y, weight, spec["penalty"], np.log(ratio)),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=300, ftol=1e-11, gtol=1e-7),
        )
    value, gradient = objective(
        result.x, x, offset, y, weight, spec["penalty"], np.log(ratio)
    )
    assert result.success and np.max(np.abs(gradient)) < 1e-4
    return dict(
        spec=spec,
        encoder=encoder,
        coefficients=result.x,
        positive_weight=ratio,
        objective=value,
        gradient_max_abs=float(np.max(np.abs(gradient))),
        iterations=int(result.nit),
    )


def predict(bundle, frame, hybrid_probability):
    x = np.column_stack(
        [
            np.ones(len(frame)),
            bundle["encoder"].transform(frame[bundle["spec"]["calendar_columns"]]),
        ]
    )
    correction = x @ bundle["coefficients"]
    offset = logit(np.clip(np.asarray(hybrid_probability, dtype=float), 1e-6, 1 - 1e-6))
    return (
        expit(offset + bundle["spec"]["correction_strength"] * correction),
        correction,
    )


def source_training_prediction(data, spline_spec, fold, ti, vi):
    directory = (
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline"
    )
    source_meta_file = directory / f"fold{fold}_metadata.json"
    source_meta = json.loads(source_meta_file.read_text())
    assert source_meta["training_ids_sha256"] == r.ids_hash(data, ti)
    assert source_meta["validation_ids_sha256"] == r.ids_hash(data, vi)
    assert source_meta["recipe"]["training_sha256"] == r.fingerprint(
        data, ["application_id", "target"] + spline_spec["columns"]
    )
    model_file = directory / f"fold{fold}_model.joblib"
    assert r.checksum(model_file) == source_meta["model_sha256"]
    path = BASE / f"fold{fold}_training.parquet"
    meta_path = BASE / f"fold{fold}_metadata.json"
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta["source_model_sha256"] == r.checksum(model_file) and meta[
            "training_prediction_sha256"
        ] == r.checksum(path)
        frame = pd.read_parquet(path)
        assert frame.application_id.tolist() == data.application_id.iloc[ti].tolist()
        return frame, meta
    model = joblib.load(model_file)
    print("SPLINE TRAINING OFFSET", fold, len(ti), flush=True)
    probability = predict_spline(model, data.iloc[ti])
    frame = pd.DataFrame(
        dict(
            application_id=data.application_id.iloc[ti].to_numpy(),
            target=data.target.iloc[ti].to_numpy(dtype=int),
            prediction=probability,
        )
    )
    BASE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    meta = dict(
        fold=fold,
        training_ids_sha256=r.ids_hash(data, ti),
        validation_ids_sha256=r.ids_hash(data, vi),
        source_model_sha256=r.checksum(model_file),
        source_metadata_sha256=r.checksum(source_meta_file),
        source_training_features_sha256=source_meta["recipe"]["training_sha256"],
        training_prediction_sha256=r.checksum(path),
        rows=len(ti),
        note="In-sample spline probabilities used only for another training stage. Both stages exclude outer-validation labels. No OOF probabilities from other outer folds are training inputs.",
    )
    r.write_json(meta_path, meta)
    return frame, meta


def run():
    data, test, spline_spec, kumo_columns = read_data()
    data, test = add_calendar(data, test)
    specs = specifications()
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    spline_dir = (
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline"
    )
    kumo_dir = r.baseline_directory(r.SPLITS, r.FOLD_SEED)
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    for fold, (ti, vi) in enumerate(folds):
        training, source_meta = source_training_prediction(
            data, spline_spec, fold, ti, vi
        )
        assert np.array_equal(training.target, y.iloc[ti])
        source_paths = {}
        hybrid = {}
        for kind, frame in [("oof", data.iloc[vi]), ("test", test)]:
            kp, sp = [
                directory / f"fold{fold}_{kind}.parquet"
                for directory in [kumo_dir, spline_dir]
            ]
            k, s = pd.read_parquet(kp), pd.read_parquet(sp)
            assert (
                k.application_id.tolist() == frame.application_id.tolist()
                and s.application_id.equals(k.application_id)
            )
            hybrid[kind] = 0.75 * k.prediction.to_numpy(
                dtype=float
            ) + 0.25 * s.prediction.to_numpy(dtype=float)
            source_paths[kind] = dict(
                kumo_path=str(kp.relative_to(r.ROOT)),
                kumo_sha256=r.checksum(kp),
                spline_path=str(sp.relative_to(r.ROOT)),
                spline_sha256=r.checksum(sp),
            )
        fitted = {}
        for spec in specs:
            directory = STAGE / spec["name"]
            columns = kumo_columns + spec["calendar_columns"]
            recipe = dict(
                **spec,
                columns=columns,
                n_splits=r.SPLITS,
                fold_seed=r.FOLD_SEED,
                training_sha256=r.fingerprint(
                    data, ["application_id", "target"] + columns
                ),
                test_sha256=r.fingerprint(test, ["application_id"] + columns),
                source_spline_columns=spline_spec["columns"],
                parameter_plan_sha256=r.checksum(PLAN),
                packages={
                    p: importlib.metadata.version(p)
                    for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
                },
            )
            definition = directory / "definition.json"
            if definition.exists():
                assert json.loads(definition.read_text()) == recipe
            else:
                r.write_json(definition, recipe)
            paths = {
                kind: directory / f"fold{fold}_{kind}.parquet"
                for kind in ["oof", "test"]
            }
            meta_path = directory / f"fold{fold}_metadata.json"
            model_path = directory / f"fold{fold}_model.joblib"
            if (
                meta_path.exists()
                and all(p.exists() for p in paths.values())
                and model_path.exists()
            ):
                meta = json.loads(meta_path.read_text())
                assert meta["recipe"] == recipe
                continue
            key = (spec["penalty"], spec["balance"])
            if key not in fitted:
                fitted[key] = fit(spec, data.iloc[ti], y.iloc[ti], training.prediction)
            bundle = {**fitted[key], "spec": spec}
            for kind, frame in [("oof", data.iloc[vi]), ("test", test)]:
                probability, correction = predict(bundle, frame, hybrid[kind])
                output = (
                    frame[["application_id", "target"]].copy()
                    if kind == "oof"
                    else frame[["application_id"]].copy()
                )
                output["prediction"], output["correction_logit"] = (
                    probability,
                    correction,
                )
                output.to_parquet(paths[kind], index=False)
            joblib.dump(bundle, model_path, compress=3)
            oof = pd.read_parquet(paths["oof"])
            meta = dict(
                recipe=recipe,
                fold=fold,
                metrics=r.scores(y.iloc[vi], oof.prediction),
                training_rows=len(ti),
                validation_rows=len(vi),
                training_ids_sha256=r.ids_hash(data, ti),
                validation_ids_sha256=r.ids_hash(data, vi),
                training_offset_metadata=str(
                    (BASE / f"fold{fold}_metadata.json").relative_to(r.ROOT)
                ),
                training_offset_metadata_sha256=r.checksum(
                    BASE / f"fold{fold}_metadata.json"
                ),
                prediction_sources=source_paths,
                model_sha256=r.checksum(model_path),
                **{kind + "_sha256": r.checksum(path) for kind, path in paths.items()},
                optimization_objective=bundle["objective"],
                optimization_gradient_max_abs=bundle["gradient_max_abs"],
            )
            r.write_json(meta_path, meta)
            print(
                "CALENDAR", spec["name"], fold, json.dumps(meta["metrics"]), flush=True
            )
        assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    from report_goal054 import audit, report_markdown

    report_markdown(audit())


def audit_calendar(data, test, fold, ti, vi, directory, meta):
    spec = meta["recipe"]
    assert not len(np.intersect1d(ti, vi)) and len(ti) == 56000 and len(vi) == 14000
    source_meta_file = r.ROOT / meta["training_offset_metadata"]
    assert r.checksum(source_meta_file) == meta["training_offset_metadata_sha256"]
    sm = json.loads(source_meta_file.read_text())
    assert sm["training_ids_sha256"] == r.ids_hash(data, ti) and sm[
        "validation_ids_sha256"
    ] == r.ids_hash(data, vi)
    spline_meta_file = (
        r.ROOT
        / f"artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline/fold{fold}_metadata.json"
    )
    assert r.checksum(spline_meta_file) == sm["source_metadata_sha256"]
    spm = json.loads(spline_meta_file.read_text())
    assert spm["training_ids_sha256"] == r.ids_hash(data, ti) and spm[
        "validation_ids_sha256"
    ] == r.ids_hash(data, vi)
    training_path = source_meta_file.parent / f"fold{fold}_training.parquet"
    assert r.checksum(training_path) == sm["training_prediction_sha256"]
    training = pd.read_parquet(training_path)
    assert training.application_id.tolist() == data.application_id.iloc[ti].tolist()
    assert np.array_equal(training.target, data.target.iloc[ti].astype(int))
    model_path = directory / f"fold{fold}_model.joblib"
    assert r.checksum(model_path) == meta["model_sha256"]
    bundle = joblib.load(model_path)
    assert all(spec[key] == value for key, value in bundle["spec"].items())
    x = np.column_stack(
        [
            np.ones(len(ti)),
            bundle["encoder"].transform(data.iloc[ti][spec["calendar_columns"]]),
        ]
    )
    offset = logit(np.clip(training.prediction.to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    y = data.target.iloc[ti].to_numpy(dtype=int)
    ratio = (
        float((y == 0).sum() / (y == 1).sum()) if spec["balance"] == "balanced" else 1.0
    )
    assert bundle["positive_weight"] == ratio
    value, gradient = objective(
        bundle["coefficients"],
        x,
        offset,
        y,
        np.where(y == 1, ratio, 1.0),
        spec["penalty"],
        np.log(ratio),
    )
    assert (
        abs(value - meta["optimization_objective"]) < 1e-12
        and np.max(np.abs(gradient)) < 1e-4
    )
    difference = 0.0
    for kind, frame in [("oof", data.iloc[vi]), ("test", test)]:
        source = meta["prediction_sources"][kind]
        kp, sp = [r.ROOT / source[k + "_path"] for k in ["kumo", "spline"]]
        assert (
            r.checksum(kp) == source["kumo_sha256"]
            and r.checksum(sp) == source["spline_sha256"]
        )
        k, s = pd.read_parquet(kp), pd.read_parquet(sp)
        assert (
            k.application_id.tolist() == frame.application_id.tolist()
            and k.application_id.equals(s.application_id)
        )
        expected, delta = predict(
            bundle,
            frame,
            0.75 * k.prediction.to_numpy(dtype=float)
            + 0.25 * s.prediction.to_numpy(dtype=float),
        )
        actual = pd.read_parquet(directory / f"fold{fold}_{kind}.parquet")
        difference = max(
            difference, float(np.max(np.abs(actual.prediction - expected)))
        )
        assert np.max(np.abs(actual.correction_logit - delta)) < 2e-14
    assert difference < 2e-14
    return dict(
        training_rows=len(ti),
        outer_validation_excluded=True,
        optimization_gradient_max_abs=float(np.max(np.abs(gradient))),
        rebuilt_prediction_max_abs_difference=difference,
    )


if __name__ == "__main__":
    run()
