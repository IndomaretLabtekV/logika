"""Correct frozen Kumo using a training-only context holdout, then blend spline.

Each outer-training partition has 56,000 rows. Kumo sees 50,000 labels; a
regularized offset logistic model sees the other 6,000 labels. Neither stage
sees outer-validation labels. All eight correction recipes are fixed in advance.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

import research_goal054 as r
from hybrid_kumo import read_data
from kumo_tabular import make_tables, predict_batches

BASE = r.DEST / "context_holdout_base/cv5_seed31027"
STAGE = r.DEST / "cv5_seed31027"
PLAN = r.DEST / "context_correction_definitions.json"


def specifications():
    plan = json.loads(PLAN.read_text())
    return [
        dict(
            name=f"kumo_offset_l2{str(penalty).replace('.', 'p')}_{balance}_d{str(strength).replace('.', 'p')}",
            family="offset_kumo",
            penalty=penalty,
            balance=balance,
            correction_strength=strength,
            correction_columns=plan["risk_features"],
            spline_weight=0.25,
            n_knots=3,
            spline_degree=2,
        )
        for penalty in plan["penalties"]
        for balance in ["unweighted", "balanced"]
        for strength in plan["correction_strengths"]
    ]


def offset_probability(p):
    return logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6))


def prepare(frame, columns, categorical):
    x = frame[columns].copy()
    for column in categorical:
        x[column] = x[column].fillna("MISSING").astype(str)
    for column in x.columns.difference(categorical):
        v = x[column].replace([np.inf, -np.inf], np.nan)
        x[column] = np.sign(v) * np.log1p(np.abs(v))
    return x


def fit_encoder(spec, data, indices, base_prediction):
    columns = spec["correction_columns"]
    assert not set(columns) & r.EXCLUDED
    training = data.iloc[indices]
    categorical = (
        training[columns]
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    numeric = [c for c in columns if c not in categorical]
    encoder = ColumnTransformer(
        [
            (
                "smooth",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        (
                            "basis",
                            SplineTransformer(
                                n_knots=3,
                                degree=2,
                                knots="quantile",
                                extrapolation="constant",
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "missing",
                MissingIndicator(features="all", sparse=False, error_on_new=False),
                numeric,
            ),
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    with threadpool_limits(limits=4):
        basis = encoder.fit_transform(prepare(training, columns, categorical))
    offset = offset_probability(base_prediction)
    mean, sd = float(offset.mean()), max(float(offset.std()), 1e-8)
    x = np.column_stack([np.ones(len(training)), (offset - mean) / sd, basis])
    bundle = dict(
        encoder=encoder,
        columns=columns,
        categorical=categorical,
        offset_mean=mean,
        offset_sd=sd,
    )
    assert np.isfinite(x).all()
    return bundle, x, offset


def design(bundle, frame, base_prediction):
    with threadpool_limits(limits=4):
        basis = bundle["encoder"].transform(
            prepare(frame, bundle["columns"], bundle["categorical"])
        )
    offset = offset_probability(base_prediction)
    x = np.column_stack(
        [
            np.ones(len(frame)),
            (offset - bundle["offset_mean"]) / bundle["offset_sd"],
            basis,
        ]
    )
    assert np.isfinite(x).all()
    return x, offset


def objective(beta, x, offset, y, weight, penalty, prior_shift):
    eta = offset + prior_shift + x @ beta
    losses = np.logaddexp(0, eta) - y * eta
    normalized = weight / weight.sum()
    value = float(normalized @ losses + 0.5 * penalty * (beta[1:] @ beta[1:]))
    gradient = x.T @ (normalized * (expit(eta) - y))
    gradient[1:] += penalty * beta[1:]
    return value, gradient


def fit_correction(spec, data, indices, base_prediction):
    bundle, x, offset = fit_encoder(spec, data, indices, base_prediction)
    y = data.target.iloc[indices].to_numpy(dtype=int)
    ratio = (
        float((y == 0).sum() / (y == 1).sum()) if spec["balance"] == "balanced" else 1.0
    )
    weight = np.where(y == 1, ratio, 1.0)
    shift = float(np.log(ratio))
    with threadpool_limits(limits=4):
        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            args=(x, offset, y, weight, spec["penalty"], shift),
            method="L-BFGS-B",
            jac=True,
            options=dict(maxiter=300, ftol=1e-11, gtol=1e-7),
        )
    value, gradient = objective(result.x, x, offset, y, weight, spec["penalty"], shift)
    assert result.success and np.max(np.abs(gradient)) < 1e-4
    bundle.update(
        coefficients=result.x,
        spec=spec,
        positive_weight=ratio,
        objective=value,
        gradient_max_abs=float(np.max(np.abs(gradient))),
        iterations=int(result.nit),
    )
    return bundle


def predict_correction(bundle, frame, base_prediction):
    x, offset = design(bundle, frame, base_prediction)
    correction = x @ bundle["coefficients"]
    # The weighted-loss prior shift was an optimization offset; remove it here
    # so beta=0 reproduces the original population probability for every recipe.
    corrected = expit(offset + bundle["spec"]["correction_strength"] * correction)
    return corrected, correction


def context_recipe(data, test, columns):
    return dict(
        family="frozen_kumo_context_holdout",
        columns=columns,
        context_limit=50000,
        estimators=8,
        estimator_batch_size=4,
        query_batch_size=8192,
        model_seed="2026+fold",
        dtype="float16",
        weights="nvidia/Kumo-Tabular",
        weights_revision="v1.0.1",
        weights_commit="3c3e10bbdb590ace29e7026847f92db3c603096d",
        n_splits=r.SPLITS,
        fold_seed=r.FOLD_SEED,
        training_sha256=r.fingerprint(data, ["application_id", "target"] + columns),
        test_sha256=r.fingerprint(test, ["application_id"] + columns),
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


def ensure_context(data, test, columns, fold, ti, vi, pool):
    recipe = context_recipe(data, test, columns)
    definition = BASE / "definition.json"
    if definition.exists():
        assert json.loads(definition.read_text()) == recipe
    else:
        r.write_json(definition, recipe)
    ci = r.context_indices(ti, data.target.astype(int), 50000, 2026 + fold)
    hi = np.setdiff1d(ti, ci)
    assert len(ci) == 50000 and len(hi) == 6000 and len(vi) == 14000
    assert np.array_equal(np.sort(np.r_[ci, hi]), np.sort(ti))
    assert not len(np.intersect1d(ci, hi)) and not len(np.intersect1d(ti, vi))
    meta_file = BASE / f"fold{fold}_metadata.json"
    paths = {
        kind: BASE / f"fold{fold}_{kind}.parquet" for kind in ["oof", "test", "holdout"]
    }
    if meta_file.exists() and all(p.exists() for p in paths.values()):
        meta = json.loads(meta_file.read_text())
        assert meta["recipe"] == recipe and meta[
            "correction_training_ids_sha256"
        ] == r.ids_hash(data, hi)
        for key, path in paths.items():
            assert r.checksum(path) == meta[key + "_sha256"]
        return hi, {key: pd.read_parquet(path) for key, path in paths.items()}, meta

    import torch
    import sdm

    started = time.monotonic()
    diagnostic = (
        r.DEST / "context_reproduction_diagnostics" / f"fold{fold}" / "attempt1"
    )
    reused_diagnostic = (diagnostic / "comparison.json").exists()
    if reused_diagnostic:
        proof = json.loads((diagnostic / "comparison.json").read_text())
        assert proof["recipe"] == recipe
        assert proof["context_ids_sha256"] == r.ids_hash(data, ci)
        assert proof["correction_training_ids_sha256"] == r.ids_hash(data, hi)
        assert proof["validation_ids_sha256"] == r.ids_hash(data, vi)
        frames = {
            kind: pd.read_parquet(diagnostic / f"{kind}.parquet") for kind in paths
        }
        assert (
            frames["holdout"].application_id.tolist()
            == data.application_id.iloc[hi].tolist()
        )
        assert np.array_equal(frames["holdout"].target, data.target.iloc[hi])
        pv, pt, ph = (
            frames[kind].prediction.to_numpy() for kind in ["oof", "test", "holdout"]
        )
        print("REUSE PRESERVED CONTEXT ATTEMPT", fold, diagnostic, flush=True)
    else:
        if not pool:
            while torch.cuda.mem_get_info()[0] < 22 * 1024**3:
                print(
                    "WAIT: need 22 GiB free GPU memory for an additional Kumo context job",
                    flush=True,
                )
                time.sleep(30)
            torch.cuda.set_per_process_memory_fraction(0.30)
            pool.update(
                model=sdm.models.KumoTabular(
                    task="classification", size="large", device="cuda"
                ),
                x=make_tables(data, columns),
                xt=make_tables(test, columns),
                yt=sdm.TableTensor.from_pandas(
                    pd.DataFrame({"target": data.target.astype(int)}),
                    stypes={"target": "categorical"},
                    device="cuda",
                ),
            )
            for parameter in pool["model"].parameters():
                parameter.requires_grad_(False)
        model, x, xt, yt = (pool[k] for k in ["model", "x", "xt", "yt"])
        torch.manual_seed(2026 + fold)
        torch.cuda.reset_peak_memory_stats()
        print(
            "CONTEXT FIT", fold, "context", len(ci), "correction", len(hi), flush=True
        )
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            model.fit(
                x[ci],
                yt[ci],
                num_estimators=8,
                estimator_batch_size=4,
                generator=torch.Generator(device="cuda").manual_seed(2026 + fold),
            )
        pv = predict_batches(model, x[vi], 8192)
        pt = predict_batches(model, xt, 8192)
        ph = predict_batches(model, x[hi], 8192)
        frames = dict(
            oof=pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[vi].to_numpy(),
                    target=data.target.iloc[vi].to_numpy(dtype=int),
                    prediction=pv,
                )
            ),
            test=pd.DataFrame(dict(application_id=test.application_id, prediction=pt)),
            holdout=pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[hi].to_numpy(),
                    target=data.target.iloc[hi].to_numpy(dtype=int),
                    prediction=ph,
                )
            ),
        )
    old_dir = r.baseline_directory(r.SPLITS, r.FOLD_SEED)
    old_meta = json.loads((old_dir / f"fold{fold}_metadata.json").read_text())
    assert old_meta["context_ids_sha256"] == r.ids_hash(data, ci)
    assert old_meta["recipe"]["training_sha256"] == recipe["training_sha256"]
    old_oof = pd.read_parquet(old_dir / f"fold{fold}_oof.parquet")
    old_test = pd.read_parquet(old_dir / f"fold{fold}_test.parquet")
    assert frames["oof"].application_id.equals(old_oof.application_id)
    assert frames["test"].application_id.equals(old_test.application_id)
    difference = float(
        max(
            np.max(np.abs(pv - old_oof.prediction)),
            np.max(np.abs(pt - old_test.prediction)),
        )
    )
    if difference >= 1e-3 and not reused_diagnostic:
        diagnostic_root = r.DEST / "context_reproduction_diagnostics" / f"fold{fold}"
        attempt = len(list(diagnostic_root.glob("attempt*"))) + 1
        diagnostic = diagnostic_root / f"attempt{attempt}"
        diagnostic.mkdir(parents=True, exist_ok=False)
        left = np.r_[pv, pt].astype(float)
        right = np.r_[old_oof.prediction, old_test.prediction].astype(float)
        error = np.abs(left - right)
        for kind, frame in frames.items():
            frame.to_parquet(diagnostic / f"{kind}.parquet", index=False)
        r.write_json(
            diagnostic / "comparison.json",
            dict(
                recipe=recipe,
                fold=fold,
                context_ids_sha256=r.ids_hash(data, ci),
                correction_training_ids_sha256=r.ids_hash(data, hi),
                validation_ids_sha256=r.ids_hash(data, vi),
                max_abs_difference=difference,
                mean_abs_difference=float(error.mean()),
                p99_abs_difference=float(np.quantile(error, 0.99)),
                above_original_tolerance=int((error >= 1e-3).sum()),
                correlation=float(np.corrcoef(left, right)[0, 1]),
                checkpoint_preserved=True,
                validation_labels_used_for_decision=False,
            ),
        )
        print(
            "REPRODUCTION DIAGNOSTIC",
            diagnostic,
            "max",
            difference,
            "mean",
            float(error.mean()),
            flush=True,
        )
    # New frozen predictions are a separate source from the public checkpoint.
    # Exact context IDs and feature bytes, not a prediction-delta threshold,
    # establish the recipe and label isolation. Report numerical variation;
    # never retry, choose or reject a source using outer-validation scores.
    for frame in frames.values():
        assert (
            np.isfinite(frame.prediction).all() and frame.prediction.between(0, 1).all()
        )
    assert np.array_equal(frames["oof"].target, data.target.iloc[vi])
    for kind, frame in frames.items():
        frame.to_parquet(paths[kind], index=False)
    meta = dict(
        recipe=recipe,
        fold=fold,
        training_ids_sha256=r.ids_hash(data, ti),
        context_ids_sha256=r.ids_hash(data, ci),
        correction_training_ids_sha256=r.ids_hash(data, hi),
        validation_ids_sha256=r.ids_hash(data, vi),
        training_rows=len(ti),
        context_rows=len(ci),
        correction_training_rows=len(hi),
        validation_rows=len(vi),
        metrics=r.scores(data.target.iloc[vi], pv),
        reproduced_frozen_prediction_max_abs_difference=difference,
        source_is_numerically_distinct_from_public_baseline=True,
        acceptance_basis="Exact feature/context hashes and disjoint labels; no metric or prediction-delta selection",
        reused_preserved_attempt=(
            str(diagnostic.relative_to(r.ROOT)) if reused_diagnostic else None
        ),
        **{kind + "_sha256": r.checksum(path) for kind, path in paths.items()},
        seconds=time.monotonic() - started,
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    r.write_json(meta_file, meta)
    print(
        "CONTEXT READY",
        fold,
        "difference",
        difference,
        "seconds",
        meta["seconds"],
        flush=True,
    )
    if pool:
        pool["model"].clear()
    gc.collect()
    torch.cuda.empty_cache()
    return hi, frames, meta


def run(fold_numbers=None):
    import torch

    torch.set_num_threads(4)
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    data, test, _, context_columns = read_data()
    y = data.target.astype(int)
    specs = specifications()
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    pool = {}
    for fold in fold_numbers if fold_numbers is not None else range(r.SPLITS):
        ti, vi = folds[fold]
        hi, sources, source_meta = ensure_context(
            data, test, context_columns, fold, ti, vi, pool
        )
        assert (
            sources["holdout"].application_id.tolist()
            == data.application_id.iloc[hi].tolist()
        )
        assert np.array_equal(sources["holdout"].target, y.iloc[hi])
        spline_dir = (
            r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline"
        )
        sv, st = [
            pd.read_parquet(spline_dir / f"fold{fold}_{kind}.parquet")
            for kind in ["oof", "test"]
        ]
        assert sv.application_id.equals(
            sources["oof"].application_id
        ) and st.application_id.equals(test.application_id)
        fitted = {}
        for spec in specs:
            directory = STAGE / spec["name"]
            columns = list(dict.fromkeys(context_columns + spec["correction_columns"]))
            recipe = dict(
                **spec,
                columns=columns,
                context_columns=context_columns,
                n_splits=r.SPLITS,
                fold_seed=r.FOLD_SEED,
                context_limit=50000,
                correction_training_rows=6000,
                training_sha256=r.fingerprint(
                    data, ["application_id", "target"] + columns
                ),
                test_sha256=r.fingerprint(test, ["application_id"] + columns),
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
            oof_file, test_file, meta_file, model_file = [
                directory / f"fold{fold}_{suffix}"
                for suffix in [
                    "oof.parquet",
                    "test.parquet",
                    "metadata.json",
                    "model.joblib",
                ]
            ]
            if all(p.exists() for p in [oof_file, test_file, meta_file, model_file]):
                meta = json.loads(meta_file.read_text())
                assert (
                    meta["recipe"] == recipe
                    and r.checksum(oof_file) == meta["oof_sha256"]
                )
                continue
            started = time.monotonic()
            key = (spec["penalty"], spec["balance"])
            if key not in fitted:
                fitted[key] = fit_correction(
                    spec, data, hi, sources["holdout"].prediction
                )
            bundle = {**fitted[key], "spec": spec}
            cv, dv = predict_correction(
                bundle, data.iloc[vi], sources["oof"].prediction
            )
            ct, dt = predict_correction(bundle, test, sources["test"].prediction)
            pv = 0.75 * cv + 0.25 * sv.prediction.to_numpy(dtype=float)
            pt = 0.75 * ct + 0.25 * st.prediction.to_numpy(dtype=float)
            output = sources["oof"][["application_id", "target"]].copy()
            output["prediction"], output["correction_logit"] = pv, dv
            test_output = sources["test"][["application_id"]].copy()
            test_output["prediction"], test_output["correction_logit"] = pt, dt
            output.to_parquet(oof_file, index=False)
            test_output.to_parquet(test_file, index=False)
            joblib.dump(bundle, model_file, compress=3)
            meta = dict(
                recipe=recipe,
                fold=fold,
                metrics=r.scores(y.iloc[vi], pv),
                training_ids_sha256=r.ids_hash(data, ti),
                context_ids_sha256=r.ids_hash(data, np.setdiff1d(ti, hi)),
                correction_training_ids_sha256=r.ids_hash(data, hi),
                validation_ids_sha256=r.ids_hash(data, vi),
                training_rows=len(ti),
                context_rows=50000,
                correction_training_rows=len(hi),
                validation_rows=len(vi),
                context_source_metadata=str(
                    (BASE / f"fold{fold}_metadata.json").relative_to(r.ROOT)
                ),
                context_source_metadata_sha256=r.checksum(
                    BASE / f"fold{fold}_metadata.json"
                ),
                spline_oof_sha256=r.checksum(spline_dir / f"fold{fold}_oof.parquet"),
                spline_test_sha256=r.checksum(spline_dir / f"fold{fold}_test.parquet"),
                oof_sha256=r.checksum(oof_file),
                test_sha256=r.checksum(test_file),
                model_sha256=r.checksum(model_file),
                optimization_objective=bundle["objective"],
                optimization_gradient_max_abs=bundle["gradient_max_abs"],
                optimization_iterations=bundle["iterations"],
                seconds=time.monotonic() - started,
            )
            # Store the sampler's original (unsorted) order consistently with the
            # foundation source, not the sorted set-difference order above.
            meta["context_ids_sha256"] = source_meta["context_ids_sha256"]
            r.write_json(meta_file, meta)
            print(
                "CORRECTED", spec["name"], fold, json.dumps(meta["metrics"]), flush=True
            )
        assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    from report_goal054 import audit, report_markdown

    report_markdown(audit())


def audit_correction(data, test, fold, ti, vi, directory, meta):
    """Verify stage separation, optimization and actual OOF/test reconstruction."""
    spec = meta["recipe"]
    ci = r.context_indices(ti, data.target.astype(int), 50000, 2026 + fold)
    hi = np.setdiff1d(ti, ci)
    assert len(ci) == 50000 and len(hi) == 6000
    assert np.array_equal(np.sort(np.r_[ci, hi]), np.sort(ti))
    assert not len(np.intersect1d(ci, hi)) and not len(np.intersect1d(ti, vi))
    assert meta["context_ids_sha256"] == r.ids_hash(data, ci)
    assert meta["correction_training_ids_sha256"] == r.ids_hash(data, hi)
    source_file = r.ROOT / meta["context_source_metadata"]
    assert r.checksum(source_file) == meta["context_source_metadata_sha256"]
    source_meta = json.loads(source_file.read_text())
    assert source_meta["recipe"]["training_sha256"] == r.fingerprint(
        data, ["application_id", "target"] + spec["context_columns"]
    )
    assert source_meta["context_ids_sha256"] == r.ids_hash(data, ci)
    assert source_meta["correction_training_ids_sha256"] == r.ids_hash(data, hi)
    assert source_meta["validation_ids_sha256"] == r.ids_hash(data, vi)
    sources = {}
    for kind in ["holdout", "oof", "test"]:
        path = source_file.parent / f"fold{fold}_{kind}.parquet"
        assert r.checksum(path) == source_meta[kind + "_sha256"]
        sources[kind] = pd.read_parquet(path)
    assert (
        sources["holdout"].application_id.tolist()
        == data.application_id.iloc[hi].tolist()
    )
    assert np.array_equal(sources["holdout"].target, data.target.iloc[hi].astype(int))
    assert (
        sources["oof"].application_id.tolist() == data.application_id.iloc[vi].tolist()
    )
    assert np.array_equal(sources["oof"].target, data.target.iloc[vi].astype(int))
    assert sources["test"].application_id.equals(test.application_id)
    model_file = directory / f"fold{fold}_model.joblib"
    assert r.checksum(model_file) == meta["model_sha256"]
    bundle = joblib.load(model_file)
    assert all(spec[key] == value for key, value in bundle["spec"].items())
    x, offset = design(bundle, data.iloc[hi], sources["holdout"].prediction)
    y = data.target.iloc[hi].to_numpy(dtype=int)
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
    assert abs(value - meta["optimization_objective"]) < 1e-12
    assert np.max(np.abs(gradient)) < 1e-4
    spline_dir = (
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline"
    )
    difference = 0.0
    for kind, frame in [("oof", data.iloc[vi]), ("test", test)]:
        spline_file = spline_dir / f"fold{fold}_{kind}.parquet"
        assert r.checksum(spline_file) == meta[f"spline_{kind}_sha256"]
        spline = pd.read_parquet(spline_file)
        actual = pd.read_parquet(directory / f"fold{fold}_{kind}.parquet")
        assert actual.application_id.equals(spline.application_id)
        corrected, delta = predict_correction(bundle, frame, sources[kind].prediction)
        expected = 0.75 * corrected + 0.25 * spline.prediction.to_numpy(dtype=float)
        difference = max(
            difference, float(np.max(np.abs(expected - actual.prediction)))
        )
        assert np.max(np.abs(delta - actual.correction_logit)) < 2e-14
    assert difference < 2e-14
    return dict(
        correction_training_rows=len(hi),
        three_way_split_disjoint=True,
        optimization_gradient_max_abs=float(np.max(np.abs(gradient))),
        rebuilt_prediction_max_abs_difference=difference,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(r.SPLITS))
    args = parser.parse_args()
    run(args.folds)


if __name__ == "__main__":
    main()
