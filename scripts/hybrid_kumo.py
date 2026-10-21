"""Fixed 75% frozen Kumo + 25% additive spline; matching-fold confirmation.

The spline is the reverse_spline recipe already tested in notebook 07. Its
preprocessing is fitted on outer training only. No test labels are available.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from improve_kumo import ROOT, fingerprint
from kumo_tabular import (
    EXCLUDED,
    context_indices,
    make_tables,
    metrics,
    ks,
    predict_batches,
)
from summarize_kumo_improvement import read_oof

DEST = ROOT / "artifacts/optimization/kumo_hybrid"
WEIGHT = 0.25


def specifications():
    spline = next(
        s
        for s in json.loads(
            (
                ROOT / "artifacts/reverse_engineering/validation_definitions.json"
            ).read_text()
        )
        if s["name"] == "reverse_spline"
    )
    kumo_columns = json.loads(
        (
            ROOT / "artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"
        ).read_text()
    )["spec"]["columns"]
    assert not set(spline["columns"] + kumo_columns) & EXCLUDED
    return spline, kumo_columns


def read_data():
    spline, kumo_columns = specifications()
    columns = list(dict.fromkeys(kumo_columns + spline["columns"]))
    full = pd.read_parquet(
        ROOT / "data/processed/application_features_reverse.parquet",
        columns=["application_id", "customer_id", "target", "split"] + columns,
    )
    train = full[full.split.eq("train")].reset_index(drop=True)
    test = full[full.split.eq("test")].reset_index(drop=True)
    assert train.application_id.is_unique and test.application_id.is_unique
    assert set(train.customer_id).isdisjoint(test.customer_id)
    return train, test, spline, kumo_columns


def ids_hash(data, indices):
    return hashlib.sha256(
        "\n".join(data.application_id.iloc[indices]).encode()
    ).hexdigest()


def baseline_directory(splits, seed):
    stage = f"cv{splits}_seed{seed}"
    old = (
        ROOT / "artifacts/optimization/kumo/large_v5_ctx50000_e8"
        if stage == "cv3_seed8247"
        else ROOT / "artifacts/optimization/kumo_improvement" / stage / "baseline_e8"
    )
    if all(
        (old / f"fold{i}_oof.parquet").exists()
        and (old / f"fold{i}_test.parquet").exists()
        for i in range(splits)
    ):
        return old
    return DEST / stage / "baseline_e8"


def ensure_baseline(data, test, columns, splits, seed):
    directory = baseline_directory(splits, seed)
    if all(
        (directory / f"fold{i}_oof.parquet").exists()
        and (directory / f"fold{i}_test.parquet").exists()
        for i in range(splits)
    ):
        meta_file = directory / "fold0_metadata.json"
        if not meta_file.exists():
            meta_file = directory / "fold0_oof.json"
        cached = json.loads(meta_file.read_text())["recipe"]
        known = cached.get("training_sha256", cached.get("training_data_sha256"))
        assert known == fingerprint(
            data, ["application_id", "target"] + columns
        ), "Baseline input changed; rebuild frozen predictions before combining."
        if "test_sha256" in cached:
            assert cached["test_sha256"] == fingerprint(
                test, ["application_id"] + columns
            )
        return directory
    import torch, sdm

    torch.set_num_threads(8)
    directory.mkdir(parents=True, exist_ok=True)
    y = data.target.astype(int)
    recipe = dict(
        columns=columns,
        n_splits=splits,
        fold_seed=seed,
        context_limit=50000,
        estimators=8,
        estimator_batch_size=4,
        query_batch_size=8192,
        autocast="float16",
        weights="nvidia/Kumo-Tabular",
        weights_revision="v1.0.1",
        weights_commit="3c3e10bbdb590ace29e7026847f92db3c603096d",
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
        training_sha256=fingerprint(data, ["application_id", "target"] + columns),
        test_sha256=fingerprint(test, ["application_id"] + columns),
    )
    folds = list(
        StratifiedKFold(splits, shuffle=True, random_state=seed).split(data, y)
    )
    model = x = xt = yt = None
    for fold, (ti, vi) in enumerate(folds):
        meta_file = directory / f"fold{fold}_metadata.json"
        oof_file = directory / f"fold{fold}_oof.parquet"
        test_file = directory / f"fold{fold}_test.parquet"
        if meta_file.exists() and oof_file.exists() and test_file.exists():
            assert json.loads(meta_file.read_text())["recipe"] == recipe
            continue
        if model is None:
            x = make_tables(data, columns)
            xt = make_tables(test, columns)
            yt = sdm.TableTensor.from_pandas(
                pd.DataFrame({"target": y}),
                stypes={"target": "categorical"},
                device="cuda",
            )
            model = sdm.models.KumoTabular(
                task="classification", size="large", device="cuda"
            )
        ci = context_indices(ti, y, 50000, 2026 + fold)
        assert not len(np.intersect1d(ci, vi))
        torch.manual_seed(2026 + fold)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            model.fit(
                x[ci],
                yt[ci],
                num_estimators=8,
                estimator_batch_size=4,
                generator=torch.Generator(device="cuda").manual_seed(2026 + fold),
            )
        p = predict_batches(model, x[vi], 8192)
        pt = predict_batches(model, xt, 8192)
        pd.DataFrame(
            dict(
                application_id=data.application_id.iloc[vi].to_numpy(),
                target=y.iloc[vi].to_numpy(),
                prediction=p,
            )
        ).to_parquet(oof_file, index=False)
        pd.DataFrame(
            dict(application_id=test.application_id, prediction=pt)
        ).to_parquet(test_file, index=False)
        meta = dict(
            recipe=recipe,
            fold=fold,
            metrics=metrics(y.iloc[vi], p),
            training_ids_sha256=ids_hash(data, ti),
            validation_ids_sha256=ids_hash(data, vi),
            context_ids_sha256=ids_hash(data, ci),
        )
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        print("FROZEN", fold, meta["metrics"], flush=True)
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return directory


def prepare(frame, columns, categorical):
    x = frame[columns].copy()
    for c in categorical:
        x[c] = x[c].fillna("MISSING").astype(str)
    for c in x.columns.difference(categorical):
        x[c] = np.sign(x[c]) * np.log1p(np.abs(x[c]))
    return x


def fit_spline(spec, data, indices):
    columns = spec["columns"]
    training = data.iloc[indices]
    params = spec["params"]
    assert params == dict(C=0.05, n_knots=5, positive_weight=1, interactions=False)
    categorical = data[columns].select_dtypes(include="object").columns.tolist()
    numeric = [c for c in columns if c not in categorical]
    continuous = [c for c in numeric if training[c].nunique(dropna=True) > 10]
    linear = [c for c in numeric if c not in continuous]
    encoder = ColumnTransformer(
        [
            (
                "spline",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        (
                            "basis",
                            SplineTransformer(
                                n_knots=5,
                                degree=3,
                                knots="quantile",
                                extrapolation="constant",
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                continuous,
            ),
            (
                "linear",
                Pipeline(
                    [
                        ("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            ),
            (
                "missing",
                MissingIndicator(features="all", sparse=False, error_on_new=False),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    with threadpool_limits(limits=4):
        x = encoder.fit_transform(prepare(training, columns, categorical))
        model = LogisticRegression(
            C=0.05, solver="newton-cholesky", max_iter=80, tol=1e-5
        )
        model.fit(x, training.target.astype(int), sample_weight=np.ones(len(training)))
    return dict(encoder=encoder, model=model, columns=columns, categorical=categorical)


def predict_spline(bundle, frame):
    x = prepare(frame, bundle["columns"], bundle["categorical"])
    with threadpool_limits(limits=4):
        p = bundle["model"].predict_proba(bundle["encoder"].transform(x))[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def evaluate(splits, seed, force=False):
    data, test, spec, kumo_columns = read_data()
    y = data.target.astype(int)
    stage = f"cv{splits}_seed{seed}"
    path = DEST / stage / "reverse_spline"
    path.mkdir(parents=True, exist_ok=True)
    baseline = ensure_baseline(data, test, kumo_columns, splits, seed)
    recipe = dict(
        spec=spec,
        n_splits=splits,
        fold_seed=seed,
        model_seed=2026,
        training_sha256=fingerprint(
            data, ["application_id", "target"] + spec["columns"]
        ),
        test_sha256=fingerprint(test, ["application_id"] + spec["columns"]),
        packages={
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
        },
    )
    definition = path / "definition.json"
    if definition.exists() and any(path.glob("fold*_oof.parquet")) and not force:
        assert json.loads(definition.read_text()) == recipe
    definition.write_text(json.dumps(recipe, indent=2) + "\n")
    folds = list(
        StratifiedKFold(splits, shuffle=True, random_state=seed).split(data, y)
    )
    for fold, (ti, vi) in enumerate(folds):
        meta_file = path / f"fold{fold}_metadata.json"
        oof_file = path / f"fold{fold}_oof.parquet"
        test_file = path / f"fold{fold}_test.parquet"
        model_file = path / f"fold{fold}_model.joblib"
        if (
            meta_file.exists()
            and oof_file.exists()
            and test_file.exists()
            and model_file.exists()
            and not force
        ):
            meta = json.loads(meta_file.read_text())
            assert meta["recipe"] == recipe
            for file, key in [
                (oof_file, "oof_sha256"),
                (test_file, "test_sha256"),
                (model_file, "model_sha256"),
            ]:
                assert hashlib.sha256(file.read_bytes()).hexdigest() == meta[key]
            continue
        assert not len(np.intersect1d(ti, vi))
        bundle = fit_spline(spec, data, ti)
        p = predict_spline(bundle, data.iloc[vi])
        pt = predict_spline(bundle, test)
        joblib.dump(bundle, model_file, compress=3)
        pd.DataFrame(
            dict(
                application_id=data.application_id.iloc[vi].to_numpy(),
                target=y.iloc[vi].to_numpy(),
                prediction=p,
            )
        ).to_parquet(oof_file, index=False)
        pd.DataFrame(
            dict(application_id=test.application_id, prediction=pt)
        ).to_parquet(test_file, index=False)
        meta = dict(
            recipe=recipe,
            fold=fold,
            metrics=metrics(y.iloc[vi], p),
            training_ids_sha256=ids_hash(data, ti),
            validation_ids_sha256=ids_hash(data, vi),
            preprocessing_fitted_on_training_only=True,
            model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
            oof_sha256=hashlib.sha256(oof_file.read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest(),
        )
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        print("SPLINE", stage, fold, meta["metrics"], flush=True)
    base = read_oof(baseline, data[["application_id", "target"]], splits).astype(
        np.float64
    )
    spline = read_oof(path, data[["application_id", "target"]], splits).astype(
        np.float64
    )
    base_folds = np.array([ks(y.iloc[vi], base[vi]) for _, vi in folds])
    rows = []
    for name, p in [
        ("frozen_baseline", base),
        ("spline_only", spline),
        ("kumo75_spline25", (1 - WEIGHT) * base + WEIGHT * spline),
    ]:
        per = np.array([ks(y.iloc[vi], p[vi]) for _, vi in folds])
        rows.append(
            dict(
                name=name,
                **metrics(y, p),
                mean_fold_ks=float(per.mean()),
                fold_ks=per.tolist(),
                fold_deltas=(per - base_folds).tolist(),
                delta_ks=ks(y, p) - ks(y, base),
                delta_mean_fold_ks=float((per - base_folds).mean()),
                fold_wins=int((per > base_folds).sum()),
            )
        )
    result = dict(
        stage=stage,
        weight_spline=WEIGHT,
        blend_dtype="float64",
        comparisons=rows,
        baseline_directory=str(baseline.relative_to(ROOT)),
        warning="Repeated model selection on the same training labels; new folds are not an untouched holdout.",
    )
    (DEST / stage / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).drop(columns=["fold_ks", "fold_deltas"]).to_csv(
        DEST / stage / "comparison.csv", index=False
    )
    return result


def export(splits=5, seed=31027, exploratory=False):
    stage = f"cv{splits}_seed{seed}"
    summary = evaluate(splits, seed)
    row = next(r for r in summary["comparisons"] if r["name"] == "kumo75_spline25")
    promotion_passed = (
        row["delta_ks"] > 0
        and row["delta_mean_fold_ks"] > 0
        and row["fold_wins"] >= (splits // 2 + 1)
    )
    assert promotion_passed or (
        exploratory and row["delta_ks"] > 0
    ), "Exploratory export still requires a positive pooled KS delta."
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    p = np.zeros(len(sample), dtype=np.float64)
    sources = {}
    for directory, weight in [
        (baseline_directory(splits, seed), 1 - WEIGHT),
        (DEST / stage / "reverse_spline", WEIGHT),
    ]:
        for fold in range(splits):
            source = directory / f"fold{fold}_test.parquet"
            part = pd.read_parquet(source)
            assert len(part) == len(sample) and part.application_id.is_unique
            ordered = sample[["application_id"]].merge(
                part, on="application_id", validate="one_to_one"
            )
            assert ordered.application_id.equals(sample.application_id)
            p += weight * ordered.prediction.to_numpy(dtype=np.float64) / splits
            sources[str(source.relative_to(ROOT))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
    assert len(p) == 30000 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    result = sample[["application_id"]].copy()
    result["default_probability"] = p
    file = ROOT / (
        "submission_kumo_spline_exploratory.csv"
        if exploratory
        else "submission_kumo_v2.csv"
    )
    result.to_csv(file, index=False)
    meta = dict(
        file=file.name,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        rows=len(result),
        recipe="75% frozen Kumo + 25% latent-feature spline",
        stage=stage,
        validation=row,
        source_prediction_sha256=sources,
        public_score=None,
        checkpoint_056262_preserved=True,
        promotion_passed=promotion_passed,
        export_mode="exploratory" if exploratory else "confirmed_local",
    )
    meta_name = (
        "exploratory_submission_metadata.json"
        if exploratory
        else "submission_metadata.json"
    )
    (DEST / meta_name).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--export-exploratory",
        action="store_true",
        help="Export a separately named pooled-KS candidate even if conservative promotion fails.",
    )
    args = parser.parse_args()
    screen = evaluate(3, 8247, args.force)
    chosen = screen["comparisons"][-1]
    if not (
        chosen["delta_ks"] > 0
        and chosen["delta_mean_fold_ks"] > 0
        and chosen["fold_wins"] >= 2
    ):
        (DEST / "verdict.json").write_text(
            json.dumps(dict(passed=False, stage="screening", selected=chosen), indent=2)
            + "\n"
        )
        return
    lock = dict(
        weight_spline=WEIGHT,
        screen=chosen,
        confirmation_stage="cv5_seed31027",
        rule="Positive pooled/mean KS and at least 3/5 fold wins.",
    )
    (DEST / "locked_recipe.json").write_text(json.dumps(lock, indent=2) + "\n")
    print("LOCKED", json.dumps(lock), flush=True)
    confirm = evaluate(5, 31027, args.force)
    row = confirm["comparisons"][-1]
    verdict = dict(
        passed=row["delta_ks"] > 0
        and row["delta_mean_fold_ks"] > 0
        and row["fold_wins"] >= 3,
        stage="confirmation",
        selected=row,
    )
    if verdict["passed"]:
        verdict["submission"] = export()
    elif args.export_exploratory and row["delta_ks"] > 0:
        verdict["exploratory_submission"] = export(exploratory=True)
    (DEST / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print("VERDICT", json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
