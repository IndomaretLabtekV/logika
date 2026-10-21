"""Five-fold smooth additive/interactions models, with optional balanced loss.

The first comparison uses a recipe declared in ebm_plan.json. All EBM binning,
interaction detection and internal early stopping stay inside outer training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r


def run(variants):
    from interpret.glassbox import ExplainableBoostingClassifier

    plan = json.loads((r.DEST / "ebm_plan.json").read_text())
    spec = next(s for s in r.specifications() if s["feature_set"] == "denoised")
    data, test = r.inputs([spec])
    x, xt = r.model_features(data, spec["columns"]), r.model_features(
        test, spec["columns"]
    )
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    for variant in variants:
        name = "ebm_denoised_" + variant
        directory = r.DEST / f"cv{r.SPLITS}_seed{r.FOLD_SEED}" / name
        recipe = dict(
            family="ebm",
            variant=variant,
            columns=spec["columns"],
            params=plan["params"],
            folds=r.SPLITS,
            seed=r.FOLD_SEED,
            training_sha256=r.fingerprint(
                data, ["application_id", "target"] + spec["columns"]
            ),
            test_sha256=r.fingerprint(test, ["application_id"] + spec["columns"]),
            packages={
                p: importlib.metadata.version(p)
                for p in ["interpret-core", "numpy", "pandas", "scikit-learn", "joblib"]
            },
        )
        definition = directory / "definition.json"
        if definition.exists() and any(directory.glob("fold*_oof.parquet")):
            assert json.loads(definition.read_text()) == recipe
        r.write_json(definition, recipe)
        for fold, (ti, vi) in enumerate(folds):
            oof_file, test_file, model_file, meta_file = [
                directory / f"fold{fold}_{suffix}"
                for suffix in [
                    "oof.parquet",
                    "test.parquet",
                    "model.joblib",
                    "metadata.json",
                ]
            ]
            if all(p.exists() for p in [oof_file, test_file, model_file, meta_file]):
                meta = json.loads(meta_file.read_text())
                assert meta["recipe"] == recipe
                assert r.checksum(oof_file) == meta["oof_sha256"]
                print("CACHED", name, fold, meta["metrics"], flush=True)
                continue
            ratio = (
                float((y.iloc[ti] == 0).sum() / (y.iloc[ti] == 1).sum())
                if variant == "class_balanced"
                else 1.0
            )
            weight = np.where(y.iloc[ti].to_numpy() == 1, ratio, 1.0)
            start = time.monotonic()
            print(
                "FIT", name, fold, "rows", len(ti), "positive_weight", ratio, flush=True
            )
            model = ExplainableBoostingClassifier(
                **plan["params"], random_state=2026 + fold
            )
            model.fit(x.iloc[ti], y.iloc[ti], sample_weight=weight)

            def corrected(frame):
                p = model.predict_proba(frame)[
                    :, np.flatnonzero(model.classes_ == 1)[0]
                ]
                return p / (ratio - (ratio - 1) * p)

            pv, pt = corrected(x.iloc[vi]), corrected(xt)
            assert np.isfinite(pv).all() and np.isfinite(pt).all()
            pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[vi].to_numpy(),
                    target=y.iloc[vi].to_numpy(),
                    prediction=pv,
                )
            ).to_parquet(oof_file, index=False)
            pd.DataFrame(
                dict(application_id=test.application_id, prediction=pt)
            ).to_parquet(test_file, index=False)
            joblib.dump(model, model_file, compress=3)
            meta = dict(
                recipe=recipe,
                fold=fold,
                metrics=r.scores(y.iloc[vi], pv),
                positive_weight=ratio,
                training_ids_sha256=r.ids_hash(data, ti),
                validation_ids_sha256=r.ids_hash(data, vi),
                training_rows=len(ti),
                validation_rows=len(vi),
                oof_sha256=r.checksum(oof_file),
                test_sha256=r.checksum(test_file),
                model_sha256=r.checksum(model_file),
                seconds=time.monotonic() - start,
                selected_terms=[
                    [int(i) for i in term] for term in model.term_features_
                ],
                best_iteration=np.asarray(model.best_iteration_).tolist(),
            )
            r.write_json(meta_file, meta)
            print(
                "RESULT",
                name,
                fold,
                json.dumps(meta["metrics"]),
                "seconds",
                meta["seconds"],
                flush=True,
            )
            assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
        result = compare()
        if result["target_reached"]:
            print(
                "TARGET REACHED: independent audit and export still required.",
                flush=True,
            )
            break


def compare():
    spec = next(s for s in r.specifications() if s["feature_set"] == "denoised")
    data, _ = r.inputs([spec])
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    kumo = r.read_oof(r.baseline_directory(r.SPLITS, r.FOLD_SEED), data, folds)
    spline = r.read_oof(
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
        data,
        folds,
    )
    baseline = 0.75 * kumo + 0.25 * spline

    def metrics(p):
        row = r.scores(y, p)
        row["fold_ks"] = [r.scores(y.iloc[vi], p[vi])["ks"] for _, vi in folds]
        row["mean_fold_ks"] = float(np.mean(row["fold_ks"]))
        return row

    base = dict(name="public_056393_recipe", **metrics(baseline))
    rows = [base]
    for variant in ["unweighted", "class_balanced"]:
        name = "ebm_denoised_" + variant
        directory = r.DEST / f"cv{r.SPLITS}_seed{r.FOLD_SEED}" / name
        if not all(
            (directory / f"fold{i}_metadata.json").exists() for i in range(r.SPLITS)
        ):
            continue
        p = r.read_oof(directory, data, folds)
        for weight in [1.0, 0.25, 0.5, 0.75]:
            row = dict(
                name=name,
                new_model_weight=weight,
                **metrics((1 - weight) * baseline + weight * p),
            )
            row.update(
                delta_ks=row["ks"] - base["ks"],
                delta_mean_fold_ks=row["mean_fold_ks"] - base["mean_fold_ks"],
                fold_wins=sum(a > b for a, b in zip(row["fold_ks"], base["fold_ks"])),
                target_reached=row["ks"] >= 0.54,
            )
            rows.append(row)
    result = dict(
        comparisons=rows,
        target_reached=any(row.get("target_reached", False) for row in rows),
        protocol=dict(splits=r.SPLITS, seed=r.FOLD_SEED),
        caveat="Repeated selection reuses training labels and folds; not untouched holdout.",
    )
    r.write_json(r.DEST / "ebm_comparison.json", result)
    print("COMPARISON", json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["unweighted", "class_balanced"],
        default=["unweighted", "class_balanced"],
    )
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    compare() if args.compare_only else run(args.variants)


if __name__ == "__main__":
    main()
