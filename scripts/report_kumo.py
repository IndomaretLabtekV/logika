"""Compare complete three-fold Kumo experiments and export a separate CSV."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kumo_tabular import ROOT, OUT, ks, metrics


def compare():
    baseline_path = ROOT / "artifacts/optimization/v5/selection_oof_tabm.parquet"
    has_baseline = baseline_path.exists()
    if has_baseline:
        anchor = pd.read_parquet(baseline_path)
        anchor["version5"] = 0.75 * anchor.version4 + 0.25 * anchor.v5_tabm_weight1
    else:
        full = pd.read_parquet(
            ROOT / "data/processed/application_features_v5.parquet",
            columns=["application_id", "target", "split"],
        )
        anchor = full[full.split.eq("train")][["application_id", "target"]].reset_index(
            drop=True
        )
    y = anchor.target.astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=8247).split(anchor, y))
    predictions = anchor[
        [c for c in ["application_id", "target", "version4", "version5"] if c in anchor]
    ].copy()
    rows = []

    def add(name, p):
        per_fold = [ks(y.iloc[vi], np.asarray(p)[vi]) for _, vi in folds]
        baseline = (
            [ks(y.iloc[vi], anchor.version5.iloc[vi]) for _, vi in folds]
            if has_baseline
            else None
        )
        rows.append(
            {
                "name": name,
                **metrics(y, p),
                "fold_ks": per_fold,
                "mean_fold_ks": float(np.mean(per_fold)),
                "delta_pooled_vs_v5": (
                    ks(y, p) - ks(y, anchor.version5) if has_baseline else None
                ),
                "delta_mean_vs_v5": (
                    float(np.mean(per_fold) - np.mean(baseline))
                    if has_baseline
                    else None
                ),
                "fold_wins_vs_v5": (
                    int(np.sum(np.array(per_fold) > baseline)) if has_baseline else None
                ),
            }
        )

    if has_baseline:
        add("version4", anchor.version4)
        add("version5", anchor.version5)
    for directory in sorted(OUT.iterdir()):
        if not directory.is_dir() or not all(
            (directory / f"fold{i}_oof.parquet").exists() for i in range(3)
        ):
            continue
        frames = [pd.read_parquet(directory / f"fold{i}_oof.parquet") for i in range(3)]
        combined = pd.concat(frames, ignore_index=True)
        assert combined.application_id.is_unique and len(combined) == len(anchor)
        joined = anchor[["application_id", "target"]].merge(
            combined,
            on="application_id",
            validate="one_to_one",
            suffixes=("", "_check"),
        )
        assert np.array_equal(joined.target, joined.target_check)
        p = joined.prediction.to_numpy()
        name = directory.name
        predictions[name] = p
        add(name, p)
        for weight in [0.25, 0.5] if has_baseline else []:
            blend_name = f"v5_plus_{name}_w{weight}"
            predictions[blend_name] = (1 - weight) * anchor.version5 + weight * p
            add(blend_name, predictions[blend_name])
    summary = {
        "validation": "StratifiedKFold(3, shuffle=True, random_state=8247)",
        "warning": "Selection on training OOF; not an unbiased test or public leaderboard score.",
        "comparisons": rows,
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    predictions.to_parquet(OUT / "comparison_oof.parquet", index=False)
    pd.DataFrame(rows).drop(columns="fold_ks").to_csv(
        OUT / "comparison.csv", index=False
    )
    return summary, predictions


def export(variant):
    summary, _ = compare()
    directory = OUT / variant
    test = None
    for fold in range(3):
        frame = pd.read_parquet(directory / f"fold{fold}_test.parquet")
        assert frame.application_id.is_unique and len(frame) == 30000
        if test is None:
            test = frame.copy()
            test.prediction = 0.0
        assert test.application_id.equals(frame.application_id)
        test.prediction += frame.prediction / 3
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    submission = sample[["application_id"]].merge(
        test, on="application_id", validate="one_to_one"
    )
    submission = submission.rename(columns={"prediction": "default_probability"})
    assert submission.application_id.equals(sample.application_id)
    assert list(submission.columns) == ["application_id", "default_probability"]
    assert np.isfinite(submission.default_probability).all()
    assert submission.default_probability.between(0, 1).all()
    path = ROOT / "results/teman/submission_kumo.csv"
    submission.to_csv(path, index=False)
    result = {
        "variant": variant,
        "file": path.name,
        "rows": len(submission),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "test_ensemble": "Mean of the same three Kumo fold models used for OOF.",
        "validation": next(x for x in summary["comparisons"] if x["name"] == variant),
        "public_score": None,
        "replaces_default": False,
    }
    public_path = OUT / "public_leaderboard_result.json"
    if public_path.exists():
        recorded = json.loads(public_path.read_text())
        if (
            recorded.get("file") == result["file"]
            and recorded.get("sha256") == result["sha256"]
        ):
            result["public_score"] = recorded["public_score"]
            result["public_score_verified_via_kaggle_api"] = recorded.get(
                "verified_via_kaggle_api", False
            )
    (OUT / "submission_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export_blend(variant, weight=0.25):
    """Use the same three-fold baseline test ensemble as the OOF comparison."""
    assert weight in [0.25, 0.5]
    assert (
        ROOT / "artifacts/optimization/v5/selection_oof_tabm.parquet"
    ).exists(), "V5 OOF is required to validate this blend."
    pure_metadata = export(variant)
    pure = pd.read_csv(ROOT / pure_metadata["file"])
    baseline = pd.read_parquet(
        ROOT / "artifacts/optimization/v5/selection_test_tabm.parquet"
    )
    baseline["v5"] = 0.75 * baseline.version4 + 0.25 * baseline.v5_tabm_weight1
    joined = pure.merge(
        baseline[["application_id", "v5"]], on="application_id", validate="one_to_one"
    )
    assert joined.application_id.equals(pure.application_id)
    joined["default_probability"] = (
        weight * joined.default_probability + (1 - weight) * joined.v5
    )
    submission = joined[["application_id", "default_probability"]]
    assert len(submission) == 30000 and submission.application_id.is_unique
    assert submission.default_probability.between(0, 1).all()
    path = ROOT / "results/teman/submission_kumo_blend.csv"
    submission.to_csv(path, index=False)
    summary = json.loads((OUT / "comparison.json").read_text())
    result = {
        "file": path.name,
        "rows": len(submission),
        "kumo_variant": variant,
        "kumo_weight": weight,
        "v5_weight": 1 - weight,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline_test_source": "artifacts/optimization/v5/selection_test_tabm.parquet",
        "baseline_test_models": "Three-fold selection models, matching the OOF comparison; not the ten-fold public V5 file.",
        "validation": next(
            x
            for x in summary["comparisons"]
            if x["name"] == f"v5_plus_{variant}_w{weight}"
        ),
        "public_score": None,
        "replaces_default": False,
    }
    public_path = OUT / "public_leaderboard_result.json"
    if public_path.exists():
        recorded = json.loads(public_path.read_text())
        if (
            recorded.get("file") == result["file"]
            and recorded.get("sha256") == result["sha256"]
        ):
            result["public_score"] = recorded["public_score"]
            result["public_score_verified_via_kaggle_api"] = recorded.get(
                "verified_via_kaggle_api", False
            )
    (OUT / "blend_submission_metadata.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", help="Complete variant folder name; creates submission_kumo.csv."
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        choices=[0.25, 0.5],
        help="Also export submission_kumo_blend.csv using the matching V5 selection ensemble.",
    )
    args = parser.parse_args()
    if args.blend_weight is not None and not args.export:
        parser.error("--blend-weight requires --export VARIANT")
    if args.export:
        print(json.dumps(export(args.export), indent=2))
        if args.blend_weight is not None:
            print(json.dumps(export_blend(args.export, args.blend_weight), indent=2))
    else:
        summary, _ = compare()
        print(
            pd.DataFrame(summary["comparisons"])
            .drop(columns="fold_ks")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
