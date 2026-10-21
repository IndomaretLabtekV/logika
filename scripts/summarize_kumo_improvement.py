"""Compare Kumo candidates on identical folds and export a separate checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from improve_kumo import DEST
from kumo_tabular import ROOT, ks, metrics


def read_oof(directory, anchor, splits):
    parts = [pd.read_parquet(directory / f"fold{i}_oof.parquet") for i in range(splits)]
    full = pd.concat(parts, ignore_index=True)
    assert full.application_id.is_unique and len(full) == len(anchor)
    joined = anchor.merge(
        full, on="application_id", validate="one_to_one", suffixes=("", "_check")
    )
    assert np.array_equal(joined.target, joined.target_check)
    return joined.prediction.to_numpy()


def summarize(stage):
    match = re.fullmatch(r"cv(\d+)_seed(\d+)", stage)
    assert match
    splits, seed = map(int, match.groups())
    path = DEST / stage
    anchor = pd.read_parquet(
        ROOT / "data/processed/application_features_v5.parquet",
        columns=["application_id", "target", "split"],
    )
    anchor = anchor.loc[
        anchor.split.eq("train"), ["application_id", "target"]
    ].reset_index(drop=True)
    y = anchor.target.astype(int)
    folds = list(
        StratifiedKFold(splits, shuffle=True, random_state=seed).split(anchor, y)
    )
    baseline_dir = (
        ROOT / "artifacts/optimization/kumo/large_v5_ctx50000_e8"
        if stage == "cv3_seed8247"
        else path / "baseline_e8"
    )
    if not (baseline_dir / "fold0_oof.parquet").exists():
        baseline_dir = path / "baseline_e8"
    baseline = read_oof(baseline_dir, anchor, splits)
    baseline_fold = np.array([ks(y.iloc[vi], baseline[vi]) for _, vi in folds])
    rows = []
    predictions = anchor.copy()
    predictions["public_056262_recipe"] = baseline

    def add(name, p, components):
        per_fold = np.array([ks(y.iloc[vi], p[vi]) for _, vi in folds])
        rows.append(
            dict(
                name=name,
                **metrics(y, p),
                fold_ks=per_fold.tolist(),
                mean_fold_ks=float(per_fold.mean()),
                delta_ks=ks(y, p) - ks(y, baseline),
                delta_mean_fold_ks=float((per_fold - baseline_fold).mean()),
                fold_wins=int((per_fold > baseline_fold).sum()),
                fold_deltas=(per_fold - baseline_fold).tolist(),
                components=components,
            )
        )

    add("public_056262_recipe", baseline, [{"name": "baseline_e8", "weight": 1.0}])
    for directory in sorted(path.iterdir()):
        if not directory.is_dir() or directory.name == "baseline_e8":
            continue
        if not all(
            (directory / f"fold{i}_oof.parquet").exists() for i in range(splits)
        ):
            continue
        p = read_oof(directory, anchor, splits)
        predictions[directory.name] = p
        add(directory.name, p, [{"name": directory.name, "weight": 1.0}])
        for w in [0.25, 0.5]:
            name = f"baseline_plus_{directory.name}_w{w}"
            blend = (1 - w) * baseline + w * p
            predictions[name] = blend
            add(
                name,
                blend,
                [
                    {"name": "baseline_e8", "weight": 1 - w},
                    {"name": directory.name, "weight": w},
                ],
            )
    result = dict(
        stage=stage,
        baseline="Kumo Large recipe with public 0.56262; reevaluated on matching folds",
        warning="Repeated model selection on the same training labels; confirmation with new splits is not an untouched holdout.",
        comparisons=rows,
    )
    (path / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_parquet(path / "comparison_oof.parquet", index=False)
    pd.DataFrame(rows).drop(columns=["components", "fold_ks", "fold_deltas"]).to_csv(
        path / "comparison.csv", index=False
    )
    return result


def export(stage, name, filename="submission_kumo_v2.csv"):
    summary = summarize(stage)
    row = next(r for r in summary["comparisons"] if r["name"] == name)
    splits = int(re.fullmatch(r"cv(\d+)_seed(\d+)", stage).group(1))
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    pred = np.zeros(len(sample), dtype=np.float64)
    source_hashes = {}
    for component in row["components"]:
        directory = (
            ROOT / "artifacts/optimization/kumo/large_v5_ctx50000_e8"
            if stage == "cv3_seed8247" and component["name"] == "baseline_e8"
            else DEST / stage / component["name"]
        )
        for fold in range(splits):
            source = directory / f"fold{fold}_test.parquet"
            frame = pd.read_parquet(source)
            assert frame.application_id.is_unique and len(frame) == len(sample)
            ordered = sample[["application_id"]].merge(
                frame, on="application_id", validate="one_to_one"
            )
            assert ordered.application_id.equals(sample.application_id)
            pred += (
                ordered.prediction.astype("float64").to_numpy()
                * component["weight"]
                / splits
            )
            source_hashes[str(source.relative_to(ROOT))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    assert filename.startswith("submission_kumo_v2") and filename.endswith(".csv")
    result = sample[["application_id"]].copy()
    result["default_probability"] = pred
    path = ROOT / filename
    result.to_csv(path, index=False)
    meta = dict(
        file=filename,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=len(result),
        stage=stage,
        recipe=name,
        validation=row,
        source_prediction_sha256=source_hashes,
        public_score=None,
        replaces_public_checkpoint=False,
    )
    (DEST / "submission_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="cv3_seed8247")
    parser.add_argument("--export")
    args = parser.parse_args()
    if args.export:
        print(json.dumps(export(args.stage, args.export), indent=2))
    else:
        summary = summarize(args.stage)
        print(
            pd.DataFrame(summary["comparisons"])
            .drop(columns=["components", "fold_ks", "fold_deltas"])
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
