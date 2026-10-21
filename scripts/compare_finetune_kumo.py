"""Compare fine-tuned Kumo against the frozen recipe on identical outer folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from finetune_kumo import DEST, NAME, ROOT, frozen_sources
from kumo_tabular import ks, metrics
from summarize_kumo_improvement import read_oof


def compare(stage):
    match = re.fullmatch(r"cv(\d+)_seed(\d+)", stage)
    assert match
    splits, seed = map(int, match.groups())
    anchor = pd.read_parquet(
        ROOT / "data/processed/application_features_v5.parquet",
        columns=["application_id", "target", "split"],
    )
    anchor = anchor.loc[
        anchor.split.eq("train"), ["application_id", "target"]
    ].reset_index(drop=True)
    baseline_directory = frozen_sources(stage, 0)[0].parent
    baseline = read_oof(baseline_directory, anchor, splits)
    fine = read_oof(DEST / stage / NAME, anchor, splits)
    y = anchor.target.astype(int)
    folds = list(
        StratifiedKFold(splits, shuffle=True, random_state=seed).split(anchor, y)
    )
    base_folds = np.array([ks(y.iloc[vi], baseline[vi]) for _, vi in folds])
    rows = []
    oof = anchor.copy()
    oof["baseline"] = baseline
    oof["finetuned"] = fine

    def add(name, p, components):
        per_fold = np.array([ks(y.iloc[vi], p[vi]) for _, vi in folds])
        rows.append(
            dict(
                name=name,
                **metrics(y, p),
                mean_fold_ks=float(per_fold.mean()),
                fold_ks=per_fold.tolist(),
                fold_deltas=(per_fold - base_folds).tolist(),
                delta_ks=ks(y, p) - ks(y, baseline),
                delta_mean_fold_ks=float((per_fold - base_folds).mean()),
                fold_wins=int((per_fold > base_folds).sum()),
                components=components,
            )
        )

    add("frozen_baseline", baseline, [dict(name="baseline_e8", weight=1.0)])
    add("finetuned", fine, [dict(name=NAME, weight=1.0)])
    for weight in [0.25, 0.5]:
        name = f"baseline_plus_finetuned_w{weight}"
        p = (1 - weight) * baseline + weight * fine
        oof[name] = p
        add(
            name,
            p,
            [
                dict(name="baseline_e8", weight=1 - weight),
                dict(name=NAME, weight=weight),
            ],
        )
    summary = dict(
        stage=stage,
        comparisons=rows,
        warning="Checkpoint selection uses inner labels only. Outer-fold results still reuse labels previously used for model/feature selection; they are not an untouched holdout.",
    )
    path = DEST / stage
    path.mkdir(parents=True, exist_ok=True)
    (path / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    oof.to_parquet(path / "comparison_oof.parquet", index=False)
    pd.DataFrame(rows).drop(columns=["components", "fold_ks", "fold_deltas"]).to_csv(
        path / "comparison.csv", index=False
    )
    return summary


def export(stage, name, filename="submission_kumo_v2.csv"):
    summary = compare(stage)
    selected = next(r for r in summary["comparisons"] if r["name"] == name)
    splits = int(re.fullmatch(r"cv(\d+)_seed(\d+)", stage).group(1))
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    prediction = np.zeros(len(sample), dtype=np.float64)
    sources = {}
    for component in selected["components"]:
        for fold in range(splits):
            source = (
                frozen_sources(stage, fold)[1]
                if component["name"] == "baseline_e8"
                else DEST / stage / NAME / f"fold{fold}_test.parquet"
            )
            part = pd.read_parquet(source)
            assert len(part) == len(sample) and part.application_id.is_unique
            ordered = sample[["application_id"]].merge(
                part, on="application_id", validate="one_to_one"
            )
            assert ordered.application_id.equals(sample.application_id)
            prediction += (
                ordered.prediction.to_numpy(dtype=np.float64)
                * component["weight"]
                / splits
            )
            sources[str(source.relative_to(ROOT))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
    assert (
        len(prediction) == 30000
        and np.isfinite(prediction).all()
        and ((prediction >= 0) & (prediction <= 1)).all()
    )
    assert filename == "submission_kumo_v2.csv"
    submission = sample[["application_id"]].copy()
    submission["default_probability"] = prediction
    file = ROOT / filename
    submission.to_csv(file, index=False)
    meta = dict(
        file=filename,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        rows=len(submission),
        stage=stage,
        selected=selected,
        source_prediction_sha256=sources,
        public_score=None,
        preserves_checkpoint="submission_kumo.csv",
        frozen_checkpoint_public_score=0.56262,
    )
    (DEST / "submission_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="cv3_seed8247")
    parser.add_argument("--export")
    args = parser.parse_args()
    result = export(args.stage, args.export) if args.export else compare(args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
