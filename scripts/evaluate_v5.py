"""Screen fixed candidates on the same three folds as the version-4 selection."""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics
from model_v5 import (
    definitions,
    basket_definitions,
    shallow_definitions,
    neural_definitions,
    core_neural_definitions,
    smoothing_definitions,
    fit_model,
    predict_model,
    REVISED_FEATURES,
)

DESTINATION = OUT / "v5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=[
            "spline_plus",
            "ebm",
            "xgb",
            "specialist",
            "basket",
            "shallow",
            "tabm",
            "neuralcore",
            "smoothing",
        ],
        required=True,
    )
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    file_name = (
        "application_features_v5.parquet"
        if args.group in ["basket", "shallow", "tabm", "neuralcore"]
        else "application_features_v4.parquet"
    )
    full = pd.read_parquet(ROOT / "data/processed" / file_name)
    data = full[full.split.eq("train")].reset_index(drop=True)
    test = full[full.split.eq("test")].reset_index(drop=True)
    v4defs = json.loads((OUT / "v4/definitions.json").read_text())
    tree_columns = next(s["columns"] for s in v4defs if s["name"] == "v4_cat5_balanced")
    specs = [s for s in definitions(tree_columns) if s["family"] == args.group]
    if args.group == "basket":
        specs = basket_definitions(
            tree_columns, [c for c in data if c.startswith("v5_")], data.columns
        )
    if args.group == "shallow":
        specs = shallow_definitions(
            tree_columns, [c for c in data if c.startswith("v5_")]
        )
    if args.group == "tabm":
        specs = neural_definitions()
    if args.group == "neuralcore":
        specs = core_neural_definitions()
    if args.group == "smoothing":
        specs = smoothing_definitions(data.columns)
    assert set(REVISED_FEATURES).issubset(data.columns)
    (DESTINATION / f"definitions_{args.group}.json").write_text(
        json.dumps(specs, indent=2)
    )
    old = pd.read_parquet(OUT / "v4/selection_oof.parquet")
    old_spline = pd.read_parquet(OUT / "v4/spline_oof.parquet")
    old_test = pd.read_parquet(OUT / "v4/selection_test_predictions.parquet")
    old_spline_test = pd.read_parquet(OUT / "v4/spline_test_predictions.parquet")
    assert old.application_id.equals(
        data.application_id
    ) and old_spline.application_id.equals(data.application_id)
    assert old.target.astype(int).equals(data.target.astype(int))
    assert old_test.application_id.equals(
        test.application_id
    ) and old_spline_test.application_id.equals(test.application_id)
    baseline = (old.public_055845 + old.v4_cat5_balanced_500 + old_spline.spline) / 3
    baseline_test = (
        old_test.public_055845 + old_test.v4_cat5_balanced_500 + old_spline_test.spline
    ) / 3
    oof = pd.DataFrame(
        {
            "application_id": data.application_id,
            "target": data.target.astype(int),
            "version4": baseline,
        }
    )
    predictions = pd.DataFrame(
        {"application_id": test.application_id, "version4": baseline_test}
    )
    for spec in specs:
        names = (
            [f'{spec["name"]}_{n}' for n in spec["stages"]]
            if "stages" in spec
            else [spec["name"]]
        )
        for name in names:
            oof[name] = np.nan
            predictions[name] = 0.0
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(
            data, data.target
        )
    )
    rows = []
    for fold, (ti, vi) in enumerate(folds, 1):
        for spec in specs:
            start = time.monotonic()
            print(f'Training fold {fold}/3: {spec["name"]}', flush=True)
            bundle = fit_model(spec, data, ti)
            for trees in spec.get("stages", [None]):
                name = spec["name"] if trees is None else f'{spec["name"]}_{trees}'
                pv = predict_model(bundle, data.iloc[vi], trees)
                pt = predict_model(bundle, test, trees)
                assert np.isfinite(pv).all() and np.isfinite(pt).all()
                oof.loc[vi, name] = pv
                predictions[name] += pt / 3
                scores = metrics(data.target.iloc[vi].astype(int), pv)
                blends = {
                    str(w): metrics(
                        data.target.iloc[vi].astype(int),
                        (1 - w) * baseline.iloc[vi] + w * pv,
                    )["ks"]
                    for w in [0.25, 0.5]
                }
                rows.append({"fold": fold, "name": name, **scores})
                print(
                    json.dumps(
                        {"fold": fold, "name": name, **scores, "blend_ks": blends}
                    ),
                    flush=True,
                )
            joblib.dump(
                bundle, DESTINATION / f'fold{fold}_{spec["name"]}.joblib', compress=3
            )
            oof.to_parquet(
                DESTINATION / f"selection_oof_{args.group}.parquet", index=False
            )
            predictions.to_parquet(
                DESTINATION / f"selection_test_{args.group}.parquet", index=False
            )
            pd.DataFrame(rows).to_csv(
                DESTINATION / f"fold_metrics_{args.group}.csv", index=False
            )
            print(
                "Completed in",
                round(time.monotonic() - start, 1),
                "seconds.",
                flush=True,
            )
    assert oof.notna().all().all()
    summary = []
    for name in oof.columns[2:]:
        per_fold = [
            metrics(data.target.iloc[vi].astype(int), oof[name].iloc[vi])["ks"]
            for _, vi in folds
        ]
        summary.append(
            {
                "name": name,
                **metrics(data.target.astype(int), oof[name]),
                "mean_fold_ks": float(np.mean(per_fold)),
                "fold_ks": per_fold,
            }
        )
    (DESTINATION / f"summary_{args.group}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("Final comparison:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
