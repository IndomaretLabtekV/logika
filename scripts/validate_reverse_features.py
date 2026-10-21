"""Test whether estimated generator signals improve a model on the existing three folds."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics, ks
from model_v5 import fit_model, predict_model
from spline_model import FEATURES

DESTINATION = ROOT / "artifacts/reverse_engineering"


def main():
    full = pd.read_parquet(ROOT / "data/processed/application_features_reverse.parquet")
    data = full[full.split.eq("train")].reset_index(drop=True)
    new_columns = [
        c for c in data if c.startswith("re_") and data[c].nunique(dropna=True) > 1
    ]
    tree_definitions = json.loads((OUT / "v4/definitions.json").read_text())
    tree_columns = next(
        s["columns"] for s in tree_definitions if s["name"] == "v4_cat5_balanced"
    )
    spline_columns = [c for c in FEATURES if c in data]
    specs = [
        {
            "name": "reverse_cat5",
            "family": "cat",
            "columns": tree_columns + new_columns,
            "iterations": 500,
            "params": {"depth": 5, "l2_leaf_reg": 25, "class_weights": [1, 6]},
        },
        {
            "name": "reverse_spline",
            "family": "spline_plus",
            "columns": spline_columns + new_columns,
            "params": {
                "C": 0.05,
                "n_knots": 5,
                "positive_weight": 1,
                "interactions": False,
            },
        },
    ]
    previous = pd.read_parquet(OUT / "v5/selection_oof_tabm.parquet")
    assert previous.application_id.equals(
        data.application_id
    ) and previous.target.astype(int).equals(data.target.astype(int))
    oof = previous[["application_id", "target"]].copy()
    oof["version5"] = 0.75 * previous.version4 + 0.25 * previous.v5_tabm_weight1
    y = data.target.astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=8247).split(data, y))
    model_dir = DESTINATION / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "validation_definitions.json").write_text(
        json.dumps(specs, indent=2)
    )
    rows = []
    for spec in specs:
        oof[spec["name"]] = np.nan
        for fold, (ti, vi) in enumerate(folds, 1):
            start = time.monotonic()
            print("Training", spec["name"], "fold", fold, flush=True)
            bundle = fit_model(spec, data, ti, seed=2026)
            pv = predict_model(bundle, data.iloc[vi])
            oof.loc[vi, spec["name"]] = pv
            joblib.dump(
                bundle, model_dir / f'fold{fold}_{spec["name"]}.joblib', compress=3
            )
            rows.append(
                {
                    "name": spec["name"],
                    "fold": fold,
                    **metrics(y.iloc[vi], pv),
                    "seconds": time.monotonic() - start,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            oof.to_parquet(DESTINATION / "validation_oof.parquet", index=False)
    comparison = []
    for name in oof.columns[2:]:
        for weight in [0] if name == "version5" else [0.25, 0.5, 1.0]:
            p = (
                oof.version5
                if name == "version5"
                else (1 - weight) * oof.version5 + weight * oof[name]
            )
            scores = metrics(y, p)
            values = [ks(y.iloc[vi], p.iloc[vi]) for _, vi in folds]
            comparison.append(
                {
                    "name": name,
                    "new_model_weight": weight,
                    **scores,
                    "fold_ks": values,
                    "mean_fold_ks": float(np.mean(values)),
                }
            )
    report = {
        "fold_seed": 8247,
        "fold_count": 3,
        "model_seed": 2026,
        "baseline": "75% version4 + 25% TabM on the original three folds",
        "new_variable_features": new_columns,
        "comparison": comparison,
        "submission_changed": False,
        "caveat": "Local screening; feature/model/weight selection reuses labels. Latent feature parameters use unlabeled histories, not target labels.",
    }
    (DESTINATION / "validation_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    (DESTINATION / "validation_fold_metrics.json").write_text(
        json.dumps(rows, indent=2)
    )
    print("Final comparison:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
