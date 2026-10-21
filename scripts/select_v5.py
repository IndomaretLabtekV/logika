"""Compare fixed coarse blends against version 4 on identical three-fold OOF rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics, ks

DESTINATION = OUT / "v5"


def main():
    files = sorted(DESTINATION.glob("summary_*.json"))
    frames = [
        pd.read_parquet(
            DESTINATION
            / ("selection_oof_" + f.stem.removeprefix("summary_") + ".parquet")
        )
        for f in files
    ]
    assert frames, "No completed screening group."
    data = frames[0]
    combined = data[["application_id", "target", "version4"]].copy()
    definitions = {}
    for path, frame in zip(files, frames):
        assert frame.application_id.equals(data.application_id) and frame.target.equals(
            data.target
        )
        assert np.allclose(frame.version4, data.version4, rtol=0, atol=0)
        columns = [c for c in frame.columns[3:] if frame[c].notna().all()]
        for c in columns:
            combined[c] = frame[c]
        specs = json.loads(
            (
                DESTINATION
                / ("definitions_" + path.stem.removeprefix("summary_") + ".json")
            ).read_text()
        )
        for spec in specs:
            for trees in spec.get("stages", [None]):
                name = spec["name"] if trees is None else f'{spec["name"]}_{trees}'
                definitions[name] = dict(spec, name=name, cache_name=spec["name"])
                if trees is not None:
                    definitions[name]["iterations"] = trees
    y = combined.target.astype(int)
    folds = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=8247).split(combined, y)
    )
    old_fold = np.array([ks(y.iloc[vi], combined.version4.iloc[vi]) for _, vi in folds])
    baseline = metrics(y, combined.version4)
    rows = []
    for name in combined.columns[3:]:
        for w in [0.25, 0.5]:
            p = (1 - w) * combined.version4 + w * combined[name]
            per_fold = np.array([ks(y.iloc[vi], p.iloc[vi]) for _, vi in folds])
            scores = metrics(y, p)
            gains = per_fold - old_fold
            eligible = (
                scores["ks"] > baseline["ks"] + 0.001
                and np.mean(gains) > 0.0005
                and (gains > 0).sum() >= 2
                and gains.min() > -0.0025
            )
            rows.append(
                {
                    "name": name,
                    "new_weight": w,
                    **scores,
                    "fold_ks": per_fold.tolist(),
                    "fold_gains": gains.tolist(),
                    "mean_fold_ks": float(per_fold.mean()),
                    "eligible": bool(eligible),
                }
            )
    rows.sort(key=lambda r: (r["eligible"], r["mean_fold_ks"], r["ks"]), reverse=True)
    (DESTINATION / "blend_ranking.json").write_text(json.dumps(rows, indent=2))
    combined.to_parquet(DESTINATION / "selection_all_oof.parquet", index=False)
    print("Best blends:", json.dumps(rows[:10]), flush=True)
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        print("No eligible improvement yet; version 4 remains unchanged.", flush=True)
        return
    selected = eligible[0]
    recipe = {
        "version4": 1 - selected["new_weight"],
        selected["name"]: selected["new_weight"],
    }
    locked = {
        "recipe": recipe,
        "spec": definitions[selected["name"]],
        "selection": selected,
        "baseline": {
            **baseline,
            "fold_ks": old_fold.tolist(),
            "mean_fold_ks": float(old_fold.mean()),
        },
        "selection_fold_seed": 8247,
        "selection_folds": 3,
        "final_fold_seed": 9173,
        "final_folds": 10,
        "final_seed_rule": "2026+fold",
        "feature_file": "application_features_v5.parquet",
        "new_public_score": None,
        "rule": "Coarse weights 25%/50%; pooled KS gain >.001, mean fold gain >.0005, at least 2 wins, worst fold loss <.0025.",
        "caveat": "Selection reuses training labels and contains selection optimism; no test labels used.",
    }
    (DESTINATION / "locked_recipe.json").write_text(json.dumps(locked, indent=2))
    print("Locked recipe:", recipe, flush=True)


if __name__ == "__main__":
    main()
