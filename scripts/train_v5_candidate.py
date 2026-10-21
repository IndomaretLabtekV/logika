"""Confirm a fixed candidate on version 4's exact ten outer folds."""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics, ks
from model_v5 import fit_model, predict_model

DESTINATION = OUT / "v5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="v5_tabm_weight1")
    parser.add_argument("--new-weight", type=float, default=0.25)
    args = parser.parse_args()
    assert 0 < args.new_weight <= 1
    specs = [
        s
        for f in sorted(DESTINATION.glob("definitions_*.json"))
        for s in json.loads(f.read_text())
    ]
    spec = next(s for s in specs if s["name"] == args.candidate)
    assert "stages" not in spec, "Lock the selected tree count before confirmation."
    manifest = {
        "spec": spec,
        "recipe": {"version4": 1 - args.new_weight, args.candidate: args.new_weight},
        "fold_count": 10,
        "fold_seed": 9173,
        "seed_rule": "2026+fold",
        "selection_status": "Exploratory confirmation. Three-fold pooled and mean KS improved slightly; only one of three fold scores improved.",
        "promotion_rule": "Pooled KS and mean-fold KS must improve, with at least six of ten fold wins.",
        "public_score": None,
        "test_labels_used": False,
    }
    (DESTINATION / f"confirmation_recipe_{args.candidate}.json").write_text(
        json.dumps(manifest, indent=2)
    )
    full = pd.read_parquet(ROOT / "data/processed/application_features_v5.parquet")
    data = full[full.split.eq("train")].reset_index(drop=True)
    test = full[full.split.eq("test")].reset_index(drop=True)
    old = pd.read_parquet(OUT / "v4/combined_final_oof.parquet")
    old_test = pd.read_parquet(OUT / "v4/combined_final_test.parquet")
    assert old.application_id.equals(data.application_id) and old.target.astype(
        int
    ).equals(data.target.astype(int))
    assert old_test.application_id.equals(test.application_id)
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=9173).split(data, y)
    )
    oof = old[["application_id", "target", "version4"]].copy()
    oof[args.candidate] = np.nan
    predicted = old_test[["application_id", "version4"]].copy()
    predicted[args.candidate] = 0.0
    model_dir = DESTINATION / "final_models"
    model_dir.mkdir(exist_ok=True)
    rows = []
    for fold, (ti, vi) in enumerate(folds, 1):
        start = time.monotonic()
        path = model_dir / f"fold{fold}_{args.candidate}.joblib"
        print(f"Confirming {args.candidate}: fold {fold}/10", flush=True)
        if path.exists():
            bundle = joblib.load(path)
            assert (
                bundle["columns"] == spec["columns"]
                and bundle["seed"] == 2026 + fold
                and bundle["params"] == spec["params"]
            )
        else:
            bundle = fit_model(spec, data, ti, seed=2026 + fold)
            joblib.dump(bundle, path, compress=3)
        pv = predict_model(bundle, data.iloc[vi])
        pt = predict_model(bundle, test)
        oof.loc[vi, args.candidate] = pv
        predicted[args.candidate] += pt / 10
        p = (1 - args.new_weight) * old.version4.iloc[vi] + args.new_weight * pv
        baseline = metrics(y.iloc[vi], old.version4.iloc[vi])
        scores = metrics(y.iloc[vi], p)
        rows.append(
            {
                "fold": fold,
                "baseline": baseline,
                "candidate": metrics(y.iloc[vi], pv),
                "new": scores,
                "ks_gain": scores["ks"] - baseline["ks"],
                "seconds": time.monotonic() - start,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        oof.to_parquet(DESTINATION / f"final_oof_{args.candidate}.parquet", index=False)
        predicted.to_parquet(
            DESTINATION / f"final_test_{args.candidate}.parquet", index=False
        )
        (DESTINATION / f"final_metrics_{args.candidate}.json").write_text(
            json.dumps(rows, indent=2)
        )
    assert oof.notna().all().all()
    oof["version5"] = (1 - args.new_weight) * oof.version4 + args.new_weight * oof[
        args.candidate
    ]
    predicted["version5"] = (
        1 - args.new_weight
    ) * predicted.version4 + args.new_weight * predicted[args.candidate]
    baseline = metrics(y, oof.version4)
    candidate = metrics(y, oof.version5)
    baseline["mean_fold_ks"] = float(np.mean([r["baseline"]["ks"] for r in rows]))
    candidate["mean_fold_ks"] = float(np.mean([r["new"]["ks"] for r in rows]))
    wins = sum(r["ks_gain"] > 0 for r in rows)
    eligible = (
        candidate["ks"] > baseline["ks"]
        and candidate["mean_fold_ks"] > baseline["mean_fold_ks"]
        and wins >= 6
    )
    report = {
        "baseline": baseline,
        "version5": candidate,
        "fold_wins": wins,
        "promote": bool(eligible),
        "recipe": manifest,
    }
    oof.to_parquet(DESTINATION / f"final_oof_{args.candidate}.parquet", index=False)
    predicted.to_parquet(
        DESTINATION / f"final_test_{args.candidate}.parquet", index=False
    )
    (DESTINATION / f"confirmation_{args.candidate}.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Confirmation:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
