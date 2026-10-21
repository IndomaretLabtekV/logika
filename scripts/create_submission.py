"""Fit the validated all-history baseline on every label and export submission.csv."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "artifacts/model_config.json").read_text())
    frame = pd.read_parquet(ROOT / "data/processed/application_features.parquet")
    train = frame[frame.split.eq("train")].copy()
    test = frame[frame.split.eq("test")].copy()
    raw_train = pd.read_csv(ROOT / "data/train_applications.csv", usecols=["application_id", "target"])
    raw_test = pd.read_csv(ROOT / "data/test_applications.csv", usecols=["application_id"])
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    columns = config["all_feature_columns"]

    assert len(train) == len(raw_train) == 70000
    assert len(test) == len(raw_test) == len(sample) == 30000
    assert train.application_id.is_unique and test.application_id.is_unique
    assert sample.application_id.is_unique
    assert set(train.application_id) == set(raw_train.application_id)
    assert set(test.application_id) == set(raw_test.application_id) == set(sample.application_id)
    assert set(train.customer_id).isdisjoint(test.customer_id)
    assert set(train.application_id).isdisjoint(test.application_id)
    assert train.target.isin([0, 1]).all() and test.target.isna().all()
    labels = raw_train.set_index("application_id").target.reindex(train.application_id).to_numpy()
    assert np.array_equal(train.target.to_numpy(), labels)
    assert not set(columns).intersection(config["excluded_columns"])
    assert set(columns).issubset(frame.columns)

    categorical = [c for c in columns if train[c].dtype == object]
    numeric = [c for c in columns if c not in categorical]
    assert not np.isinf(frame[numeric].to_numpy()).any()
    encoder = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)
    parameters = dict(config["hyperparameters"])
    parameters.update(objective="binary:logistic", eval_metric="logloss", random_state=config["seed"])
    model = Pipeline([("encode", encoder), ("model", XGBClassifier(**parameters))])
    print(f"Training XGBoost on {len(train):,} labels and {len(columns)} features.", flush=True)
    model.fit(train[columns], train.target.astype(int))
    assert np.array_equal(model.named_steps["model"].classes_, [0, 1])
    probabilities = model.predict_proba(test[columns])[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    predictions = pd.DataFrame({"application_id": test.application_id.to_numpy(),
                                "default_probability": probabilities})
    submission = sample[["application_id"]].merge(
        predictions, on="application_id", how="left", validate="one_to_one", sort=False)
    assert submission.application_id.equals(sample.application_id)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission.default_probability.notna().all()

    destination = ROOT / "results/teman/submission.csv"
    temporary = destination.with_suffix(".csv.tmp")
    submission.to_csv(temporary, index=False, float_format="%.12f")
    reread = pd.read_csv(temporary)
    assert reread.shape == (30000, 2)
    assert reread.columns.tolist() == ["application_id", "default_probability"]
    assert reread.application_id.equals(sample.application_id)
    assert reread.application_id.is_unique
    assert reread.default_probability.notna().all()
    assert reread.default_probability.between(0, 1).all()
    assert np.allclose(reread.default_probability, submission.default_probability, rtol=0, atol=6e-13)
    temporary.replace(destination)

    artifacts = ROOT / "artifacts"
    joblib.dump(model, artifacts / "submission_model.joblib", compress=3)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": "data-science-competition-logika-ui-2026",
        "file": "submission.csv",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "train_rows": len(train), "test_rows": len(test), "feature_count": len(columns),
        "model": "XGBoost all-history baseline, fitted on all training labels",
        "parameters": parameters,
        "columns": reread.columns.tolist(),
        "probability_summary": reread.default_probability.describe().to_dict(),
        "checks": {"matches_sample_order": True, "unique_ids": True,
                   "matches_test_id_set": True, "no_missing": True,
                   "all_probabilities_between_0_and_1": True, "no_exported_index": True},
    }
    (artifacts / "submission_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Created {destination}: 30,000 rows; all format and probability checks passed.", flush=True)
    print(json.dumps(metadata["probability_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
