"""Independently audit held-out predictions, KS, contexts and submission alignment."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

from improve_kumo import DEST
from kumo_tabular import EXCLUDED, ROOT, context_indices


def main():
    checkpoint = ROOT / "results/teman/submission_kumo.csv"
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == "85ac74d1b4325eba4830fd1d31bb1c21b9b9a3a6fb180d48c0c9d4b921d59d40"
    )
    data = pd.read_parquet(
        ROOT / "data/processed/application_features_v5.parquet",
        columns=["application_id", "target", "split"],
    )
    data = data[data.split.eq("train")].reset_index(drop=True)
    y = data.target.astype(int)
    verified = []
    for stage in sorted(DEST.glob("cv*_seed*")):
        splits, seed = map(int, stage.name.removeprefix("cv").split("_seed"))
        folds = list(
            StratifiedKFold(splits, shuffle=True, random_state=seed).split(data, y)
        )
        for directory in sorted(stage.iterdir()):
            if not directory.is_dir():
                continue
            complete = []
            for fold, (ti, vi) in enumerate(folds):
                metafile = directory / f"fold{fold}_metadata.json"
                if not metafile.exists():
                    continue
                meta = json.loads(metafile.read_text())
                spec = meta["recipe"]["spec"]
                assert not set(spec["columns"]) & EXCLUDED
                held = pd.read_parquet(directory / f"fold{fold}_oof.parquet")
                assert held.application_id.is_unique
                assert np.array_equal(held.application_id, data.application_id.iloc[vi])
                assert np.array_equal(held.target, y.iloc[vi])
                assert (
                    np.isfinite(held.prediction).all()
                    and held.prediction.between(0, 1).all()
                )
                independent_ks = ks_2samp(
                    held.loc[held.target.eq(1), "prediction"],
                    held.loc[held.target.eq(0), "prediction"],
                ).statistic
                assert abs(independent_ks - meta["metrics"]["ks"]) < 1e-12
                model_seed = 2026 + fold + spec["seed_offset"]
                ci = context_indices(ti, y, spec["context_limit"], model_seed)
                assert not len(np.intersect1d(ci, vi))
                expected_hash = hashlib.sha256(
                    "\n".join(data.application_id.iloc[ci]).encode()
                ).hexdigest()
                if "context_ids_sha256" in meta:
                    assert expected_hash == meta["context_ids_sha256"]
                if "negative_ratio" in spec:
                    positive, negative = ci[y.iloc[ci].eq(1)], ci[y.iloc[ci].eq(0)]
                    n = min(len(negative), int(len(positive) * spec["negative_ratio"]))
                    rng = np.random.default_rng(model_seed)
                    members = [
                        rng.permutation(
                            np.concatenate(
                                [positive, rng.choice(negative, n, replace=False)]
                            )
                        )
                        for _ in range(spec["estimators"])
                    ]
                    hashes = [
                        hashlib.sha256(m.astype(np.int64).tobytes()).hexdigest()
                        for m in members
                    ]
                    assert hashes == meta["context_sampling"]["member_index_sha256"]
                    assert all(not len(np.intersect1d(m, vi)) for m in members)
                complete.append(fold)
            if complete:
                verified.append(
                    dict(stage=stage.name, candidate=directory.name, folds=complete)
                )
    submission = None
    metafile = DEST / "submission_metadata.json"
    if metafile.exists():
        meta = json.loads(metafile.read_text())
        source = ROOT / meta["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["sha256"]
        sample = pd.read_csv(ROOT / "data/sample_submission.csv")
        actual = pd.read_csv(source)
        assert actual.columns.tolist() == sample.columns.tolist()
        assert (
            actual.application_id.equals(sample.application_id)
            and actual.application_id.is_unique
        )
        assert len(actual) == 30000
        assert (
            np.isfinite(actual.default_probability).all()
            and actual.default_probability.between(0, 1).all()
        )
        expected = np.zeros(len(sample), dtype=np.float64)
        splits = int(meta["stage"].split("_")[0][2:])
        for c in meta["validation"]["components"]:
            for fold in range(splits):
                file = DEST / meta["stage"] / c["name"] / f"fold{fold}_test.parquet"
                assert (
                    hashlib.sha256(file.read_bytes()).hexdigest()
                    == meta["source_prediction_sha256"][str(file.relative_to(ROOT))]
                )
                part = pd.read_parquet(file)
                aligned = sample[["application_id"]].merge(
                    part, on="application_id", validate="one_to_one"
                )
                expected += (
                    aligned.prediction.to_numpy(dtype=np.float64) * c["weight"] / splits
                )
        max_difference = float(np.max(np.abs(expected - actual.default_probability)))
        assert max_difference < 2e-15
        submission = dict(
            file=meta["file"],
            sha256=meta["sha256"],
            rows=len(actual),
            source_ensemble_max_abs_difference=max_difference,
            public_score=None,
        )
    result = dict(
        scipy_ks_agrees=True,
        all_contexts_disjoint_from_validation=True,
        checkpoint_056262_unchanged=True,
        verified=verified,
        submission=submission,
    )
    (DEST / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
