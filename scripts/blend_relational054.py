"""Fixed blends of completed relational and context-correction OOF predictions.

Weights are declared before observing any relational fold score. No meta-model
is fitted to OOF labels. Independent audit rebuilds both OOF and test mixtures.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r

STAGE = r.DEST / "cv5_seed31027"
RELATIONAL = "kumo_relational_v5_e8_ctx50000_h1_readout_b8192"
CORRECTION = "kumo_offset_l20p003_unweighted_d0p5"


def audit_blend(data, test, fold, ti, vi, directory, meta):
    components = meta["recipe"]["components"]
    assert (
        len(components) == 2 and abs(sum(c["weight"] for c in components) - 1) < 1e-15
    )
    maximum = 0.0
    for kind, anchor in [("oof", data.iloc[vi]), ("test", test)]:
        expected = np.zeros(len(anchor))
        for component in components:
            source = STAGE / component["candidate"]
            md_path = source / f"fold{fold}_metadata.json"
            proof = meta["source_metadata"][component["candidate"]]
            assert r.checksum(md_path) == proof["metadata_sha256"]
            md = json.loads(md_path.read_text())
            assert md["training_ids_sha256"] == r.ids_hash(data, ti)
            assert md["validation_ids_sha256"] == r.ids_hash(data, vi)
            path = source / f"fold{fold}_{kind}.parquet"
            assert r.checksum(path) == md[f"{kind}_sha256"]
            values = pd.read_parquet(path)
            assert values.application_id.tolist() == anchor.application_id.tolist()
            if kind == "oof":
                assert np.array_equal(values.target, anchor.target.astype(int))
            expected += component["weight"] * values.prediction.to_numpy(dtype=float)
        actual = pd.read_parquet(directory / f"fold{fold}_{kind}.parquet")
        maximum = max(maximum, float(np.max(np.abs(actual.prediction - expected))))
    assert maximum < 2e-14
    return dict(
        fixed_weights=True,
        oof_labels_used_to_fit_weights=False,
        source_hashes_and_fold_isolation_verified=True,
        rebuilt_prediction_max_abs_difference=maximum,
    )


def run():
    plan = json.loads((r.DEST / "relational_blend_plan.json").read_text())
    assert (
        plan["relational_candidate"] == RELATIONAL
        and plan["correction_candidate"] == CORRECTION
    )
    assert plan["relational_weights"] == [0.25, 0.5, 0.75]
    assert not plan["relational_fold_scores_observed_at_freeze"]
    sources = {name: STAGE / name for name in [RELATIONAL, CORRECTION]}
    for source in sources.values():
        assert all(
            (source / f"fold{i}_metadata.json").exists() for i in range(5)
        ), "Blending requires five completed source folds."
    definitions = {
        name: json.loads((path / "definition.json").read_text())
        for name, path in sources.items()
    }
    columns = list(
        dict.fromkeys(
            c
            for definition in definitions.values()
            for c in definition.get("spec", definition)["columns"]
        )
    )
    # The correction includes a few spline-only columns beyond the TFM inputs.
    data, test = r.inputs(r.specifications() + [dict(columns=columns)])
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=31027).split(data, data.target)
    )
    for weight in plan["relational_weights"]:
        name = f"relational_context_fixed_w{str(weight).replace('.', 'p')}"
        directory = STAGE / name
        recipe = dict(
            family="fixed_ensemble",
            columns=columns,
            n_splits=5,
            fold_seed=31027,
            components=[
                dict(candidate=RELATIONAL, weight=weight),
                dict(candidate=CORRECTION, weight=1 - weight),
            ],
            parameter_plan_sha256=r.checksum(r.DEST / "relational_blend_plan.json"),
            training_sha256=r.fingerprint(data, ["application_id", "target"] + columns),
            test_sha256=r.fingerprint(test, ["application_id"] + columns),
        )
        definition = directory / "definition.json"
        if definition.exists():
            assert json.loads(definition.read_text()) == recipe
        else:
            r.write_json(definition, recipe)
        for fold, (ti, vi) in enumerate(folds):
            output = {}
            proofs = {}
            for kind, anchor in [("oof", data.iloc[vi]), ("test", test)]:
                expected = np.zeros(len(anchor))
                for component in recipe["components"]:
                    source = sources[component["candidate"]]
                    md_path = source / f"fold{fold}_metadata.json"
                    md = json.loads(md_path.read_text())
                    assert md["training_ids_sha256"] == r.ids_hash(data, ti)
                    assert md["validation_ids_sha256"] == r.ids_hash(data, vi)
                    path = source / f"fold{fold}_{kind}.parquet"
                    assert r.checksum(path) == md[f"{kind}_sha256"]
                    values = pd.read_parquet(path)
                    assert (
                        values.application_id.tolist() == anchor.application_id.tolist()
                    )
                    expected += component["weight"] * values.prediction.to_numpy(
                        dtype=float
                    )
                    proofs[component["candidate"]] = dict(
                        metadata_sha256=r.checksum(md_path)
                    )
                result = anchor[
                    (
                        ["application_id", "target"]
                        if kind == "oof"
                        else ["application_id"]
                    )
                ].copy()
                result["prediction"] = expected
                result.to_parquet(directory / f"fold{fold}_{kind}.parquet", index=False)
                output[kind] = directory / f"fold{fold}_{kind}.parquet"
            meta = dict(
                recipe=recipe,
                fold=fold,
                training_ids_sha256=r.ids_hash(data, ti),
                validation_ids_sha256=r.ids_hash(data, vi),
                source_metadata=proofs,
                metrics=r.scores(
                    data.target.iloc[vi], pd.read_parquet(output["oof"]).prediction
                ),
                **{kind + "_sha256": r.checksum(path) for kind, path in output.items()},
            )
            audit_blend(data, test, fold, ti, vi, directory, meta)
            r.write_json(directory / f"fold{fold}_metadata.json", meta)
    print("FIXED RELATIONAL BLENDS READY", flush=True)


if __name__ == "__main__":
    run()
