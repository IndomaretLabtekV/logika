"""Independently audit complete goal-0.54 experiments and export a candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r


def audit():
    spline_spec = next(
        s
        for s in json.loads(
            (
                r.ROOT / "artifacts/reverse_engineering/validation_definitions.json"
            ).read_text()
        )
        if s["name"] == "reverse_spline"
    )
    specs = r.specifications() + [dict(columns=spline_spec["columns"])]
    data, test = r.inputs(specs)
    stage = r.DEST / f"cv{r.SPLITS}_seed{r.FOLD_SEED}"
    definitions = sorted(stage.glob("*/definition.json"))
    if any(
        json.loads(p.read_text()).get("family") == "calendar_transfer"
        for p in definitions
    ):
        from research_calendar054 import add_calendar

        data, test = add_calendar(data, test)
    y = data.target.astype(int)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    assert data.customer_id.is_unique and test.customer_id.is_unique
    kumo_dir = r.baseline_directory(r.SPLITS, r.FOLD_SEED)
    spline_dir = (
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline"
    )
    baseline = 0.75 * r.read_oof(kumo_dir, data, folds) + 0.25 * r.read_oof(
        spline_dir, data, folds
    )

    def independent_metrics(values):
        result = r.scores(y, values)
        fpr, tpr, _ = roc_curve(y, values, drop_intermediate=False)
        assert abs(result["ks"] - np.max(np.abs(tpr - fpr))) < 1e-12
        result["fold_ks"] = [r.scores(y.iloc[vi], values[vi])["ks"] for _, vi in folds]
        result["mean_fold_ks"] = float(np.mean(result["fold_ks"]))
        return result

    base = dict(name="public_056393_recipe", **independent_metrics(baseline))
    starting = json.loads((r.DEST / "research_plan.json").read_text())[
        "starting_baseline"
    ]
    assert abs(base["ks"] - starting["pooled_ks"]) < 1e-12
    assert abs(base["mean_fold_ks"] - starting["mean_fold_ks"]) < 1e-12
    rows = [base]
    proofs = []
    incomplete = []
    completed_fold_counts = {}
    for definition in definitions:
        directory = definition.parent
        recipe = json.loads(definition.read_text())
        completed = [
            i
            for i in range(r.SPLITS)
            if (directory / f"fold{i}_metadata.json").exists()
        ]
        completed_fold_counts[directory.name] = len(completed)
        if len(completed) < r.SPLITS:
            incomplete.append(directory.name)
        if not completed:
            continue
        spec = recipe.get("spec", recipe)
        columns = spec["columns"]
        assert not set(columns) & r.EXCLUDED and len(columns) == len(set(columns))
        assert recipe["training_sha256"] == r.fingerprint(
            data, ["application_id", "target"] + columns
        )
        assert recipe["test_sha256"] == r.fingerprint(
            test, ["application_id"] + columns
        )
        for fold, (ti, vi) in enumerate(folds):
            if fold not in completed:
                continue
            meta = json.loads((directory / f"fold{fold}_metadata.json").read_text())
            correction_proof = None
            calendar_proof = None
            relational_proof = None
            blend_proof = None
            assert meta["recipe"] == recipe
            assert meta["training_ids_sha256"] == r.ids_hash(data, ti)
            assert meta["validation_ids_sha256"] == r.ids_hash(data, vi)
            assert not set(data.customer_id.iloc[ti]) & set(data.customer_id.iloc[vi])
            if spec.get("family") == "tabpfn35":
                ci = r.context_indices(ti, y, spec["context_limit"], 2026 + fold)
                assert meta["context_ids_sha256"] == r.ids_hash(data, ci)
                assert len(ci) == meta["context_rows"] and not len(
                    np.intersect1d(ci, vi)
                )
            elif spec.get("family") == "offset_kumo":
                from research_context_correction054 import audit_correction

                correction_proof = audit_correction(
                    data, test, fold, ti, vi, directory, meta
                )
            elif spec.get("family") == "calendar_transfer":
                from research_calendar054 import audit_calendar

                calendar_proof = audit_calendar(
                    data, test, fold, ti, vi, directory, meta
                )
            elif spec.get("family") == "kumo_relational":
                from research_relational054 import audit_relational

                relational_proof = audit_relational(data, test, fold, ti, vi, meta)
            elif spec.get("family") == "fixed_ensemble":
                from blend_relational054 import audit_blend

                blend_proof = audit_blend(data, test, fold, ti, vi, directory, meta)
            oof_file = directory / f"fold{fold}_oof.parquet"
            assert r.checksum(oof_file) == meta["oof_sha256"]
            oof = pd.read_parquet(oof_file)
            assert oof.application_id.tolist() == data.application_id.iloc[vi].tolist()
            assert np.array_equal(oof.target.astype(int), y.iloc[vi])
            independent = r.scores(oof.target, oof.prediction)
            assert abs(independent["ks"] - meta["metrics"]["ks"]) < 1e-12
            test_file = directory / f"fold{fold}_test.parquet"
            if test_file.exists():
                assert r.checksum(test_file) == meta["test_sha256"]
                pt = pd.read_parquet(test_file)
                assert pt.application_id.equals(test.application_id)
                assert (
                    len(pt) == 30000
                    and np.isfinite(pt.prediction).all()
                    and pt.prediction.between(0, 1).all()
                )
            proofs.append(
                dict(
                    candidate=directory.name,
                    fold=fold,
                    ks=independent["ks"],
                    contexts_checked=spec.get("family")
                    in {"tabpfn35", "offset_kumo", "kumo_relational"},
                    prediction_hash_verified=True,
                    correction_proof=correction_proof,
                    calendar_proof=calendar_proof,
                    relational_proof=relational_proof,
                    blend_proof=blend_proof,
                )
            )
        if len(completed) < r.SPLITS:
            continue
        p = r.read_oof(directory, data, folds)
        weights = (
            [1.0]
            if spec.get("family")
            in {"offset_kumo", "calendar_transfer", "fixed_ensemble"}
            else [1.0, 0.25, 0.5, 0.75]
        )
        for weight in weights:
            row = dict(
                candidate=directory.name,
                new_model_weight=weight,
                **independent_metrics((1 - weight) * baseline + weight * p),
            )
            row["name"] = directory.name + f"_w{weight}"
            row.update(
                delta_ks=row["ks"] - base["ks"],
                delta_mean_fold_ks=row["mean_fold_ks"] - base["mean_fold_ks"],
                fold_wins=sum(a > b for a, b in zip(row["fold_ks"], base["fold_ks"])),
                target_reached=row["ks"] >= 0.54,
            )
            rows.append(row)
    report = dict(
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(splits=r.SPLITS, seed=r.FOLD_SEED, rows=len(data)),
        comparisons=rows,
        incomplete=incomplete,
        completed_fold_counts=completed_fold_counts,
        fold_proofs=proofs,
        target_reached=any(row.get("target_reached", False) for row in rows),
        public_056393_checkpoint_preserved=True,
        caveat="Model and feature selection reuse training labels and folds. This is not an untouched holdout; changed submissions have unknown public score.",
    )
    r.write_json(r.DEST / "independent_verification.json", report)
    return report


def export(report, candidate, weight):
    row = next(
        p
        for p in report["comparisons"]
        if p.get("candidate") == candidate and p.get("new_model_weight") == weight
    )
    baseline = report["comparisons"][0]
    assert row["ks"] > baseline["ks"], "Export requires a positive pooled KS delta."
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    sample = pd.read_csv(r.ROOT / "data/sample_submission.csv")
    assert len(sample) == 30000 and sample.application_id.is_unique
    stage = r.DEST / f"cv{r.SPLITS}_seed{r.FOLD_SEED}"
    components = [(stage / candidate, weight)]
    if weight < 1:
        components.extend(
            [
                (r.baseline_directory(r.SPLITS, r.FOLD_SEED), 0.75 * (1 - weight)),
                (
                    r.ROOT
                    / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
                    0.25 * (1 - weight),
                ),
            ]
        )
    expected = np.zeros(len(sample), dtype=float)
    sources = []
    for directory, component_weight in components:
        for fold in range(r.SPLITS):
            source = directory / f"fold{fold}_test.parquet"
            part = pd.read_parquet(source)
            assert part.application_id.is_unique and len(part) == len(sample)
            ordered = sample[["application_id"]].merge(
                part, on="application_id", validate="one_to_one"
            )
            assert ordered.application_id.equals(sample.application_id)
            p = ordered.prediction.to_numpy(dtype=float)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            expected += component_weight * p / r.SPLITS
            sources.append(
                dict(
                    path=str(source.relative_to(r.ROOT)),
                    sha256=r.checksum(source),
                    weight=component_weight / r.SPLITS,
                )
            )
    assert abs(sum(s["weight"] for s in sources) - 1) < 1e-12
    name = f"submission_goal054_{candidate}_w{weight:g}.csv"
    file = r.ROOT / name
    output = sample[["application_id"]].copy()
    output["default_probability"] = expected
    payload = output.to_csv(index=False).encode()
    if file.exists():
        assert (
            file.read_bytes() == payload
        ), "Never overwrite a changed candidate with the same name."
    else:
        file.write_bytes(payload)
    reread = pd.read_csv(file)
    assert reread.application_id.equals(sample.application_id)
    assert np.max(np.abs(reread.default_probability - expected)) < 2e-15
    metadata = dict(
        file=name,
        sha256=r.checksum(file),
        rows=len(output),
        validation=row,
        target_reached=row["ks"] >= 0.54,
        public_score=None,
        sources=sources,
        preserved_public_checkpoint=r.PUBLIC_CHECKPOINT,
        preserved_public_sha256=r.PUBLIC_SHA256,
    )
    r.write_json(r.DEST / "submission_metadata.json", metadata)
    assert r.checksum(r.ROOT / r.PUBLIC_CHECKPOINT) == r.PUBLIC_SHA256
    return metadata


def report_markdown(report):
    rows = report["comparisons"]
    lines = [
        "# Riset KS OOF 0,54",
        "",
        (
            "**Target tercapai pada validasi utama. Audit/notebook/submission harus lengkap sebelum goal selesai.**"
            if report["target_reached"]
            else "**Target belum tercapai pada eksperimen lima fold yang selesai.**"
        ),
        "",
        "Lima fold stratified seed 31027, 70.000 OOF. Public baseline 0,56393 menurut pengguna; public semua kandidat baru belum diketahui.",
        "",
        "| Kandidat | Bobot baru | KS pooled | Rata-rata KS fold | Δ pooled | Fold membaik |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('new_model_weight',0):g} | {row['ks']:.6f} | {row['mean_fold_ks']:.6f} | {row.get('delta_ks',0):+.6f} | {row.get('fold_wins',0)}/5 |"
        )
    lines += [
        "",
        "Model yang belum menyelesaikan lima fold tidak diberi skor pooled: "
        + ", ".join(report["incomplete"]),
        "",
        report["caveat"],
        "",
        "Audit menghitung KS dengan SciPy dan sklearn, memeriksa semua ID outer-training/validation, konteks pretrained, fingerprint fitur dan checksum prediksi.",
        "",
    ]
    (r.DEST / "laporan_riset.md").write_text("\n".join(lines))
    print(r.DEST / "laporan_riset.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-candidate")
    parser.add_argument(
        "--weight", type=float, choices=[0.25, 0.5, 0.75, 1.0], default=0.5
    )
    args = parser.parse_args()
    report = audit()
    report_markdown(report)
    if args.export_candidate:
        print(
            json.dumps(export(report, args.export_candidate, args.weight)), flush=True
        )
    print(
        "AUDITED",
        len(report["fold_proofs"]),
        "folds; target reached:",
        report["target_reached"],
        flush=True,
    )


if __name__ == "__main__":
    main()
