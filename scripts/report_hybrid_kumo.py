"""Audit spline checkpoints and independently verify the Kumo/spline submission."""

from __future__ import annotations

import hashlib
import json
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

from hybrid_kumo import (
    DEST,
    ROOT,
    WEIGHT,
    baseline_directory,
    read_data,
    ids_hash,
    predict_spline,
    fingerprint,
)


def verify():
    assert (
        hashlib.sha256((ROOT / "results/teman/submission_kumo.csv").read_bytes()).hexdigest()
        == "85ac74d1b4325eba4830fd1d31bb1c21b9b9a3a6fb180d48c0c9d4b921d59d40"
    )
    data, test, spec, kumo_columns = read_data()
    y = data.target.astype(int)
    records = []
    for splits, seed in [(3, 8247), (5, 31027)]:
        stage = f"cv{splits}_seed{seed}"
        directory = DEST / stage / "reverse_spline"
        if not all(
            (directory / f"fold{i}_metadata.json").exists() for i in range(splits)
        ):
            continue
        baseline = baseline_directory(splits, seed)
        baseline_meta = (
            baseline / "fold0_oof.json"
            if (baseline / "fold0_oof.json").exists()
            else baseline / "fold0_metadata.json"
        )
        recipe = json.loads(baseline_meta.read_text())["recipe"]
        expected = recipe.get("training_sha256", recipe.get("training_data_sha256"))
        assert expected == fingerprint(
            data, ["application_id", "target"] + kumo_columns
        ), "Frozen baseline feature fingerprint mismatch."
        if "test_sha256" in recipe:
            assert recipe["test_sha256"] == fingerprint(
                test, ["application_id"] + kumo_columns
            )
        folds = list(
            StratifiedKFold(splits, shuffle=True, random_state=seed).split(data, y)
        )
        for fold, (ti, vi) in enumerate(folds):
            meta = json.loads((directory / f"fold{fold}_metadata.json").read_text())
            assert meta["training_ids_sha256"] == ids_hash(data, ti) and meta[
                "validation_ids_sha256"
            ] == ids_hash(data, vi)
            assert (
                not len(np.intersect1d(ti, vi))
                and meta["preprocessing_fitted_on_training_only"]
            )
            oof_file = directory / f"fold{fold}_oof.parquet"
            test_file = directory / f"fold{fold}_test.parquet"
            model_file = directory / f"fold{fold}_model.joblib"
            for file, key in [
                (oof_file, "oof_sha256"),
                (test_file, "test_sha256"),
                (model_file, "model_sha256"),
            ]:
                assert hashlib.sha256(file.read_bytes()).hexdigest() == meta[key]
            oof = pd.read_parquet(oof_file)
            pt = pd.read_parquet(test_file)
            assert np.array_equal(
                oof.application_id, data.application_id.iloc[vi]
            ) and np.array_equal(oof.target, y.iloc[vi])
            assert pt.application_id.equals(test.application_id)
            independent = float(
                ks_2samp(
                    oof.loc[oof.target.eq(1), "prediction"],
                    oof.loc[oof.target.eq(0), "prediction"],
                ).statistic
            )
            assert abs(independent - meta["metrics"]["ks"]) < 1e-12
            bundle = joblib.load(model_file)
            pv = predict_spline(bundle, data.iloc[vi])
            pv_test = predict_spline(bundle, test)
            difference = float(
                max(
                    np.max(np.abs(pv - oof.prediction)),
                    np.max(np.abs(pv_test - pt.prediction)),
                )
            )
            assert difference < 5e-12
            records.append(
                dict(
                    stage=stage,
                    fold=fold,
                    scipy_ks=independent,
                    prediction_rebuild_max_abs_difference=difference,
                )
            )
            print(
                "VERIFIED",
                stage,
                fold,
                "max prediction difference",
                difference,
                flush=True,
            )
    candidate = None
    meta_file = DEST / "submission_metadata.json"
    if not meta_file.exists():
        meta_file = DEST / "exploratory_submission_metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        file = ROOT / meta["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"]
        sample = pd.read_csv(ROOT / "data/sample_submission.csv")
        actual = pd.read_csv(file)
        assert (
            actual.columns.tolist() == sample.columns.tolist()
            and actual.application_id.equals(sample.application_id)
        )
        assert (
            len(actual) == 30000
            and actual.application_id.is_unique
            and actual.default_probability.between(0, 1).all()
        )
        p = np.zeros(len(sample), dtype=np.float64)
        for directory, weight in [
            (baseline_directory(5, 31027), 1 - WEIGHT),
            (DEST / "cv5_seed31027/reverse_spline", WEIGHT),
        ]:
            for fold in range(5):
                source = directory / f"fold{fold}_test.parquet"
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == meta["source_prediction_sha256"][str(source.relative_to(ROOT))]
                )
                ordered = sample[["application_id"]].merge(
                    pd.read_parquet(source), on="application_id", validate="one_to_one"
                )
                p += weight * ordered.prediction.to_numpy(dtype=np.float64) / 5
        difference = float(np.max(np.abs(p - actual.default_probability)))
        assert difference < 2e-15
        candidate = dict(
            file=meta["file"],
            sha256=meta["sha256"],
            rows=len(actual),
            ensemble_max_abs_difference=difference,
            public_score=None,
            export_mode=meta.get("export_mode"),
            promotion_passed=meta.get("promotion_passed"),
        )
    result = dict(
        frozen_checkpoint_056262_unchanged=True,
        contexts_and_validation_disjoint=True,
        baseline_feature_fingerprints_match=True,
        spline_predictions_reproduced_from_saved_models=True,
        folds=records,
        submission=candidate,
    )
    (DEST / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_export():
    """Audit the newly exported CSV without repeating unchanged model inference."""
    path = DEST / "submission_metadata.json"
    if not path.exists():
        path = DEST / "exploratory_submission_metadata.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    file = ROOT / meta["file"]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"]
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    actual = pd.read_csv(file)
    assert (
        actual.columns.tolist() == sample.columns.tolist()
        and actual.application_id.equals(sample.application_id)
    )
    assert len(actual) == 30000 and actual.application_id.is_unique
    assert (
        np.isfinite(actual.default_probability).all()
        and actual.default_probability.between(0, 1).all()
    )
    expected = np.zeros(len(sample), dtype=np.float64)
    counts = {"kumo": 0, "spline": 0}
    for relative, digest in meta["source_prediction_sha256"].items():
        source = ROOT / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        component = "spline" if "reverse_spline" in source.parts else "kumo"
        counts[component] += 1
        weight = 0.25 if component == "spline" else 0.75
        ordered = sample[["application_id"]].merge(
            pd.read_parquet(source), on="application_id", validate="one_to_one"
        )
        assert ordered.application_id.equals(sample.application_id)
        expected += weight * ordered.prediction.to_numpy(dtype=np.float64) / 5
    assert counts == {"kumo": 5, "spline": 5}
    difference = float(np.max(np.abs(expected - actual.default_probability)))
    assert difference < 2e-15
    parts = []
    fold_ks = []
    base_fold = []

    def independent(y, p):
        return float(ks_2samp(p[y == 1], p[y == 0]).statistic)

    for fold in range(5):
        a = pd.read_parquet(baseline_directory(5, 31027) / f"fold{fold}_oof.parquet")
        b = pd.read_parquet(
            DEST / "cv5_seed31027/reverse_spline" / f"fold{fold}_oof.parquet"
        )
        assert a.application_id.equals(b.application_id) and np.array_equal(
            a.target, b.target
        )
        base = a.prediction.to_numpy(dtype=np.float64)
        mix = 0.75 * base + 0.25 * b.prediction.to_numpy(dtype=np.float64)
        y = a.target.to_numpy(dtype=int)
        fold_ks.append(independent(y, mix))
        base_fold.append(independent(y, base))
        parts.append(
            pd.DataFrame(
                dict(
                    application_id=a.application_id, target=y, base=base, prediction=mix
                )
            )
        )
    oof = pd.concat(parts, ignore_index=True)
    assert len(oof) == 70000 and oof.application_id.is_unique
    pooled = independent(oof.target.to_numpy(), oof.prediction.to_numpy())
    baseline_pooled = independent(oof.target.to_numpy(), oof.base.to_numpy())
    row = meta["validation"]
    assert (
        abs(pooled - row["ks"]) < 1e-12
        and abs((pooled - baseline_pooled) - row["delta_ks"]) < 1e-12
    )
    assert np.allclose(fold_ks, row["fold_ks"], rtol=0, atol=1e-12)
    assert abs(np.mean(fold_ks) - row["mean_fold_ks"]) < 1e-12
    assert (
        abs(np.mean(fold_ks) - np.mean(base_fold) - row["delta_mean_fold_ks"]) < 1e-12
    )
    assert sum(a > b for a, b in zip(fold_ks, base_fold)) == row["fold_wins"]
    assert meta["public_score"] is None
    promotion = (
        row["delta_ks"] > 0 and row["delta_mean_fold_ks"] > 0 and row["fold_wins"] >= 3
    )
    assert promotion == meta["promotion_passed"]
    result = dict(
        file=meta["file"],
        sha256=meta["sha256"],
        rows=len(actual),
        ensemble_max_abs_difference=difference,
        independent_pooled_ks=pooled,
        independent_baseline_ks=baseline_pooled,
        export_mode=meta["export_mode"],
        promotion_passed=promotion,
        public_score=None,
    )
    verification_path = DEST / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification["submission"] = result
    verification["independent_hybrid_ks_verified"] = True
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    print("CSV VERIFIED", json.dumps(result), flush=True)
    return result


def report():
    screen = json.loads((DEST / "cv3_seed8247/comparison.json").read_text())
    confirmation = json.loads((DEST / "cv5_seed31027/comparison.json").read_text())
    verdict = json.loads((DEST / "verdict.json").read_text())
    base = confirmation["comparisons"][0]
    selected = confirmation["comparisons"][-1]
    for stage in ["cv3_seed8247", "cv5_seed31027"]:
        shutil.copyfile(
            DEST / stage / "comparison.csv",
            ROOT / "artifacts/tables" / f"kumo_hybrid_{stage}.csv",
        )
    x = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, base["fold_ks"], "o-", label="Kumo baseline", color="#475569")
    axes[0].plot(
        x, selected["fold_ks"], "o-", label="75% Kumo + 25% spline", color="#0284c7"
    )
    axes[0].set(xlabel="Fold (seed 31027)", ylabel="KS validation", xticks=x)
    axes[0].legend(fontsize=8)
    axes[1].bar(
        x,
        selected["fold_deltas"],
        color=["#059669" if d > 0 else "#e11d48" for d in selected["fold_deltas"]],
    )
    axes[1].axhline(0, color="#475569", linewidth=0.8)
    axes[1].set(
        xlabel="Fold yang sama", ylabel="Selisih KS terhadap baseline", xticks=x
    )
    fig.suptitle("Validasi lokal; skor public kandidat belum diketahui")
    fig.tight_layout()
    fig.savefig(ROOT / "artifacts/figures/kumo_hybrid_confirmation.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Kumo dan spline: hasil improvement",
        "",
        f"Gabungan tetap **75% Kumo Large + 25% spline fitur laten**, {'lolos' if verdict['passed'] else 'belum lolos'} pemeriksaan lima fold.",
        "",
        "| Ukuran pada lima fold identik | Kumo baseline | Gabungan | Selisih |",
        "|---|---:|---:|---:|",
    ]
    for label, key in [
        ("KS pooled OOF", "ks"),
        ("Rata-rata KS fold", "mean_fold_ks"),
        ("ROC-AUC pooled", "auc"),
    ]:
        lines.append(
            f"| {label} | {base[key]:.6f} | {selected[key]:.6f} | {selected[key]-base[key]:+.6f} |"
        )
    lines += [
        "",
        f"KS membaik pada **{selected['fold_wins']}/5 fold**.",
        "",
        "![Selisih KS per fold](../../figures/kumo_hybrid_confirmation.png)",
        "",
        "## Resep dan seleksi",
        "",
        "Kumo Large memakai 90 fitur, 8 anggota, maksimum 50.000 baris konteks, FP16, bobot pretrained v1.0.1 yang dibekukan. Spline memakai resep reverse_spline dari notebook 07: basis kubik 5 knot, C=0,05, tanpa interaksi, tanpa pembobotan kelas.",
        "Imputasi, knot, standardisasi, encoding kategori, dan koefisien spline dipelajari hanya dari outer-training setiap fold. Estimasi fitur laten tidak memakai target, tetapi memakai fitur seluruh train seperti audit sebelumnya.",
        "Prediksi OOF sebelumnya dipakai untuk screening dua keluarga model dengan bobot tambahan 25% dan 50%. Campuran spline 25% diteruskan. Spline tiga fold kemudian dilatih ulang dalam lingkungan Kumo agar pipeline memakai versi paket yang sama.",
        "Bobot 75/25 dikunci sebelum lima fold dihitung. Baseline dan kandidat dibandingkan pada baris dan seed identik. Probabilitas test dirata-ratakan dari lima model per komponen lalu dicampurkan.",
        "",
        "### Reproduksi screening tiga fold",
        "",
        "| Model | KS pooled | Rata-rata KS fold | Δ pooled | Fold membaik |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in screen["comparisons"]:
        lines.append(
            f"| {row['name']} | {row['ks']:.6f} | {row['mean_fold_ks']:.6f} | {row['delta_ks']:+.6f} | {row['fold_wins']}/3 |"
        )
    lines += [
        "",
        "## Batas interpretasi",
        "",
        "Label training dan lima split ini sudah pernah dipakai untuk eksperimen lain. Hasil merupakan pemeriksaan konsistensi dengan pemilihan model berulang, bukan holdout yang belum pernah disentuh. Kenaikan kecil belum membuktikan signifikansi statistik dan tidak menjamin kenaikan Kaggle.",
        "Skor tiga dan lima fold tidak dibandingkan langsung. Checkpoint public yang sudah dikonfirmasi tetap submission_kumo.csv dengan skor **0,56262**. Public kandidat baru **belum diketahui**.",
        "",
        "## Reproduksi dan audit",
        "",
        "Gunakan Python 3.11 dan environments/teman/requirements-kumo.txt. Notebook [10_kumo_spline.ipynb](../../../notebook/10_kumo_spline.ipynb) memuat EDA, seluruh agregasi, kode Kumo/spline, validasi dan ekspor.",
        "```bash",
        "python scripts/hybrid_kumo.py",
        "python scripts/report_hybrid_kumo.py",
        "python scripts/build_hybrid_kumo_notebook.py --execute",
        "```",
        "",
        "Jalur CLI memerlukan fitur V5 dan reverse. Notebook membangun fitur dari data mentah jika cache belum tersedia. Bobot Kumo dan data mentah tetap lokal; tidak ada upload otomatis ke Kaggle.",
        "Audit menghitung ulang KS dengan SciPy, memeriksa hash input baseline dan batas fold, serta menghitung ulang prediksi OOF/test spline dari model tersimpan. CSV diperiksa terhadap sample submission dan sumber ensemble.",
    ]
    metadata_path = DEST / "submission_metadata.json"
    if not metadata_path.exists():
        metadata_path = DEST / "exploratory_submission_metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        lines += [
            "",
            "## Submission kandidat",
            "",
            f"[`{meta['file']}`](../../../{meta['file']}) — 30.000 probabilitas; public score belum diketahui.",
            "",
            f"SHA-256: `{meta['sha256']}`.",
        ]
        if meta.get("export_mode") == "exploratory":
            lines += [
                "",
                "File ini **eksploratif**: pooled KS naik, tetapi rata-rata KS turun dan hanya 1/5 fold membaik. File belum memenuhi aturan promosi dan tidak mengganti checkpoint public 0,56262.",
            ]
    (DEST / "laporan_hybrid.md").write_text("\n".join(lines) + "\n")
    print(DEST / "laporan_hybrid.md", flush=True)


def main():
    verify()
    verify_export()
    report()


if __name__ == "__main__":
    main()
