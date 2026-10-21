"""Export a confirmed version-5 candidate with ID, probability, and KS checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from optimize_ks import ROOT, OUT, metrics, ks

DESTINATION = OUT / "v5"


def write_figure(y, baseline, new):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, p, color in [
        ("Versi 4", baseline, "#64748b"),
        ("Versi 5", new, "#0284c7"),
    ]:
        fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
        fraction = y.mean() * tpr + (1 - y.mean()) * fpr
        axes[0].plot(
            fraction, tpr - fpr, color=color, label=f"{name}: KS {ks(y,p):.5f}"
        )
        axes[1].plot(
            fpr, tpr, color=color, label=f'{name}: AUC {metrics(y,p)["roc_auc"]:.5f}'
        )
    axes[0].set(
        xlabel="Proporsi aplikasi dengan skor tertinggi",
        ylabel="TPR - FPR",
        title="Kurva KS",
    )
    axes[1].set(
        xlabel="False positive rate", ylabel="True positive rate", title="Kurva ROC"
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.15)
    fig.suptitle("70.000 prediksi OOF pada sepuluh fold yang sama")
    fig.tight_layout()
    fig.savefig(ROOT / "artifacts/figures/ks_v4_v5.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="v5_tabm_weight1")
    args = parser.parse_args()
    confirmation = json.loads(
        (DESTINATION / f"confirmation_{args.candidate}.json").read_text()
    )
    assert confirmation["version5"]["ks"] > confirmation["baseline"]["ks"]
    assert (
        confirmation["version5"]["mean_fold_ks"]
        > confirmation["baseline"]["mean_fold_ks"]
    )
    old_file = ROOT / "results/teman/submission_v4.csv"
    old_digest = hashlib.sha256(old_file.read_bytes()).hexdigest()
    assert (
        old_digest == "890e05a3f0b09500142395d5498e939928328b2770c509ca035f760a8dea8344"
    )
    oof = pd.read_parquet(DESTINATION / f"final_oof_{args.candidate}.parquet")
    predicted = pd.read_parquet(DESTINATION / f"final_test_{args.candidate}.parquet")
    recipe = confirmation["recipe"]["recipe"]
    for frame in [oof, predicted]:
        assert frame.application_id.is_unique
        assert np.isfinite(frame[list(recipe)]).all().all()
        assert (
            frame[list(recipe)].ge(0).all().all()
            and frame[list(recipe)].le(1).all().all()
        )
        assert np.array_equal(
            frame.version5.to_numpy(),
            sum(w * frame[n] for n, w in recipe.items()).to_numpy(),
        )
    y = oof.target.to_numpy(dtype=int)
    assert (
        abs(
            ks_2samp(oof.version5[y == 1], oof.version5[y == 0]).statistic
            - confirmation["version5"]["ks"]
        )
        < 1e-12
    )
    sample = pd.read_csv(ROOT / "data/sample_submission.csv")
    assert set(sample.application_id) == set(predicted.application_id)
    submission = sample[["application_id"]].merge(
        predicted[["application_id", "version5"]],
        on="application_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    submission.rename(columns={"version5": "default_probability"}, inplace=True)
    assert (
        submission.columns.tolist() == sample.columns.tolist()
        and submission.shape == (30000, 2)
    )
    assert submission.application_id.equals(sample.application_id)
    assert (
        np.isfinite(submission.default_probability).all()
        and submission.default_probability.between(0, 1).all()
    )
    old_prediction = sample[["application_id"]].merge(
        predicted[["application_id", "version4"]],
        on="application_id",
        sort=False,
        validate="one_to_one",
    )
    assert (
        np.max(
            np.abs(old_prediction.version4 - pd.read_csv(old_file).default_probability)
        )
        < 1e-11
    )
    # Conditional paired bootstrap: fixed predictions, not a correction for model selection.
    rng = np.random.default_rng(20261013)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    gains = []
    for _ in range(200):
        rows = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        gains.append(
            ks(y[rows], oof.version5.iloc[rows]) - ks(y[rows], oof.version4.iloc[rows])
        )
    bootstrap = {
        "replicates": 200,
        "mean_gain": float(np.mean(gains)),
        "p05": float(np.quantile(gains, 0.05)),
        "p95": float(np.quantile(gains, 0.95)),
        "fraction_positive": float(np.mean(np.array(gains) > 0)),
        "caveat": "Fixed OOF predictions; conditional on selected features/models/weights. Does not remove selection optimism.",
    }
    file = ROOT / "results/teman/submission_v5.csv"
    submission.to_csv(file, index=False, float_format="%.12f")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    public_result_file = DESTINATION / "public_leaderboard_result.json"
    public_result = (
        json.loads(public_result_file.read_text())
        if public_result_file.exists()
        else None
    )
    # A reported score belongs to this prediction file, not to a newly trained variant.
    if public_result and (
        public_result.get("file") != file.name or public_result.get("sha256") != digest
    ):
        public_result = None
    public_score = public_result["public_score"] if public_result else None
    reread = pd.read_csv(file)
    assert (
        reread.application_id.equals(sample.application_id)
        and reread.default_probability.between(0, 1).all()
    )
    assert (
        np.max(np.abs(reread.default_probability - submission.default_probability))
        < 5.1e-13
    )
    if confirmation["promote"]:
        shutil.copy2(file, ROOT / "results/teman/submission.csv")
    metadata = {
        "file": file.name,
        "sha256": digest,
        "baseline_file": old_file.name,
        "baseline_sha256": old_digest,
        "training_rows": 70000,
        "test_rows": 30000,
        "recipe": recipe,
        "selection_3fold": {
            "baseline_ks": 0.524109452121715,
            "new_ks": 0.5252849865414044,
            "baseline_mean_fold_ks": 0.5274039087623731,
            "new_mean_fold_ks": 0.5275239228609559,
            "fold_wins": 1,
            "caveat": "Small exploratory improvement; only one fold improved.",
        },
        "confirmation_10fold": confirmation,
        "paired_bootstrap": bootstrap,
        "best_recorded_public_score": 0.55845,
        "new_public_score": public_score,
        "public_leaderboard_result": public_result,
        "promoted_to_submission_csv": confirmation["promote"],
        "probability_summary": reread.default_probability.describe().to_dict(),
        "checks": {
            "header": True,
            "sample_order": True,
            "unique_ids": True,
            "finite_probability": True,
            "probability_bounds": True,
            "baseline_reproduction": True,
            "independent_ks": True,
        },
        "caveat": "Features, model variants and weights reuse training labels. Validation contains selection optimism; leaderboard observations are recorded separately with their source.",
    }
    # Read the exact selection metrics rather than relying on rounded report text.
    selection = pd.read_parquet(DESTINATION / "selection_oof_tabm.parquet")
    w = recipe[args.candidate]
    selection_new = (1 - w) * selection.version4 + w * selection[args.candidate]
    folds = list(
        StratifiedKFold(3, shuffle=True, random_state=8247).split(
            selection, selection.target
        )
    )
    metadata["selection_3fold"]["new_ks"] = ks(selection.target, selection_new)
    metadata["selection_3fold"]["new_mean_fold_ks"] = float(
        np.mean(
            [ks(selection.target.iloc[vi], selection_new.iloc[vi]) for _, vi in folds]
        )
    )
    (DESTINATION / "submission_v5_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    (DESTINATION / "locked_recipe.json").write_text(
        json.dumps(confirmation["recipe"], indent=2)
    )
    if confirmation["promote"]:
        (ROOT / "artifacts/submission_metadata.json").write_text(
            json.dumps(dict(metadata, file="submission.csv"), indent=2)
        )
    oof.to_parquet(DESTINATION / "combined_final_oof.parquet", index=False)
    predicted.to_parquet(DESTINATION / "combined_final_test.parquet", index=False)
    write_figure(y, oof.version4, oof.version5)
    old, new = confirmation["baseline"], confirmation["version5"]
    public_note = (
        f"Skor public versi 5 **{public_score:.5f}**, dilaporkan pengguna pada "
        f"{public_result['reported_on']}; selisih terhadap versi 3 (0,55845) adalah "
        f"**{public_score - .55845:+.5f}**. Hasil ini belum diverifikasi ulang melalui API Kaggle. "
        "Peningkatan OOF belum diikuti peningkatan public leaderboard."
        if public_result
        else "Skor public versi 5 **belum tersedia**."
    )
    report = f"""# Kandidat submission versi 5

Versi 5 menggabungkan 75% versi 4 dan 25% TabM. TabM dilatih dari label kompetisi; tidak memakai model pretrained atau data eksternal. Harga per unit dari basket completed dan cancelled dipakai sebagai sinyal pilihan produk. Nilai cancelled tidak dihitung sebagai belanja. Waktu seluruh transaksi diperiksa sebelum keputusan.

| Validasi pada baris dan fold yang sama | Versi 4 | Versi 5 |
|---|---:|---:|
| KS OOF, tiga fold | {metadata['selection_3fold']['baseline_ks']:.6f} | {metadata['selection_3fold']['new_ks']:.6f} |
| Rata-rata KS, tiga fold | {metadata['selection_3fold']['baseline_mean_fold_ks']:.6f} | {metadata['selection_3fold']['new_mean_fold_ks']:.6f} |
| KS OOF, sepuluh fold | {old['ks']:.6f} | {new['ks']:.6f} |
| Rata-rata KS, sepuluh fold | {old['mean_fold_ks']:.6f} | {new['mean_fold_ks']:.6f} |
| ROC-AUC, sepuluh fold | {old['roc_auc']:.6f} | {new['roc_auc']:.6f} |
| Log loss, sepuluh fold | {old['log_loss']:.6f} | {new['log_loss']:.6f} |

KS meningkat pada **{confirmation['fold_wins']} dari 10 fold**. Pemilihan kandidat tiga fold hanya meningkatkan satu fold dan tidak memenuhi aturan seleksi ketat pada `select_v5.py`; kandidat ini diteruskan sebagai konfirmasi eksploratif karena KS gabungan dan rata-ratanya meningkat sedikit. Bobot 25% dikunci sebelum pelatihan sepuluh fold. Aturan promosi akhir: KS gabungan dan rata-rata fold harus meningkat, dengan minimal enam fold membaik. Kandidat ini meningkatkan kedua statistik agregat, tetapi **belum memenuhi syarat enam fold**. Tidak ada perubahan bobot dari hasil sepuluh fold. File kandidat tetap disediakan untuk review dan pengujian public leaderboard.

![Kurva KS dan ROC](../../figures/ks_v4_v5.png)

Early stopping TabM memakai split stratified 12,5% **di dalam training setiap outer fold**, berdasarkan weighted log loss. Setelah jumlah epoch dipilih, encoder dan model dilatih ulang pada seluruh training outer fold. Median, skala, kategori, dan bin embedding dipelajari pada training terkait. Label outer validation tidak dipakai untuk early stopping. TabM memiliki 16 anggota yang dilatih dengan loss masing-masing, lalu probabilitasnya dirata-ratakan. Transformasi bobot kelas dilakukan per anggota jika bobot positif lebih dari satu.

Eksperimen yang belum dipromosikan meliputi spline dengan fitur tambahan/interaksi/regularisasi berbeda, Explainable Boosting Machine, XGBoost dangkal, CatBoost dangkal, specialist menurut ketersediaan riwayat pembayaran, dan TabM dengan fitur inti. Definisi dan ringkasan seluruh kelompok tersedia dalam folder ini; `blend_ranking.json` menunjukkan hasil pencarian. Pencarian menggunakan label train yang sama sehingga hasil lokal mengandung optimisme seleksi. Bootstrap pada metadata hanya memakai prediksi OOF tetap dan tidak mengoreksi proses seleksi. Tidak ada label test yang digunakan.

{public_note} Skor public terbaik yang tercatat tetap versi 3: **0,55845**. File dibuat untuk diunggah oleh pengguna; tidak ada submit otomatis. `submission_v5.csv` berisi kandidat versi 5; `submission.csv` tetap versi 4 karena syarat promosi belum terpenuhi. Versi 4 dan checkpoint versi 3 tetap tersedia.

SHA-256: `{metadata['sha256']}`.

```bash
python -m pip install -r environments/teman/requirements-v5.txt
python scripts/build_features_v5.py
python scripts/evaluate_v5.py --group tabm
python scripts/train_v5_candidate.py --candidate v5_tabm_weight1 --new-weight .25
python scripts/export_v5_submission.py
```

Fitur/model versi 4 harus tersedia terlebih dahulu, atau jalankan notebook 06 yang membangun seluruh komponennya. Lingkungan referensi memakai Python 3.10, PyTorch 1.13.1+cu116, dan GPU A100. CPU didukung; durasi dan pembulatan dapat berbeda pada lingkungan lain.
"""
    (DESTINATION / "laporan_submission_v5.md").write_text(report)
    print(
        json.dumps(
            {
                "file": str(file),
                "baseline_ks": old["ks"],
                "new_ks": new["ks"],
                "fold_wins": confirmation["fold_wins"],
                "sha256": metadata["sha256"],
                "bootstrap": bootstrap,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
