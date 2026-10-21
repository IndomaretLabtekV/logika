"""Write a concise Indonesian report and matching-fold comparison plot."""

from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from improve_kumo import DEST
from kumo_tabular import ROOT


def main():
    screen = json.loads((DEST / "cv3_seed8247/comparison.json").read_text())
    confirm = json.loads((DEST / "cv5_seed31027/comparison.json").read_text())
    verdict = json.loads((DEST / "confirmation_verdict.json").read_text())
    selected = verdict["selected"]
    baseline = next(
        r for r in confirm["comparisons"] if r["name"] == "public_056262_recipe"
    )
    for stage in ["cv3_seed8247", "cv5_seed31027"]:
        shutil.copyfile(
            DEST / stage / "comparison.csv",
            ROOT / "artifacts/tables" / f"kumo_improvement_{stage}.csv",
        )
    figure = ROOT / "artifacts/figures/kumo_improvement_confirmation.png"
    x = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        x, baseline["fold_ks"], "o-", label="Resep Kumo public 0,56262", color="#475569"
    )
    axes[0].plot(
        x, selected["fold_ks"], "o-", label="Kandidat terkunci", color="#0284c7"
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
        xlabel="Fold yang sama", ylabel="Selisih KS kandidat − baseline", xticks=x
    )
    fig.suptitle("Konfirmasi lokal; public score kandidat belum diketahui")
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    lines = [
        "# Improvement Kumo: hasil eksperimen",
        "",
        f"Kandidat terkunci: `{selected['name']}`. Konfirmasi **{'lolos' if verdict['passed'] else 'belum lolos'}**.",
        "",
        "Checkpoint Kaggle yang sudah terkonfirmasi tetap `submission_kumo.csv`, public **0,56262**.",
        "Skor public kandidat baru belum diketahui. Tidak ada upload otomatis.",
        "",
        "## Perbandingan lima fold identik",
        "",
        "| Ukuran | Baseline Kumo | Kandidat | Selisih |",
        "|---|---:|---:|---:|",
    ]
    for label, key in [
        ("KS pooled OOF", "ks"),
        ("Rata-rata KS fold", "mean_fold_ks"),
        ("ROC-AUC pooled", "auc"),
    ]:
        lines.append(
            f"| {label} | {baseline[key]:.6f} | {selected[key]:.6f} | {selected[key]-baseline[key]:+.6f} |"
        )
    lines += [
        "",
        f"KS membaik pada **{selected['fold_wins']} dari 5 fold**.",
        "",
        "![Selisih per fold](../../figures/kumo_improvement_confirmation.png)",
        "",
        "Baseline memakai 90 fitur, 8 anggota ensemble, dan maksimum 50.000 baris konteks.",
        "Setiap kandidat menggunakan hanya konteks outer-training; semua prediksi validation adalah OOF.",
        "Probabilitas test dirata-ratakan dari lima konteks fold, sesuai bobot yang dikunci.",
        "",
        "## Seleksi awal tiga fold",
        "",
        "| Kandidat / campuran | KS pooled | Rata-rata KS fold | Δ pooled | Fold membaik |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in screen["comparisons"]:
        lines.append(
            f"| `{row['name']}` | {row['ks']:.6f} | {row['mean_fold_ks']:.6f} | {row['delta_ks']:+.6f} | {row['fold_wins']}/3 |"
        )
    lines += [
        "",
        "Pada seleksi awal, kandidat dipilih hanya jika pooled KS dan rata-rata KS naik serta minimal dua fold membaik.",
        "Bobot campuran kandidat terhadap baseline dibatasi 25% dan 50%. Resep awal dikunci sebelum konfirmasi lima fold.",
        "Sampling kelas seimbang merupakan tahap cadangan bersyarat jika seluruh kandidat awal gagal; rencana tersimpan sebelum sampling diuji.",
    ]
    prior_file = DEST / "confirmation_verdict_blend_failed.json"
    if prior_file.exists():
        prior = json.loads(prior_file.read_text())["selected"]
        lines += [
            "",
            "## Konfirmasi awal yang tidak lolos",
            "",
            f"Campuran 75% baseline + 25% konteks seimbang: pooled KS **{prior['ks']:.6f}** (Δ **{prior['delta_ks']:+.6f}**), rata-rata KS **{prior['mean_fold_ks']:.6f}** (Δ **{prior['delta_mean_fold_ks']:+.6f}**), membaik pada **{prior['fold_wins']}/5 fold**.",
            "Campuran tidak diekspor karena pooled KS turun. Sesudah itu, resep konteks penuh ditetapkan sebelum prediksinya dihitung, memakai lima split yang sama.",
            "Konteks penuh memakai 56.000 baris per fold, dibanding batas 50.000 pada baseline. Fitur, delapan anggota ensemble, dan seed tetap sama.",
            "Pada tiga fold awal, training hanya 46.666–46.667 baris sehingga kedua batas konteks menghasilkan input identik. Efek konteks tambahan baru diuji pada lima fold.",
            "Uji lanjutan ini merupakan tambahan pemilihan model pada lima split yang sama; hasilnya dapat optimistis.",
        ]
    lines += [
        "",
        "## Batas interpretasi dan reproduksi",
        "",
        "Pergantian seed menggunakan label training yang sama, sehingga konfirmasi ini bukan holdout yang belum pernah dipakai.",
        "Pencarian berulang mengandung optimisme seleksi. Selisih kecil belum menunjukkan signifikansi statistik atau menjamin kenaikan leaderboard.",
        "Skor tiga dan lima fold tidak dibandingkan langsung karena ukuran training dan pembagian baris berbeda.",
        "Estimasi noise tidak memakai target, tetapi parameter unsupervised memanfaatkan fitur seluruh train seperti audit sebelumnya.",
        "Pemeriksaan ulang fitur dari data mentah cocok dalam toleransi rtol=1e-6 dan atol=1e-7; seed RNG asli belum diketahui.",
        "",
        "Gunakan Python 3.11 dan `environments/teman/requirements-kumo.txt`, dengan GPU CUDA. Bobot Kumo v1.0.1 serta TabICLv2 resmi dijalankan lokal.",
        "Notebook [09_kumo_improvement.ipynb](../../../notebook/09_kumo_improvement.ipynb) memuat EDA, agregasi, semua kode model, perbandingan dan ekspor.",
        "",
        "```bash",
        "python scripts/run_kumo_improvement.py",
        "python scripts/verify_kumo_improvement.py",
        "python scripts/report_kumo_improvement.py",
        "python scripts/build_kumo_improvement_notebook.py --execute",
        "```",
        "",
        "Jalur CLI memerlukan fitur V5 dan reverse yang dibangun dari notebook sebelumnya. Pipeline otomatis menghitung kandidat yang belum tersedia dan memeriksa cache sebelum memakainya. Untuk menjalankan hanya tahap awal:",
        "",
        "```bash",
        "python scripts/improve_kumo.py --candidates core37_e8 denoised_e8 v5_latent100_e8 v5_e16 --test",
        "```",
        "",
        "Data mentah, fitur parquet, dan prediksi per baris tidak disertakan di Git. Definisi, metadata fold, metrik, serta checksum disertakan.",
        "Cache diperiksa terhadap data, resep, dan versi paket. Model pralatih tetap membekukan bobot.",
    ]
    if verdict["passed"]:
        meta = json.loads((DEST / "submission_metadata.json").read_text())
        lines += [
            "",
            "## Berkas kandidat",
            "",
            f"[`{meta['file']}`](../../../{meta['file']}) memiliki 30.000 probabilitas dalam urutan sample submission.",
            f"SHA-256: `{meta['sha256']}`.",
            "",
            "CSV lama dipertahankan. Public score baru harus dicatat untuk checksum kandidat ini setelah submit Kaggle.",
        ]
    (DEST / "laporan_improvement.md").write_text("\n".join(lines) + "\n")
    print(DEST / "laporan_improvement.md")


if __name__ == "__main__":
    main()
