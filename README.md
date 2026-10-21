# DSC LOGIKA UI 2026 — Credit Risk

Satu repository untuk fitur, model, validasi, notebook, dan riwayat riset dari dua sumber proyek DSC LOGIKA UI 2026. Targetnya risiko DPD ≥ 90 hari dalam 12 bulan; metrik utama **KS**.

## Status terakhir — 21 Oktober 2026

- **Riset dipause atas permintaan user.** Tidak ada eksperimen atau job GPU yang berjalan.
- Public terbaik repository: **0,56405**, resep **v31**.
- Submission terakhir: **v59**, resep ensemble v41 yang diaudit ulang; public **0,56296**, ID **57486639**,21 Okt, 21.48 WIB. Bukti lokal kenaikannya lemah: OOF 0,526220 vs v31 0,526126; bootstrap P>0=0,365. **Belum ada breakthrough baru.**
- Validasi Kumo fixed-union v56 terputus pada **11/15 fit** setelah hang/reboot. Jangan menganggapnya lengkap atau otomatis menjalankannya kembali.
- Notebook saat ini mereproduksi v59. Seluruh5 sel kode mode replay sudah dijalankan lokal; CSV identik byte demi byte. Seluruh training GPU tidak diulang pada sesi terakhir.
- Kaggle memakai9.000 public / 21.000 private; rubrik lomba memberi bobot45% public dan 45% private. Deadline notebook 30 Oktober 2026, 23.59 WIB.

Mulai dari [status dan checkpoint](handoff/00_mulai_di_sini/status_sekarang.md), [riwayat eksperimen](docs/integration/indeks_eksperimen.md), dan [hasil submission terakhir](artifacts/history_utama/results/v59/v41_lowvar_today/post_submission_analysis.json).

## Struktur

| Lokasi | Isi |
|---|---|
| `eda/` | Fitur utama, validation_v2, runner model, analisis dan pemeriksaan |
| `scripts/` | Pipeline fitur serta eksperimen asal repository teman yang telah diadaptasi |
| `notebook/` | Notebook submission, builder, dan notebook historis kedua sumber |
| `research/` | Referensi metode, resep terkunci, dan snapshot kode historis |
| `artifacts/` | Laporan, konfigurasi, ringkasan metrik dan bukti submission yang kecil |
| `docs/` | Konteks kompetisi, panduan data, integrasi dan catatan cleanup |
| `handoff/` | Status, aturan, hasil, eksperimen null, dan batasan validasi |
| `environments/` | Snapshot runtime utama dan dependensi historis yang tidak kompatibel |
| `tools/` | Pemeriksaan repository sebelum commit/push |
| `third_party/` | Lisensi dan NOTICE upstream yang dipertahankan |
| `data/`, `data_pq/`, `results/` | Data serta artefak **lokal**, tidak dipublikasikan ke GitHub |

Repo teman sudah terintegrasi: **508 file dipetakan dan diverifikasi**; folder `Logika-main/` tidak lagi diperlukan. Lihat [perbandingan](docs/integration/perbandingan_repository.md) dan [pemetaan asal file](docs/integration/pemetaan_file.md). Implementasi yang berbeda dipertahankan dengan asal yang jelas; nomor versi kedua sumber tidak disamakan.

## Environment dan data

Workspace lokal memakai `.venv` yang sudah tersedia; cleanup tidak mengganti atau menghapusnya. Untuk clone baru, snapshot paket berada di [requirements utama](environments/utama/requirements-runtime.txt). Ini catatan runtime yang diuji, bukan jaminan kompatibilitas seluruh platform. Dependensi historis sumber teman dijelaskan di [environments/README.md](environments/README.md).

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps causilo==1.0.3
```

Sediakan delapan tabel CSV kompetisi dan `sample_submission.csv` di `data/`; nama tabel terdapat di [panduan data](docs/data_guide.md). Dataset, model, OOF, prediksi test dan kredensial tidak disertakan. Runtime GPU/CUDA harus sesuai mesin. Jangan menginstal ulang environment lokal hanya untuk melakukan cleanup atau pemeriksaan Git.

## Entry point

Pemeriksaan kode dan file yang akan masuk GitHub, tanpa training atau data:

```bash
python3 tools/check_repository.py
```

Membangun notebook dari kode repository, tanpa menjalankan model:

```bash
.venv/bin/python notebook/build_final.py
```

Notebook hasil build: [notebook/final.ipynb](notebook/final.ipynb). Mode `replay` memerlukan 12 file prediksi lokal yang tercatat pada audit; **clone baru tanpa artefak tersebut tidak dapat menjalankan replay**. Mode `fresh` menyediakan seluruh langkah fitur dan training secara berurutan, membutuhkan data serta bobot pretrained publik. Mode itu belum dijalankan ulang penuh pada sesi terakhir. Builder v34 sebelumnya tersimpan byte-identik dalam [arsip produksi](notebook/archive/production_v34_21okt/).

Runner validasi utama tetap `eda/valid2.py`, `eda/run_valid2.py`, dan `eda/blend2.py`. Rebuild fitur alternatif dengan reference per fitting partition menggunakan `eda/166_teammate_nested_features.py`. Semua perintah training hanya dijalankan ketika user meminta melanjutkan riset, satu job berat pada satu waktu.

## Kebijakan riset

Data panitia saja; pretrained publik diperbolehkan. Pseudo-labeling/semi-supervised telah diizinkan kembali, tetapi teacher, seleksi, threshold dan stopping wajib bersarang; outer fold hanya evaluasi. Tidak ada future leakage, LB probing, push kernel, atau git push otomatis.

Status **breakthrough** membutuhkan gain lokal > 0,003, 5 fold × minimal 3 seed, mayoritas fold dan paired bootstrap P(Δ>0) ≥ 0,9 terhadap pembanding yang relevan. Permintaan submission incremental terakhir tidak mengubah arti bukti breakthrough. Hasil public kecil tidak menjadi alasan menyetel ulang bobot.

## File lokal dan GitHub

Kode, laporan ringkas, konfigurasi, snapshot resep dan lisensi dipublikasikan. Prediksi, OOF, dataset, model, log runtime, cache, dan contoh diagnostik per baris disimpan lokal serta diabaikan Git. Snapshot implementasi lama yang sebelumnya berada di `results/` tersedia dalam [research/snapshots/](research/snapshots/); ringkasan historisnya berada di `artifacts/history_utama/`.

Lihat [persiapan GitHub dan batasan history](docs/github.md) serta [catatan penghapusan](docs/integration/cleanup.md). Cleanup ini **tidak melakukan push atau menulis ulang history Git**.
