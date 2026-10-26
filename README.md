# DSC LOGIKA UI 2026 — Credit Risk

Repository riset untuk memprediksi risiko **DPD ≥ 90 hari dalam 12 bulan**, dengan metrik **Kolmogorov–Smirnov (KS)**. Tiga sumber pipeline tim dikumpulkan di sini: pipeline utama, pipeline teman (goal054), dan DSC0152.

## Mulai di sini

- [Panduan data](docs/data_guide.md) dan [konteks kompetisi](docs/context.md).
- [Peta kode utama](eda/README.md), [pipeline teman dan DSC0152](scripts/README.md), serta [panduan notebook](notebook/README.md).
- [Indeks handoff](handoff/README.md) dan [jurnal merge 25 Oktober](handoff/04_riwayat_versi/jurnal_25okt_merge.md).
- [Persiapan commit](docs/github.md) untuk membedakan file publik dan artefak lokal.

## Hasil yang tercatat

Menurut catatan 25 Oktober 2026, public terbaik tim adalah **0,56470** dari goal054 Kumo + offset. Resep merge yang dipromosikan, **merge_R2DK**, memperoleh pooled OOF KS **0,52886** pada lima fold seed 42. Resep ini menggabungkan R2 (pipeline teman dua seed), D (DSC0152), dan K64 (Kumo utama), dengan bobot rank sama.

Merge belum disubmit pada catatan tersebut. OOF dan public berasal dari evaluasi berbeda; angka OOF bukan klaim kenaikan leaderboard. Detail, eksperimen null, dan batas validasinya ada di [jurnal merge](handoff/04_riwayat_versi/jurnal_25okt_merge.md). Catatan bertanggal lebih lama dipertahankan sebagai riwayat, bukan status proses saat ini.

## Struktur repository

| Folder | Isi |
|---|---|
| `eda/` | Fitur, validation_v2, runner, audit, dan eksperimen pipeline utama |
| `scripts/` | Pipeline teman, DSC0152, dan runner pembangun famili merge |
| `notebook/` | Notebook final, builder, notebook per versi dan sumber |
| `artifacts/` | Protokol terkunci, konfigurasi, laporan, metrik dan gambar ringkas |
| `research/` | Referensi metode, resep terkunci dan snapshot implementasi historis |
| `docs/` | Aturan, panduan data, integrasi, dan catatan cleanup |
| `handoff/` | Indeks konteks riset, jurnal, hasil dan serah terima bertanggal |
| `environments/` | Snapshot dependensi utama dan tiap sumber pipeline |
| `tools/` | Pemeriksaan publikasi dan pengujiannya, tanpa dependensi ML |
| `third_party/` | Lisensi dan NOTICE upstream |

`data/`, `data_pq/`, `results/`, model, environment virtual, log runtime, dan ZIP hanya tersedia lokal. Laporan agregat yang sudah tercatat di `eda/**/out/` serta `outputs/04_risk_exploration/` dipertahankan sebagai bukti historis; output baru di folder tersebut diabaikan Git sampai dipilih secara eksplisit.

## Environment dan data

Untuk mesin baru, lihat [panduan environment](environments/README.md). Contoh environment pipeline utama:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps causilo==1.0.3
```

Snapshot lengkap ada di [requirements runtime](environments/utama/requirements-runtime.txt). Dependensi beberapa pipeline lama berbeda; gunakan environment terpisah sesuai sumbernya. GPU/CUDA harus sesuai mesin penerima.

Sediakan delapan tabel CSV kompetisi dan `sample_submission.csv` di `data/`, sesuai [panduan data](docs/data_guide.md). Bobot pretrained publik diunduh terpisah bila diperlukan. Clone GitHub tidak membawa dataset, model, OOF, atau prediksi test.

## Entry point

Periksa file yang akan masuk commit, tanpa training, GPU, atau jaringan:

```bash
python3 tools/check_repository.py
python3 -m unittest discover -s tools/tests
```

Builder notebook final default menggunakan resep v59:

```bash
python3 notebook/build_final.py
```

Builder merge menggunakan resep terkunci [recipe_merge.json](artifacts/optimization/continuation_25okt/recipe_merge.json), dan membutuhkan manifest hasil merge lokal untuk mengisi bukti validasinya:

```bash
LOGIKA_NOTEBOOK_RECIPE=merge python3 notebook/build_final.py
```

Lihat [panduan notebook](notebook/README.md) sebelum mengeksekusi sel. Mode `replay` membutuhkan prediksi lokal; mode `fresh` menjalankan pipeline model dan membutuhkan data, bobot pretrained, serta runtime yang sesuai. Fresh run penuh seluruh famili belum diverifikasi ulang oleh cleanup ini.

Runner utama: `eda/valid2.py`, `eda/run_valid2.py`, dan `eda/blend2.py`. Entry point merge: `eda/260_merge_pipeline.py` dengan tahap `plan`, `build`, `blend`, dan `verify`. Skrip pembangun DSC0152 tersedia di `scripts/run_dsc_s42.sh`.

## Kebijakan riset

Data panitia saja; pretrained publik diperbolehkan. Tidak ada future leakage. Teacher, seleksi, threshold dan stopping harus bersarang di dalam training partition; outer fold hanya untuk evaluasi. Rincian aturan kerja dan perubahan historisnya ada di [handoff](handoff/README.md).

Klaim breakthrough memerlukan gain lokal > 0,003, lima fold × minimal tiga seed, mayoritas fold menang, dan paired bootstrap P(Δ > 0) ≥ 0,9 terhadap pembanding yang relevan. Hasil public kecil tidak menjadi alasan menyetel ulang bobot.

Lihat [panduan GitHub](docs/github.md) dan [cleanup 26 Oktober](docs/integration/cleanup_26okt.md). Pembersihan checkout tidak mengecilkan history Git lama.
