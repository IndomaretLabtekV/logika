# LOGIKA: deep EDA dan preprocessing, 13 Oktober 2026

Semua tahap sudah dijalankan sebagai analisis CPU. **Tidak ada training, inference model, eksekusi notebook, download bobot, API model, atau submission Kaggle.** Data mentah, notebook, dan hasil v1–v3 tidak ditimpa.

Baca [RESULTS.md](RESULTS.md) untuk interpretasi angka, grafik, keputusan, dan batas kesimpulannya.

## Menjalankan ulang

Dari root project:

```bash
rtk .venv/bin/python -B eda/deep/run.py
```

Satu tahap:

```bash
rtk .venv/bin/python -B eda/deep/run.py 04
```

Runner memakai maksimal empat thread Polars/OpenMP secara default, satu thread OpenBLAS, backend gambar noninteraktif, dan menjalankan tahap secara berurutan. Output angka dicetak dan disimpan sebagai CSV/JSON; gambar PNG ada di `out/`; runner menyimpan stdout/stderr tahap ke `out/<nama_script>.log`. Menjalankan ulang menimpa output EDA sendiri, bukan sumber data.

Urutan dependensi: 03 memerlukan 02; 07 memerlukan 02+03; 08 memerlukan hasil 04. Jalankan berurutan untuk reproduksi lengkap. `data_pq/features.parquet` dan OOF/test v2–v3 yang sudah tersedia adalah input diagnostik. Pembuat fitur lama tidak dijalankan ulang untuk melatih apa pun.

Environment yang dipakai: Python 3.13.16, numpy 2.5.4, polars 2.0.0, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2, pyarrow 26.0.0. Semua sudah tersedia; tidak ada dependency baru di-install. Ini environment EDA, **bukan environment kompatibilitas model**; TabPFN-Rel saat dicek mendokumentasikan Python 3.11/3.12.

Peak RSS yang terukur untuk tahap temporal sekitar **4,61 GiB**; tahap ekspor relasional sekitar 3,31 GiB, preprocessing sekitar 1,26 GiB. Angka berasal dari environment sesi, bukan benchmark langsung CPU laptop pengguna. Hindari menjalankan semua tahap paralel. Output keseluruhan sekitar **523 MiB**, terutama ekspor tabel dan matriks. Intel Arc tidak digunakan.

## File analisis

| Script | Angka dan gambar yang dihasilkan |
|---|---|
| [00_integrity.py](00_integrity.py) | Primary/foreign keys, cutoff, duplikasi, grain transaksi, contoh deduplikasi yang salah |
| [01_validation.py](01_validation.py) | KS OOF, segment/month, paired bootstrap, simulasi public/private, drift, korelasi model |
| [02_temporal_features.py](02_temporal_features.py) | 79 kandidat fitur, profil temporal, kesempatan observasi DPD, tes injeksi future rows |
| [03_residual_screen.py](03_residual_screen.py) | Screening 289 fitur, discovery/confirmation, FDR, permutation, conditioning sensitivity |
| [04_preprocessing.py](04_preprocessing.py) | Tiga transformasi, KS fitur pada lima validation fold, missing flags, contoh kelas sebelum/sesudah |
| [05_relational_export.py](05_relational_export.py) | Ekspor 10 tabel, validasi 12 foreign keys, label terpisah, diagram relasi |
| [06_feature_budget.py](06_feature_budget.py) | Daftar fitur per fold, coverage per keluarga/thin-file, riset keenam model |
| [07_observation_followup.py](07_observation_followup.py) | Kandidat spending terkuat, kontrol panjang histori, panel customer tetap |
| [08_numeric_semantics.py](08_numeric_semantics.py) | Audit presisi angka, konteks price tier, preprocessing alternatif dan pemeriksaannya |

`deep_common.py` berisi pembacaan data dan utilitas statistik; `run.py` runner. Tidak ada estimator prediktif yang di-fit. Metode `NumericPreprocessor.fit` hanya mempelajari distribusi fitur tanpa target, di training fold.

## Artefak preprocessing

- [out/candidate_features.parquet](out/candidate_features.parquet): 100.000 baris, dua ID + 79 kandidat; **jangan otomatis memasukkan semuanya**.
- [out/preprocessed/base_features_raw.parquet](out/preprocessed/base_features_raw.parquet): fitur lama dengan tipe asli dan missing, tanpa target. ID dan datetime adalah metadata, **wajib dikeluarkan dari input model**.
- [out/preprocessed/semantic_feature_view.parquet](out/preprocessed/semantic_feature_view.parquet): alternatif pembulatan tiga fitur turunan dan kategori `rate_transition`; belum dinilai oleh model.
- [out/preprocessed/train_labels.parquet](out/preprocessed/train_labels.parquet): label train terpisah.
- [out/preprocessed/folds.parquet](out/preprocessed/folds.parquet): ID fold untuk eksperimen berikutnya.
- [out/preprocessed/nn_fold0.npz](out/preprocessed/nn_fold0.npz): train 55.999, valid 14.001, test 30.000; 348 fitur numerik (210 nilai + 138 indikator missing), empat kategori.
- [out/preprocessed/nn_fold0_semantic.npz](out/preprocessed/nn_fold0_semantic.npz): fold/ID yang sama; lima kategori termasuk transisi bunga.
- [out/04_preprocessing_manifest.json](out/04_preprocessing_manifest.json) dan [out/08_semantic_manifest.json](out/08_semantic_manifest.json): urutan kolom, vocabulary, dan interpretasi array.
- [out/fm_feature_views.json](out/fm_feature_views.json): all/top100/family-balanced100, dipilih hanya dari training fold masing-masing.
- [out/relational/schema.json](out/relational/schema.json): skema intermediate dan semantik waktu, **bukan konfigurasi TabPFN-Rel siap fit**.

NPZ dapat dibaca dengan `np.load(path, allow_pickle=False)`. ID berupa string biasa; kolom `X_*`, `cat_*`, `y_*` terpisah. Kategori 0 dicadangkan untuk unknown; ukuran vocabulary untuk model adalah `len(vocab)+1`. Jangan menyamakan 0 kategori dengan kode missing numerik.

Array fold 0 sengaja satu fold saja untuk hemat storage. Lima fold sudah diaudit, tetapi pada training remote **fit ulang preprocessing per fold**, jangan memecah matriks transformasi fold 0 lagi menjadi CV baru. Split audit ini bukan ID fold model v3. Seluruh ID OOF lama diikat ke urutan cached feature matrix lewat verifikasi metric dan kecocokan CSV; file NPY lama memang tidak menyimpan ID.

Normalisasi interpolated menyatukan nilai di luar rentang training pada batas distribusi; bukan pemetaan invertible. Klaim preservasi KS hanya pada pemeriksaan validation yang tercatat. Missing flags dipelajari per kolom; transformasi gagal eksplisit apabila muncul missing pada kolom yang sepenuhnya observed di fit. Pengelompokan 138 mask menjadi 29 mask cocok pada split saat ini, tetapi alternatif ini memakai mask per kolom agar tidak mengandalkan kebetulan pola identik.

Untuk GBDT dan foundation model dengan preprocessing native, mulai dari raw/NaN atau semantic raw, **jangan otomatis memakai matriks Gaussian untuk semua model**.

## Batas evaluasi

- KS model yang dihitung adalah KS dari prediksi lama. KS setelah training model dengan fitur/preprocessing baru **belum diketahui**.
- KS fitur tunggal, conditional association, dan grafik kelas bukan bukti kenaikan KS ensemble.
- Discovery/confirmation dibagi baru, tetapi label sebelumnya sudah digunakan oleh EDA dan training/early stopping v3. P-value dan bootstrap adalah diagnostik eksploratif, bukan fresh independent validation.
- Tidak ada cara menjamin public dan private dekat hanya dari KS validation. Ukuran public asli dan label test tidak tersedia.
- Tidak dilakukan probing leaderboard atau rekonstruksi label test.
