# Audit v5 dan EDA lanjutan

Mulai dari [RESULTS.md](RESULTS.md) untuk temuan, keputusan, kesalahan v5, dan prioritas eksperimen. Rangkuman EDA sebelumnya tetap ada di [../deep/RESULTS.md](../deep/RESULTS.md).

Folder ini hanya menjalankan analisis CPU, statistik atas prediksi tersimpan, dan preprocessing. Tidak mengimpor/menjalankan model, mengeksekusi notebook, mengunduh bobot, atau mengirim submission.

| Script | Analisis dan output utama |
|---|---|
| [00_saved_predictions.py](00_saved_predictions.py) | Rekonstruksi submission/OOF, KS v3–v5, bootstrap berpasangan, KS per bulan/segmen, simulasi public–private, korelasi prediksi. |
| [01_price_preprocessing.py](01_price_preprocessing.py) | Duplikasi dan grain order, referensi harga per fold dan waktu, 29 kandidat, uji perturbasi, contoh dua kelas, grafik sebelum/sesudah. |
| [02_residual_screen.py](02_residual_screen.py) | Uji asosiasi bersyarat terhadap v5, discovery/confirmation, FDR, kontrol negatif, korelasi dan cohort temporal. |
| [03_numeric_preprocessing.py](03_numeric_preprocessing.py) | Perbandingan ECDF global/per fold, transformasi interpolasi, missing masks, probe EXAONE diperbaiki, ekspor numerik gabungan. |
| [04_feature_views.py](04_feature_views.py) | Audit DFS versus handcrafted+DFS; seleksi training-fold saja; fitur raw untuk perbandingan model kelak. |
| [05_drift_and_timing.py](05_drift_and_timing.py) | Drift fitur/prediksi train–test, mixture bulan, availability dan cutoff tiap tabel. |
| [preprocess.py](preprocess.py) | Menggunakan ulang NumericPreprocessor yang sudah diverifikasi di EDA sebelumnya; fungsi pembagian probe stratified yang diperbaiki. |
| [shared.py](shared.py), [run.py](run.py) | I/O, assert, pembatas thread, runner berurutan dan log. |

Jalankan dari root project dengan environment yang sudah ada:

```bash
rtk proxy .venv/bin/python -B -u eda/v5_review/run.py
```

Satu analisis, misalnya setelah output 01 tersedia:

```bash
rtk proxy .venv/bin/python -B -u eda/v5_review/run.py 02_residual_screen
```

Dependensi yang dipakai: NumPy, pandas, Polars, SciPy, matplotlib, pyarrow. Tidak menambahkan dependensi. Runner memakai subprocess berurutan; jangan menjalankan dua runner ke folder output yang sama. Batas default: Polars 4, OMP 2, OpenBLAS 1 thread. Referensi file tetap relatif terhadap project, bukan working directory pemanggil. Runtime dan batas thread efektif dicatat saat verifikasi.

Semua hasil ada di [out/](out/): CSV/JSON untuk angka, PNG untuk grafik, log untuk output lengkap, Parquet/NPZ untuk preprocessing. Enam script analisis masing-masing menghasilkan angka dan gambar. Puncak RAM run final sekitar 3.6 GiB; angka waktu di `out/run_stats.json` adalah pengukuran lingkungan ini, bukan janji kecepatan laptop.

**Cara memakai output preprocessing**

- `01_price_fold0.parquet` sampai `01_price_fold4.parquet`: 100.000 aplikasi per file. Referensi hanya berasal dari customer training fold terkait; observasi pembentuk referensi harus sebelum awal bulan keputusan query. Simpan fitur per fold, jangan digabung menjadi satu referensi global.
- `01_candidates_oof.parquet`: bagian train memakai referensi fold masing-masing; bagian test berupa mean lima referensi untuk EDA. Ini bukan matriks universal untuk melatih semua fold.
- `03_fold_assignment.parquet`: ID dan fold diagnostik seed 20261014. Pembagi ini deterministik tetapi **bukan** StratifiedKFold sklearn milik v5. Untuk modeling, bangun ulang semua transformasi pada fold aktual model.
- `03_fold0_numeric_diagnostic.npz`: ekspor gabungan harga aman + preprocessing numerik training-fold0. Berisi 100.000 baris, 391 kolom numerik (240 fitur + 151 mask), 4 kategori, ID, nama kolom, dan ID baris fit. Tidak berisi prediksi baru. Dapat dibuka dengan `np.load(..., allow_pickle=False)`.
- `04_joint_raw_fold0.parquet`: ID, split, 100 fitur numerik raw + 4 kategori. NaN tetap NaN; hanya untuk fold0 yang sama. Daftar `exaone_columns_total100` di manifest membatasi ke **96 numerik + 4 kategori**. Ini budget untuk menghindari tahap seleksi internal EXAONE, bukan jaminan kompatibilitas T4.
- `00_bound_oof.parquet` dan hash input mengikat interpretasi hasil ke file yang diaudit. NPY lama tidak memiliki ID bawaan; kesesuaian log dan submission mendukung alignment, bukan bukti identitas baris yang mutlak.
- Mengulang suite menimpa hanya hasil di folder `out/` ini. Notebook, results/v1–v5, dan raw data tidak diedit.

Assert meliputi ID/grain, referensi independen, perturbasi validation dan future, finiteness/mask, row-order/batch invariance, pemisahan kelas probe, serta rekonstruksi submission. KS model setelah preprocessing baru masih belum tersedia karena model tidak dijalankan.
