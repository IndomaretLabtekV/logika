# Review v1–v7 dan preprocessing CPU

Keputusan dan penjelasan: **[RESULTS.md](RESULTS.md)**.

Tidak ada training/inference model, eksekusi notebook, atau submission. Semua hasil berasal dari notebook yang dibaca sebagai JSON, prediksi tersimpan, dan operasi statistik/preprocessing CPU. Notebook dan hasil Kaggle asli tidak diubah.

Jalankan berurutan dari root proyek:

```sh
.venv/bin/python eda/v7_review/00_versions.py
.venv/bin/python eda/v7_review/01_metric_and_errors.py
.venv/bin/python eda/v7_review/02_price_measurement.py
.venv/bin/python eda/v7_review/03_realmlp_inputs.py
.venv/bin/python eda/v7_review/04_fold_score_scales.py
```

Kelima skrip mencetak angka dan menyimpan tabel serta PNG di `out/`. `shared7.py` hanya helper; menggunakan ulang helper dari review sebelumnya. Tidak ada dependency baru; environment `.venv` yang sudah tersedia digunakan, dengan Polars maksimum empat thread.

| Skrip | Pertanyaan yang dijawab |
|---|---|
| 00 | Di versi mana gain terjadi? Apakah v7 berbeda secara meyakinkan dari v5/v6? |
| 01 | Apakah AUC sejalan dengan KS? Bagaimana aggregation tetap dan error pada threshold dari separuh lain? |
| 02 | Seberapa stabil premium harga dengan sedikit order? Apakah order-equal mean, median, atau fixed shrinkage membantu? |
| 03 | Bagaimana menyediakan input finite untuk RealMLP tanpa menghapus arti missingness dan tipe kategori? |
| 04 | Apakah mean fold KS lebih tinggi karena masalah skala prediksi, atau karena maksimum KS dihitung pada subset? |

Artefak preprocessing:

- `out/02_price_candidates.parquet`: 100.000 aplikasi, ID dan enam kandidat fitur harga; bukan fitur yang sudah terbukti meningkatkan model.
- `out/02_reference_fold0.parquet`: referensi harga dari training customers pada fold diagnostik 0, historis sebelum bulan query.
- `out/03_median_inputs_fold0.parquet`: ID + 228 numerik terimputasi + 228 missing flags + empat kategori string. Tanpa target. Native model preprocessing/scaling belum dijalankan.
- `out/03_training_manifest.parquet`: application_id, target, diagnostic fold. Pisahkan dari predictors.
- `out/03_preprocessor.json`: urutan kolom, median, vocabulary, seed, dan batas penggunaan.
- `02_class_samples.csv`, `03_class_samples.csv`: sampel kedua kelas untuk inspeksi.

**Jangan menggunakan preprocessing fold 0 untuk semua fold model.** Refit median, vocabulary, dan reference pada training split yang sesungguhnya. Fold diagnostik 71015 bukan fold sklearn notebook. Skrip 04 secara terpisah merekonstruksi fold seed 42 dan mencocokkannya dengan KS per-fold di log TabPFN; tidak menginstal atau menjalankan estimator sklearn.

Khusus v4, hasil OOF diambil dari log notebook karena `results/v4` hanya menyediakan submission utama. v1 disejajarkan dengan manifest ID lama; v2–v6 memakai urutan legacy yang konsisten dengan log. v7 diverifikasi dengan ID dan target asli. Detail provenance ini penting saat membaca perbandingan.
