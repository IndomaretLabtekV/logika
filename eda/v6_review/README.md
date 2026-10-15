# Audit v6 — CPU EDA dan preprocessing

Baca **[RESULTS.md](RESULTS.md)** untuk keputusan dan interpretasi. Semua angka baru berasal dari eksekusi skrip di folder ini; tidak ada model atau notebook yang dijalankan. Input notebook, hasil Kaggle, dan cache fitur tidak diubah.

Jalankan dari root proyek dengan environment yang sudah tersedia:

```sh
.venv/bin/python eda/v6_review/00_v6_vs_v5.py
.venv/bin/python eda/v6_review/01_exposure_and_provider.py
.venv/bin/python eda/v6_review/02_price_reference.py
.venv/bin/python eda/v6_review/03_synthetic_integrity.py
```

Setiap skrip bernomor mencetak angka dan menyimpan PNG serta CSV/JSON di `out/`. `review_common.py` adalah helper, bukan analisis terpisah. Implementasi memakai helper statistik dan referensi harga yang sudah ada di `eda/v5_review`, tanpa instalasi dependency baru. Polars dibatasi empat thread.

| Skrip | Hasil utama |
|---|---|
| 00 | KS/AUC prediksi tersimpan, perubahan per segmen, bootstrap berpasangan, simulasi sampling, korelasi, contoh error, hash input |
| 01 | Audit opportunity DPD 30/60/90, perubahan skor biro dalam provider yang sama, 15 kandidat dan screening residual |
| 02 | Perbandingan referensi harga v6 dengan referensi training-customer-only, uji perturbasi dan pemeriksaan mean/std independen |
| 03 | Identitas keuangan sintetis, quantization, duplikasi, contoh agregasi order yang benar |

Artefak preprocessing:

- `out/01_candidates.parquet`: 100.000 aplikasi, ID + 15 fitur berbasis histori sendiri; null berarti tidak teramati/tidak ada exposure, bukan otomatis nol.
- `out/02_price_fold0.parquet` dan `02_reference_fold0.parquet`: demonstrasi satu fold diagnostik, **bukan** fitur untuk dipakai di semua fold Kaggle. File pertama mengandung target untuk inspeksi; hanya `fold_price_mean` adalah kandidat predictor. Refit reference pada training IDs setiap fold model yang sebenarnya.
- `out/03_order_spend.parquet`: agregasi sekali per order, ID + nominal dan jumlah order, termasuk kolom naive untuk pembanding. Bukan pengganti seluruh fitur e-commerce.
- `*_class_samples.csv`: sampel preprocessing dari kedua kelas, agar hasil transformasi bisa diperiksa langsung.

Skrip 00 menghitung ulang KS dari OOF tersimpan, bukan validasi model baru. NPY lama tidak menyimpan ID; kesesuaian log dan rekonstruksi submission mendukung urutan baris, tetapi tidak membuktikan provenance baris training secara independen. Simpan ID bersama prediksi pada eksperimen Kaggle berikutnya.
