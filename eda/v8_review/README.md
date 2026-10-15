# Audit v8 — EDA, provenance dan preprocessing CPU

Hasil dan keputusan: **[RESULTS.md](RESULTS.md)**.

Tidak ada model/inference atau notebook dijalankan. Tidak ada notebook baru, submission baru, atau perubahan hasil Kaggle. Semua analisis menggunakan prediksi tersimpan dan statistik dataset.

Jalankan dari root proyek dengan environment yang sudah tersedia:

```sh
.venv/bin/python eda/v8_review/00_artifacts_and_predictions.py
.venv/bin/python eda/v8_review/01_snapshot_identifiability.py
.venv/bin/python eda/v8_review/02_historical_domain.py
.venv/bin/python eda/v8_review/03_selected_history_features.py
.venv/bin/python eda/v8_review/04_screening_resolution.py
```

Semua skrip bernomor mencetak angka dan menyimpan gambar/tabel di `out/`. `shared8.py` hanya helper. Skrip 04 membutuhkan hasil 03 serta kandidat review v6/v7 yang sudah ada; hasil lama tidak diubah. Polars dibatasi empat thread melalui helper proyek.

| Skrip | Analisis |
|---|---|
| 00 | Mencocokkan CSV aktual dengan komponen prediksi, menghitung OOF KS, paired augmentation bootstrap, segmen dan status eksekusi |
| 01 | Mengukur ambigu tidaknya rekonstruksi snapshot historis; contoh dua payment ledger dengan snapshot akhir sama |
| 02 | Distribution/coverage shift historis vs selected dengan cutpoint bin yang sama; manifest customer split |
| 03 | Sembilan fitur histori yang sudah matang pada selected cutoff, tanpa membuat baris training historis |
| 04 | Mengulang screening 30 kandidat dengan bin yang tidak menghilangkan nilai positif langka atau kategori diskret |

Artefak preprocessing/diagnostik:

- `out/03_selected_history_features.parquet`: 100.000 aplikasi, application_id/customer_id + sembilan predictors. Target kompetisi tidak ada di file. Rasio yang tidak teramati tetap null; count kosong nol dengan flag coverage.
- `out/01_snapshot_quality.parquet`: manifest kualitas snapshot, **bukan feature matrix**. Mengandung label historis dan waktu cutoff untuk audit.
- `out/02_group_manifest.parquet`: manifest customer untuk satu diagnostic fold, **bukan predictor**. Memisahkan history milik training/validation/test customers. Bangun ulang untuk fold Kaggle yang sesungguhnya.
- `out/00_bound_oof.parquet`: actual submission blend yang direkonstruksi pada OOF, dengan application_id dan target; untuk evaluasi, **jangan dimasukkan ke fitur training**.
- `03_class_samples.csv`, `01_ambiguous_examples.csv`: contoh dari kedua kelas untuk inspeksi.

Rekonstruksi bobot CSV memakai penyelesaian persamaan linear pada **prediksi test tersimpan dan CSV yang sudah ada**, tanpa label test/target kompetisi dan tanpa optimasi bobot berdasarkan KS. Ini pemeriksaan provenance, bukan training model baru.

Fitur 03 memakai proxy historical default yang sudah ada di proyek, dengan syarat seluruh horizon 12 bulan berakhir sebelum selected cutoff. Ini tidak mengklaim bahwa ledger pembayaran harian atau historical profile dapat direkonstruksi. Nilai KS model setelah penambahan fitur belum diketahui.
