# Hasil Kumo Tabular — 18 Oktober 2026

Kumo Large memperbaiki KS pada ketiga fold dibanding V5. Kandidat utama adalah `submission_kumo.csv`, Kumo murni. Public score Kumo murni adalah **0,56262**, dilaporkan pengguna dan dikonfirmasi melalui API Kaggle pada 18 Oktober 2026. Skor campuran belum diketahui. Kenaikan terhadap V3 (0,55845) adalah **0,00417** dan terhadap V5 (0,55680) **0,00582**. Riwayat akun juga memuat skor 0,56188 dari submission lain yang modelnya tidak teridentifikasi di repo; skor baru unggul 0,00074 terhadap catatan tersebut.

| Model | KS OOF pooled | Rata-rata KS fold | Selisih pooled vs V5 | Fold membaik |
|---|---:|---:|---:|---:|
| V4 | 0.52410945 | 0.52740391 | -0.00117589 | 2/3 |
| V5 | 0.52528534 | 0.52752392 | +0.00000000 | 0/3 |
| Kumo Large | 0.52806752 | 0.52907139 | +0.00278218 | 3/3 |
| 75% V5 + 25% Kumo Large | 0.52674351 | 0.52932389 | +0.00145817 | 3/3 |
| 50% V5 + 50% Kumo Large | 0.52791038 | 0.52994360 | +0.00262504 | 3/3 |
| Kumo Medium | 0.52656468 | 0.52756783 | +0.00127934 | 2/3 |
| 75% V5 + 25% Kumo Medium | 0.52719269 | 0.52905783 | +0.00190735 | 3/3 |
| 50% V5 + 50% Kumo Medium | 0.52847668 | 0.52922682 | +0.00319134 | 2/3 |

## Metode dan ruang lingkup

- `StratifiedKFold(n_splits=3, shuffle=True, random_state=8247)`, identik dengan seleksi V5. Seluruh 70.000 baris memiliki tepat satu prediksi OOF.
- Baseline V5 dihitung dari 75% V4 + 25% TabM pada OOF seleksi tiga fold. Angka akhir V5 sepuluh fold 0,52446 tidak dipakai sebagai pembanding karena pembagian training berbeda.
- Bobot pretrained NVIDIA `nvidia/Kumo-Tabular`, revisi `v1.0.1`, commit `3c3e10bbdb590ace29e7026847f92db3c603096d`; paket SDM 0.2.1. Bobot dibekukan: `fit` membentuk preprocessing dan cache konteks, bukan gradient training.
- Sebanyak 90 fitur mengikuti konfigurasi TabM V5. ID, tanggal mentah, split, dan target dikeluarkan dari X. Agregasi riwayat tetap mengikuti cutoff sebelum keputusan.
- Medium: subset stratified 32.768 konteks, 4 anggota ensemble. Large: seluruh 46.666–46.667 baris training fold, 8 anggota ensemble. Tidak ada baris validation dalam konteks prediksinya sendiri.
- Resep preprocessing default SDM dipelajari dari konteks setiap fold. Label validation tidak dipakai dalam imputasi, transformasi, ataupun konteks. Proporsi positif konteks mengikuti data asli.
- Query batch 8.192, estimator batch 1, autocast float16, GPU A100 80 GB. Stabilitas prediksi antarukuran batch diperiksa pada setiap fold. Selisih kecil float16 dicatat pada metadata.
- Pemilihan fitur/model menggunakan label training yang sama dengan eksperimen sebelumnya. Hasil ini mengandung optimisme seleksi; belum ada konfirmasi sepuluh fold. Hasil public Kumo murni 0,56262 merupakan evaluasi test terpisah; skor private belum diketahui. Perbedaan ukuran, konteks, dan jumlah estimator tidak mengisolasi efek ukuran model.

## CSV dan notebook

- `submission_kumo.csv`: rata-rata prediksi test tiga model konteks Kumo Large yang sama dengan validasi.
- `submission_kumo_blend.csv`: 75% baseline V5 dari model seleksi tiga fold + 25% Kumo Large. Ini berbeda dari mencampur CSV V5 public yang memakai sepuluh fold.
- Bobot campuran 25% sudah digunakan sebagai kandidat saat eksperimen Medium, sebelum seluruh hasil Large tersedia. Bobot 50% ditampilkan sebagai diagnosis, bukan dipromosikan hanya karena memaksimalkan rata-rata OOF.
- Masing-masing CSV berisi 30.000 probabilitas dan urutan `application_id` sama dengan sample. `submission.csv` dan checkpoint V3/V4/V5 dipertahankan.
- [Notebook 08](../../../notebook/08_kumo_tabular.ipynb) memuat EDA, seluruh kode fitur, inferensi kedua konfigurasi, perbandingan OOF, dan ekspor. `FORCE_RERUN_KUMO=True` mengulang inferensi; aktifkan juga ketika mengubah kode preprocessing.
- Data mentah, bobot model, dan prediksi per baris tetap lokal serta dikecualikan dari Git. Metadata konfigurasi, metrik, checksum CSV, dan verifikasi disertakan.

Sumber model: [NVIDIA Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular); [implementasi SDM](https://github.com/NVIDIA/structured-data-models).

Catatan public beserta checksum dan waktu upload: [public_leaderboard_result.json](public_leaderboard_result.json). API mengonfirmasi skor serta nama file; checksum lokal dicatat terpisah karena API tidak menyediakan checksum upload.
