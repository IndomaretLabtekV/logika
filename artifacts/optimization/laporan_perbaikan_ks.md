# Peningkatan submission berdasarkan KS

Metrik resmi adalah Kolmogorov-Smirnov (KS), sesuai kutipan Evaluation yang diberikan pengguna. Skor publik baseline terkonfirmasi melalui API Kaggle: 0,52623. Skor publik versi baru belum tersedia; tidak ada pengunggahan otomatis.

Pemilihan parameter dan bobot gabungan menggunakan set tuning 10.500 baris yang dipisahkan dari holdout asli 17.500 baris. Semua fitur dibentuk sebelum keputusan; ID dan target tidak dipakai sebagai fitur. Kandidat diperiksa pada KS, ROC-AUC, AP, dan log loss. Hasil tuning lengkap dan seluruh resep tersimpan di direktori ini.

| Validasi | KS baseline | KS baru |
|---|---:|---:|
| Holdout stratified yang sama | 0.50671 | 0.51703 |
| Temporal Oktober–Desember | 0.48612 | 0.50381 |

Selisih KS holdout: 0.01032. Interval bootstrap berpasangan 95% untuk selisih: [-0.00150, 0.01884]. Holdout asli sebelumnya dipakai untuk melaporkan baseline; pada tahap optimasi ini ia tidak dipakai memilih parameter atau bobot. Hasil lokal tidak menjamin skor publik/private yang sama atau mencapai 0,56.

Fitur tambahan mencakup beban pinjaman aktif, perkiraan sisa pokok berdasarkan amortisasi terjadwal, keterlambatan terbaru dan per pinjaman, riwayat/tren biro, perubahan rekening, belanja 30/60/90/180/365 hari, kategori dan metode pembayaran. Perkiraan sisa pokok adalah fitur estimasi, bukan saldo aktual. Subset compact dipilih dari gain model yang dilatih pada bagian development; confirmation tidak digunakan untuk seleksi fitur.

Resep yang dikunci sebelum confirmation:

```json
{
  "extended/lgb_leaves31": 1.0
}
```

Model akhir dilatih ulang pada seluruh 70.000 label dengan jumlah pohon dan bobot tetap. `submission_v2.csv` mempunyai 30.000 probabilitas kontinu, dua kolom yang benar, ID unik dan urutan yang sama dengan sample submission. `submission.csv` diperbarui dengan isi yang sama; versi baseline dipertahankan sebagai `submission_baseline_0_52623.csv`.

Probabilitas tetap kontinu; mengubah threshold atau melakukan transformasi monoton tidak menambah kemampuan pemisahan yang diukur KS. Tidak ada threshold yang diterapkan pada submission.
