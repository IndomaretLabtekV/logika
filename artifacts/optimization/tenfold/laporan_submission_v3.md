# Submission versi 3 untuk KS

Versi sebelumnya (`submission_ks.csv`) memperoleh public KS **0,54710**, terkonfirmasi lewat API Kaggle. Versi 3 memperoleh public KS **0,55845** terkonfirmasi melalui API Kaggle, meningkat **0,01135**. Saat pengecekan 12 Oktober 2026, peringkat 10 bernilai 0,56333. Tidak ada upload otomatis.

Penambahan fitur mencakup harga per unit dan komposisi belanja per kategori, variasi belanja bulanan, shrinkage frekuensi keterlambatan historis, skor biro berbobot usia laporan, dan cakupan cicilan oleh arus kas. Semua observasi diaudit sebelum keputusan. Prior keterlambatan berasal dari kejadian historis yang diketahui, bukan label target.

| Validasi yang sama, tiga fold | Versi 0,54710 | Resep baru |
|---|---:|---:|
| Rata-rata KS fold | 0.51492 | 0.52285 |
| KS seluruh OOF | 0.51147 | 0.51970 |
| ROC-AUC OOF | 0.83690 | 0.84124 |

Setelah memilih jumlah pohon dan bobot, resep dikunci sebelum pelatihan akhir 10 fold. Setiap model baru berlatih pada 90% label dan seed bervariasi menurut fold. Parameter dan bobot tidak diubah berdasarkan hasil 10 fold. Prediksi tiap anggota dirata-ratakan antarfold; jika resep memuat `public_054710`, bagian itu memakai prediksi versi lama yang sudah tersimpan.

```json
{
  "price_lgb31_balanced_250": 0.3333333333333333,
  "price_cat5_balanced_700": 0.3333333333333333,
  "public_054710": 0.3333333333333333
}
```

KS OOF gabungan pada tahap akhir: **0.51901**, dibanding **0.51147** untuk prediksi OOF versi lama pada 70.000 baris yang sama. OOF baru memakai model dengan proporsi training 90%; OOF versi lama 67%. Skor rata-rata fold 3 dan 10 tidak dibandingkan langsung karena ukuran fold memengaruhi statistik maksimum KS.

Seleksi fitur, jumlah pohon, dan bobot pernah memakai label training yang sama; hasil lokal dapat mengandung optimisme seleksi dan tidak menjamin skor public/private tertentu. Tidak ada label test yang digunakan.

File `submission_v3.csv` berisi 30.000 probabilitas kontinu sesuai header dan urutan sample submission; seluruh probabilitas finite dan di [0,1]. Checkpoint 0,54710 dipertahankan terpisah. `submission.csv` diperbarui ke versi 3.

Reproduksi:

```bash
python scripts/build_features_v3.py
python scripts/refine_ks.py
python scripts/refine_ks.py --balanced
python scripts/select_refinement.py
python scripts/train_tenfold_submission.py
```
