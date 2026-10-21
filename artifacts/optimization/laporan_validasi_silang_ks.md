# Validasi silang dan submission KS

Metrik resmi kompetisi adalah KS berdasarkan Evaluation yang diberikan pengguna. Nilai publik baseline terdahulu adalah 0,52623. Skor publik submission_ks.csv terkonfirmasi 0,54710 melalui API Kaggle.

Tiga kandidat berikut dipilih pada set tuning terpisah, kemudian parameternya dibekukan sebelum validasi silang 3 fold dengan seed 8247: extended/xgb_depth3, extended/lgb_leaves31, compact/cat_depth5. Jumlah pohon juga dibekukan; tidak ada early stopping pada fold validasi ini. Setiap prediksi OOF berasal dari model yang tidak dilatih pada baris tersebut.

| Ukuran | Baseline asli | Kandidat terpilih |
|---|---:|---:|
| Rata-rata KS tiga fold | 0.49455 | 0.51492 |
| KS seluruh prediksi OOF | 0.49294 | 0.51147 |
| ROC-AUC OOF | 0.82623 | 0.83690 |
| Average precision OOF | 0.46355 | 0.48747 |

Bobot gabungan dipilih dari grid kecil menggunakan rata-rata KS per fold. Oleh karena OOF digunakan untuk memilih bobot, angka ini merupakan hasil validasi dan dapat mengandung optimisme seleksi. Skor ini bukan hasil leaderboard dan tidak menjamin mencapai 0,56.

```json
{
  "extended/lgb_leaves31": 0.25,
  "compact/cat_depth5": 0.75
}
```

Prediksi test merupakan rata-rata tiga model fold untuk setiap anggota resep, kemudian digabungkan dengan bobot di atas. Fitur tidak memakai ID atau target, dan audit waktu observasi dilakukan sebelum pelatihan. `submission_ks.csv` mempunyai 30.000 baris sesuai urutan sample submission, dengan probabilitas kontinu valid. Versi model satu kali fit pada seluruh data tetap tersedia di `submission_v2.csv`. `submission.csv` diperbarui menjadi versi validasi silang.

Reproduksi:

```bash
python -m pip install -r environments/teman/requirements-optimization.txt
python scripts/extend_features.py
python scripts/optimize_ks.py --features base
python scripts/optimize_ks.py --features extended
python scripts/optimize_ks.py --features compact
python scripts/crossvalidate_ks.py
```
