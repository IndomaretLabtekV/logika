# Probe submission 16 Okt 2026

Diminta user: "generate 3 submission buat ngetest, yang bisa dijadiin bahan analisis", disubmit satu per satu, nama
file tetap `submission.csv`, penjelasan di description, analisis tiap skor sebelum lanjut. Kuota hari itu 3 submit
(CLI: "2 submissions remaining today" setelah submit pertama), jadi **batas harian = 3**.

File ada di `submissions/probe/` (README.md + expectations.csv). S1/S2 (rancangan awal, tidak disubmit) sudah dihapus.

## Pembanding
Blend v8 yang disubmit (public 0,56145). Bobot rank-nya direkonstruksi dengan NNLS dari `results/v8/submission.csv`
terhadap rank 16 prediksi test v8 (korelasi 0,99999999993):

| Model | Bobot | Model | Bobot |
|---|---|---|---|
| lgb | 0,032 | lgb_aug | 0,108 |
| lgb_s | 0,018 | lgb_s_aug | 0,038 |
| xgb | 0,015 | xgb_aug | 0,093 |
| xgb_s | 0,027 | xgb_s_aug | 0,057 |
| cat | 0,012 | cat_aug | 0,025 |
| cat_s | 0,022 | cat_s_aug | 0,020 |
| tabm | 0,145 | tabm_aug | 0,207 |
| tabpfn | 0,045 | causilo | 0,137 |

GBDT total 0,467; NN/FM 0,533. OOF KS blend v8 dengan bobot ini: **0,52703**.

Perkiraan public = 0,56145 + 1,4 × ΔOOF (slope empiris dari versi lama). sd selisih public = sd selisih KS pada
subsample 9.000 baris OOF (30% dari 30k).

## Hasil

| Urutan | Submission | Isi | OOF KS | ΔOOF | Perkiraan | Public | z |
|---|---|---|---|---|---|---|---|
| 1 (15:08 UTC) | **S3** | stack logit-LR 9 model v8 (lgb, lgb_s, xgb, xgb_s, cat, cat_s, tabm, tabpfn, causilo) + TabICL17; OOF dari CV 5-fold | 0,52813 | +0,0011 | 0,563 ± 0,0028 | **0,55778** | −1,86 |
| 2 (15:13 UTC) | **S5** | v8, bagian NN/FM saja (tabm .27, tabm_aug .39, tabpfn .08, causilo .26), rank-average | 0,52776 | +0,0007 | 0,5625 ± 0,0024 | **0,55866** | −1,6 |
| 3 (15:14 UTC) | **S4** | v8, bagian GBDT saja (12 tree termasuk _aug, bobot v8 dinormalisasi) | 0,52376 | −0,0033 | 0,5569 ± 0,0026 | **0,56133** | +1,7 |

## Analisis per langkah
- **Setelah S3:** TabICL17 dengan bobot positif di blend rank v8 justru menurunkan OOF (5%: −0,0005; 10%: −0,0007;
  20%: −0,0018). Gain S3 di OOF hanya datang dari bobot negatif TabICL (−0,22, "kontras" terhadap TabPFN). Pola:
  blend rank bobot positif (v5/v6/v8: 0,56145/0,56123/0,56145) vs stack LR berbobot negatif (v9 0,55917, S3 0,55778).
  Rencana S1/S2 diganti dengan dekomposisi blend v8 (S4/S5) supaya hasilnya bisa langsung dibandingkan dengan v8.
- **Setelah S5:** semua kandidat "OOF > v8" (v6, v9, S3, S5) turun di public. S3 dan S5 sama-sama mengurangi porsi
  GBDT → S4 jadi uji penentu.
- **Setelah S4:** GBDT saja ≈ v8 di public walau OOF-nya −0,0033. S4−S5: OOF −0,0040 (perkiraan public −0,0056),
  public +0,0027, sd 0,0036, **z 2,31, p ≈ 0,03**. Lalu dicari penyebabnya di kode → `bias_oof_early_stopping.md`.

## Catatan
- Ini bukan LB probing label (tidak ada rekonstruksi label test); yang dibandingkan adalah keluarga model utuh.
- Satu probe saja selalu ±0,003; kesimpulan baru kuat karena S4 dan S5 berlawanan arah dari OOF.
- Analisis ulang skor: `.venv/bin/python eda/s04_probe_analysis.py S3=0.55778 S4=0.56133 S5=0.55866`. Script
  menghitung z tiap submission terhadap perkiraannya; |z| < 2 per submission dianggap masih dalam noise (kesimpulan
  kuat datang dari selisih S4−S5, bukan dari satu baris). Baris S1/S2 di expectations.csv tidak punya file lagi.
