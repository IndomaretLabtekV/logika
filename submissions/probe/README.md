# Probe submissions (16 Okt 2026)
Pembanding: blend v8 yang disubmit (bobot direkonstruksi NNLS dari submission, korelasi 0.99999999993), OOF KS 0.52703, public 0.56145.

| file | isi | OOF KS | ΔOOF vs v8 | ekspektasi public | sd selisih public (9k baris) | Spearman test vs v8 |
|---|---|---|---|---|---|---|
| S1_gbdt_only (tidak disubmit, file dihapus) | rank-avg 6 GBDT (lgb, lgb_s, xgb, xgb_s, cat, cat_s) | 0.52314 | −0.0039 | ~0.556 | 0.0031 | 0.9969 |
| S2_nn_fm_only (tidak disubmit, file dihapus) | rank-avg tabm, tabm_aug, tabpfn, causilo, tabicl17 | 0.52470 | −0.0023 | ~0.558 | 0.0028 | 0.9966 |
| S3_stack_tabicl17 | logit-LR stack 9 model v8 + tabicl17 (OOF dari CV 5-fold) | 0.52813 | +0.0011 | ~0.563 | 0.0028 | 0.9978 |
| S4_v8_gbdt_part | bagian GBDT blend v8 (12 tree, bobot v8 dinormalisasi) | 0.52376 | −0.0033 | ~0.5569 | 0.0026 | 0.9983 |
| S5_v8_nn_part | bagian NN/FM blend v8 (tabm, tabm_aug, tabpfn, causilo) | 0.52776 | +0.0007 | ~0.5625 | 0.0024 | 0.9987 |

Ekspektasi public = 0.56145 + 1.4·ΔOOF (slope empiris). Bacaan:
- S1 vs S2: apakah public lebih "suka" keluarga GBDT atau NN/FM dibanding OOF. Kalau urutannya terbalik dari OOF
  dengan selisih > ~2·sd (0.006), subset public punya karakter berbeda dari train.
- S3 vs v8: apakah gain OOF stack terbawa ke public. Selisih < 0.006 = tidak bisa dibedakan dari noise.
- Catatan: v9 (Spearman 0.9993 vs v8-stack) dapat −0.0023 di public → perubahan kecil pun menggeser public ±0.002–0.003.

## Hasil
- **S3** (16 Okt 15:08 UTC): public **0.55778** = v8 −0.0037 (perkiraan +0.0015, z −1.86).
  TabICL17 dengan bobot positif di blend rank v8 justru menurunkan OOF (5%: −0.0005, 20%: −0.0018); gain S3 di OOF
  hanya datang dari bobot negatif (kontras ICL) yang tidak terbawa ke public.
  Pola: blend rank bobot positif (v5 0.56145, v6 0.56123, v8 0.56145) vs stack logit-LR berbobot negatif
  (v9 0.55917, S3 0.55778): dua-duanya stack LR kalah dari v8.
- **S5** v8 bagian NN/FM saja (tabm .27, tabm_aug .39, tabpfn .08, causilo .26): public **0.55866** = v8 −0.0028 (OOF +0.0007, z −1.6).
- **S4** v8 bagian GBDT saja (12 tree termasuk _aug): public **0.56133** = v8 −0.0001 (OOF −0.0033, z +1.7).
- S4−S5: OOF −0.0040 (perkiraan public −0.0056), public **+0.0027**; sd 0.0036 → **z 2.31, p≈0.03**.

## Diagnosis
Semua model v8 early-stop pada fold validasi = baris OOF (LGB valid_sets=[dva], XGB evals=[(dva)], CatBoost eval_set
use_best_model, TabM pilih epoch terbaik via AUC y[vai]). OOF v8 jadi optimis, dan TabM lebih optimis daripada GBDT:
- LGB: v8 0.5239 vs kontrol inner-stop bersih v10 0.5191 (gap ~0.005; sebagian karena 3 seed vs 1)
- TabM: v8 0.5257 vs kontrol bersih v11 0.5175 (gap ~0.008)
Akibatnya bobot blend greedy dan semua kandidat "OOF > v8" berbasis OOF v8 condong ke NN dan tidak terbawa ke public
(v6, v9, S3, S5 semuanya turun). Titik public: α_GBDT 0 → 0.55866, 0.467 (v8) → 0.56145, 1 → 0.56133.

Langkah lanjut: jalankan ulang pipeline v8 dengan early stopping di inner split (atau iterasi tetap dari inner CV)
supaya OOF jujur, lalu pilih bobot blend dari OOF bersih itu, bukan dari public.

Penjelasan lengkap: `handoff/07_validasi_dan_leaderboard/probe_submission_16okt.md` dan `bias_oof_early_stopping.md`.
