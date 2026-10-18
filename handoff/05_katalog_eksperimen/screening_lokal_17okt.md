# Screening fold 0 (17–18 Okt, validation_v2)

Semua di outer fold 0, seed 42 (14.000 baris), context/training hanya dari train fold, stopping hanya di inner 1/7.
Pembanding di fold yang sama: **resep v8 bersih 0,52107**, trees+causilo 0,52327, causilo 0,52490. Bootstrap baris
1.000× (`eda/screen2.py`). Satu fold sd ~0,008, jadi hanya angka paired yang bermakna. Output di `results/screen/`.

## Resep v8 per model (fold 0, `results/screen/ref_f0`)
lgb 0,5189 · lgb_s 0,5128 · xgb 0,5196 · xgb_s 0,5147 · cat 0,5163 · cat_s 0,5238 · TabM 0,5231 (epoch terbaik 5).

## Model zoo
| Model | KS fold 0 | Δ sendiri vs v8 (P>0) | Δ sbg +20% ke trees+causilo (P>0) |
|---|---|---|---|
| Causilo e16 / f150 / f300 / f60 | 0,5243 / 0,5253 / 0,5217 / 0,4998 | – | – (100 fitur, 4 est tetap titik terbaik) |
| TabPFN-3.5 (token, lokal) | 0,5175 | −0,0034 (0,23) | −0,0005 (0,41) |
| RealMLP-TD default (pytabkit, val = inner) | 0,5157 | −0,0075 (0,10) | −0,0005 (0,43) |
| WOE scorecard + LR (`woe_lr`) | 0,5162 | −0,0060 (0,15) | +0,0012 (0,70) |
| AutoGluon LinearModel | 0,5223 | +0,0002 (0,51) | +0,0009 (0,65) |
| AutoGluon CatBoost default | 0,5237 | +0,0014 (0,64) | +0,0004 (0,55) |
| AutoGluon NN Torch / FastAI | 0,5189 / 0,5005 | −0,0050 / −0,0227 | +0,0006 / −0,0005 |
| TabSwift | 0,5103 | −0,0115 (0,01) | −0,0001 |
| AutoGluon LGB / XGB default | 0,5160 / 0,5118 | negatif | negatif |
| RF / ExtraTrees / KNN | 0,4880 / 0,4943 / 0,4421 | sangat negatif | negatif |
| TabICLv2 (AutoGluon, jatuh ke CPU, 14 menit) | 0,5163 | −0,0055 | +0,0010 |
| TabICLv2 (NVIDIA sdm) | 0,5164 | – | – |
| iLTM (Kaggle v23) | 0,5216 | – | – |
| TabFM fp32 ctx 8k / 4k (Kaggle v27) | 0,5173 / 0,5169 | – | +20% ke v31: +0,0006 ± 0,0022 (0,60) |
| Mitra 8k baris (Kaggle v25) | 0,4488 | – | – |
| LimiX-2, TabPFN-3, Mitra, TabDPT (lokal) | OOM / ditolak di 8 GB | – | – |
| Causilo + 14 fitur utang eksternal (`eda/100`) | 0,5237 vs 0,5249 | null | – |

## Kumo-Tabular (lihat `04_riwayat_versi/v26_v28_v33_v36_kumo.md`)
| Varian | KS fold 0 |
|---|---|
| small ctx 20k / 56k; medium ctx 20k | 0,5240 / 0,5230; 0,5256 |
| large ctx 20k (draw pertama, beruntung) / 3 draw dirata-rata | 0,5276 / 0,5267 |
| large ctx 20k e8 (wrapper) | 0,5196 (bukan soal estimator) |
| large ctx 20k, 198 fitur | 0,5240 |
| large ctx penuh e4 / diulang / e8 / f154 / medium | 0,52756 / 0,52585 / 0,52780 / 0,52305 / 0,52361 |

## LR rank-gauss (`lr_q`, 5 fold lokal, CPU, detik)
OOF 0,52119 (fold 0,524/0,525/0,523/0,514/0,544) ≈ TabM bersih. Blend cross-fit v22 + lr_q: greedy 0,52503 (+0,0009 vs
resep v8, P 0,74, 5/5 fold), bobot lr_q 0,14. Dipakai di v35 (bobot 0,1).

## Pelajaran screening
- Untuk in-context learner dengan subsample context, satu draw tidak cukup (varians ±0,005). Pakai context penuh atau
  rata-rata beberapa draw; ulangi dengan seed generator lain sebelum percaya selisih < 0,002.
- Satu fold menipu ke dua arah: Kumo 20k fold 0 0,5256 terlihat terbaik, tapi pooled 0,52484 < Causilo.
- AutoGluon default (tanpa bagging, tuning_data = inner) di bawah resep v8 sendiri untuk GBDT.
