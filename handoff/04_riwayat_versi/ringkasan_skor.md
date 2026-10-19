# Ringkasan skor semua versi dan submission

⚠ Angka OOF v1–v9 **optimis** karena early stopping di fold validasi (TabM paling parah). Lihat
`07_validasi_dan_leaderboard/bias_oof_early_stopping.md`. Run v10–v21 memakai inner split (bersih), jadi angka
OOF-nya lebih rendah tapi jujur; jangan dibandingkan langsung dengan v1–v9.

## Semua submission (dikonfirmasi lewat `kaggle competitions submissions`)
| Tanggal (UTC) | Submission | Isi | Public |
|---|---|---|---|
| 11 Okt | submission_lightgbm / xgboost / catboost | baseline teman (repo awal, sebelum v1) | 0,50227 / 0,50053 / 0,50662 |
| 13 Okt 04:08 | v1 | baseline 176 fitur, equal rank blend | 0,53139 |
| 13 Okt 07:14 | v2 | +38 fitur residual + shallow trees | 0,54192 |
| 13 Okt 13:59 | v3 | TabPFN ctx penuh + TabM | 0,54545 |
| 14 Okt 06:01 | v5 | **price premium pp_*** | **0,56145** |
| 14 Okt 11:34 | v6 | referensi harga aman-waktu, nn_prep per fold | 0,56123 |
| 15 Okt 06:41 | v8 | augmentasi historis, blend rank greedy (54,8% bobot di model _aug) | **0,56145** |
| 15 Okt 13:36 | v9 | stack logit-LR 11 model | 0,55917 |
| 16 Okt 15:08 | probe S3 | stack logit-LR 9 model v8 + TabICL17 | 0,55778 |
| 16 Okt 15:13 | probe S5 | v8 bagian NN/FM saja | 0,55866 |
| 16 Okt 15:14 | probe S4 | v8 bagian GBDT saja | 0,56133 |

Belum pernah disubmit: v4, v7, `results/v8/submission_v9_stack.csv` (stack LR 9 model v8).
Batas submit **3 per hari** (dikonfirmasi 16 Okt).

## Leaderboard (laporan user / sesi)
- Top-1: 0,57059 (13 Okt) → 0,57184 → 0,57458 → 0,57493 (15 Okt) → **0,57768** (16 Okt, tim "cupuu").
- Cutoff top-8 (lolos final) sekitar **0,57005** (16 Okt).
- Tim kita "lagi mager lomba bentar", peringkat sekitar 34 dengan 0,56145.

## OOF per versi (v1–v9, bias)
| Versi | Perubahan utama | OOF / held-out | Public |
|---|---|---|---|
| v1 | baseline | 0,5051 | 0,53139 |
| v2 | fitur residual + shallow | 0,5135 (greedy in-sample) | 0,54192 |
| v3 | TabPFN ctx penuh, TabM (LimiX hang, dibuang) | 0,5142 | 0,54545 |
| v4 | eksperimen Astra (spend/semantic) | 0,5114 | tidak tercatat |
| v5 | pp_* (+0,014) | 0,5274 | 0,56145 |
| v6 | kepatuhan cutoff, tabm_p | 0,5283 in-sample / 0,5263 held-out | 0,56123 |
| v7 | Causilo, TabPFN-Rel, EXAONE crash | 0,5268 held-out | – |
| v8 | augmentasi historis, TabICL crash | blend aktual 0,52703 | 0,56145 |
| v8-stack | stack LR 9 model (lokal) | 0,5281 held-out | – |
| v9 | stack LR 11 model | 0,5272 held-out | 0,55917 |

## Eksperimen 16 Okt (protokol bersih kecuali disebut lain)
| Run | Isi | OOF KS | + stack v8 (bias) |
|---|---|---|---|
| v10 | LGB nested tuning: control / tuned | 0,5191 / 0,5214 | campuran 80/20 −0,0013 |
| v11 | TabM optimizer: control / tuned | 0,5175 / 0,5199 | campuran 80/20 −0,0008 |
| v12 | TabM kapasitas: control / tuned | 0,5186 / 0,5192 | −0,0003 |
| v13 | EBM aditif / 20 interaksi | 0,5172 / 0,5185 | −0,0001 |
| v14 | hybrid set-encoder / events saja | 0,5187 / 0,5107 | −0,0002 / −0,0001 |
| v15 | rank_xendcg / XGB pairwise / L2 meta / L2 full | 0,5178 / 0,5001 / 0,5248 / 0,5229 | ≈ 0 |
| v16 | TabICL ctx 40k | 0,5079 | +0,0006 (bobot negatif) |
| v17 | TabICL ctx penuh, 8 est | 0,5133 | +0,0006–0,0007 (bobot negatif, P>0 0,69) |
| v18 | linear_tree / monotone / ExtraTrees | 0,5197 / 0,5177 / 0,5001 | ≈ 0 |
| v19 | pseudo-label (base / soft1 / soft.3 / hard) | 0,5200 / 0,5197 / 0,5206 / 0,5208 | −0,0001 |
| v20 | Causilo/TabPFN 16 est | tidak jalan (kuota GPU) | – |
| v21 | LGB base / +mature / pruned | 0,5214 / 0,5203 / 0,5191 | ≈ 0 |

## KS per model v5–v9 (OOF, bias; dari handoff lama)
| Model | v5 | v6 | v7 | v8 | v9 |
|---|---|---|---|---|---|
| lgb | 0,5241 | 0,5246 | — | 0,5239 | 0,5248 |
| lgb_s | — | 0,5243 | — | 0,5245 | 0,5245 |
| xgb | 0,5235 | 0,5224 | — | 0,5224 | 0,5224 |
| xgb_s | — | 0,5241 | — | 0,5241 | 0,5241 |
| cat | 0,5234 | 0,5211 | — | 0,5213 | 0,5216 |
| cat_s | — | 0,5204 | — | 0,5208 | 0,5213 |
| tabm | 0,5257 | 0,5248 | 0,5253 | 0,5257 | 0,5253 |
| tabm_p | — | 0,5256 | 0,5251 | — | 0,5245 |
| realmlp | — | — | — | — | 0,5223 |
| tabpfn (fitur manual) | — | 0,5186 | 0,5189 | 0,5186 | 0,5187 |
| tabpfn DFS / TabPFN-Rel | 0,4912 | — | 0,4917 | — | — |
| causilo | — | — | 0,5226 | 0,5241 | 0,5239 |
| EXAONE | gagal | probe 0,524 (3k) | crash 56k | tidak jalan | dibuang |

v8 juga punya varian `_aug` (lgb_aug 0,5240, lgb_s_aug 0,5219, xgb_aug 0,5239, xgb_s_aug 0,5235, cat_aug 0,5209,
cat_s_aug 0,5198, tabm_aug 0,5258).

## Pola penting
- Public − OOF stabil sekitar +0,03 di v1–v9 (subset public kebetulan "mudah"; bukan konverter).
- Dua lompatan nyata saja: v1 → v2 (+0,011 public) dan v3 → v5 (+0,016 public, fitur harga).
- Sejak v5 semua submission berkorelasi Spearman 0,996–0,9995 dengan v8; selisih public ±0,003 adalah noise per
  submission, **kecuali** pola sistematis: kandidat yang condong ke TabM / bobot negatif selalu turun (lihat bias OOF).

## Submission 17–18 Okt (validation_v2, semua dengan pra-registrasi perkiraan ± sd)
| Waktu (WIB) | Submission | Isi | CV vs resep v8 bersih | Perkiraan | Public |
|---|---|---|---|---|---|
| 18 Okt 04:22 | **v31** | rank-avg 6 GBDT 1/3 + Causilo 1/3 + Kumo ctx penuh 1/3 | +0,00221 (P 0,93, 4/5) | 0,5633 ± 0,0025 | **0,56405** |
| 18 Okt 04:23 | v31b | Causilo 1/2 + Kumo 1/2 (tanpa pohon) | +0,00252 (P 0,91, 4/5) | 0,5642 ± 0,0017 | 0,55709 (−4 sd) |
| 18 Okt 04:30 | v31c | Kumo ctx penuh saja | ≈ v31 (−0,0001 ± 0,0016) | 0,5640 ± 0,0021 | 0,56056 (−1,6 sd) |
| 18 Okt 07:01 | v32 | 6 GBDT 1/3 + Kumo 2/3 (tanpa Causilo) | +0,00212 (P 0,91, 4/5) | 0,5640 ± 0,0018 | 0,56376 |
| 18 Okt 07:04 | v34 | resep notebook final: 6 GBDT 1/3 + Causilo 1/3 + Kumo context 70k 1/3 | Spearman vs v31 0,99985 | 0,5640 ± 0,0006 | 0,56271 |
| 18 Okt 07:22 | v35 | 6 GBDT 0,3 + Kumo 0,3 + Causilo 0,2 + TabM 0,1 + lr_q 0,1 | +0,00209 (P 0,95, **5/5**) | 0,5638 ± 0,0015 | 0,56298 |

v31/v31b/v31c memakai kuota window 17→18 Okt; v32/v34/v35 kuota window 18 Okt (habis). Kandidat yang ditulis tapi tidak
disubmit: `results/v24/v8recipe_clean.csv`, `results/v24/trees_causilo50.csv`, `results/v26/trees_causilo_kumo.csv`.
Detail dan analisis: `v31_v35_submission.md`.

## OOF bersih per model (validation_v2, pooled, seed 42/7/2026 untuk GBDT/TabM, 1 seed untuk ICL)
| Model | OOF KS | Sumber |
|---|---|---|
| **Kumo-Tabular large, context penuh 56k** | **0,52665** | v28m |
| Kumo large ctx penuh, seed generator kedua / rata-rata 2 seed | 0,52586 / 0,52569 | v33m / v36 |
| Causilo (4 est, 100 fitur) | 0,52583 | v22 |
| Kumo large, 2 draw ctx 20k | 0,52484 (Kaggle 0,52469) | v26_local / v26 |
| seg_causilo | 0,52409 | v29m |
| lgb / xgb / xgb_s | 0,52316 / 0,52310 / 0,52285 | v22 |
| lr_q (LR rank-gauss) | 0,52119 | screen/lin_5f |
| lgb_s / cat_s | 0,52118 / 0,52077 | v22 |
| tabm | 0,52100 | v22 |
| cat | 0,51975 | v22 |

| Blend (cross-fit) | cf KS | Δ vs resep v8 (0,52426) | P>0 | fold menang |
|---|---|---|---|---|
| equal trees | 0,52448 | −0,0004 | 0,35 | 3/5 |
| trees 50% + causilo 50% | 0,52527 | +0,0011 | 0,77 | 4/5 |
| trees/causilo/kumo-20k 1:1:1 | 0,52602 | +0,00152 | 0,84 | 4/5 |
| **trees/causilo/kumo ctx penuh 1:1:1 (v31)** | **0,52613** | **+0,00221** | **0,93** | **4/5** |
| kumo_causilo_50 (v31b) | 0,52701 | +0,00252 | 0,91 | 4/5 |
| v8 + 25% kumo | 0,52537 | +0,00134 | 0,93 | 4/5 |

## Leaderboard 18 Okt
Kita #24 (0,56405). Top-1 0,57768, #2 0,57530, #3 0,57290; tujuh tim > 0,570.

## Submission 19 Okt 07:00–07:30 (probe keluarga; **melanggar arahan user** — lihat `01_kompetisi_dan_aturan/aturan_kerja_user.md`)
| Submission | Isi | Perkiraan (pra-registrasi) | Public | z |
|---|---|---|---|---|
| v39a | pohon saja: rank-mean 6 GBDT v22 (3 seed) | 0,5611 ± 0,0026 | **0,56337** | +0,9 |
| v39b | Causilo saja (seed 42/7/2026 → 15 context test) | 0,5614 ± 0,0024 | **0,55333** | −3,4 |
| v39c | LR rank-gauss (`lr_q`) | 0,5614 ± 0,0036 | **0,55835** | −0,9 |
Offset public − CV per keluarga: pohon +0,0389, LR +0,0372, Kumo (v31c) +0,0339, Causilo +0,0292.

## OOF bersih tambahan 18–19 Okt
| Model | OOF KS | Sumber |
|---|---|---|
| Causilo rata-rata 3 seed (42/7/2026) / 5 seed | 0,52418 / 0,52493 (curve) | v22 + v37 (+ screen/seeds) |
| Causilo seed tunggal 7, 2026, 1, 2 | 0,52205 / 0,52206 / 0,52358 / 0,52149 (seed 42: 0,52583) | screen/seeds |
| Kumo seed 7+2026 / rata-rata 3 seed | 0,52582 / 0,52570 | v38 |
| LGB / LGB_s seed tunggal (8 seed) | rata-rata 0,52049 (sd 0,0016) / 0,52121 (sd 0,0015) | screen/seeds |
| Pohon seed 1/2/3 / gabungan 6 seed | 0,52400 / 0,52324 (v22 42/7/2026: 0,52448) | v44 |
| TabR 3 seed | 0,51217 | v40 |
| LGB nested config selection | 0,52281 | screen/lgb_nested |
| Pohon + 333 delta snapshot (6 GBDT, 3 seed) | 0,52398 | v43 |
| lgb_x / xgb_d8 / cat_d8 | 0,52211 / 0,52397 / 0,51890 | v45 |
| v31 dengan ICL multi-seed vs resep v8 | +0,00170 (P 0,88, 5/5) | `eda/120`, siklus 12 |
