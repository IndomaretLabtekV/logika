# Ringkasan skor semua versi

| Versi | Tanggal run | Perubahan utama | OOF / held-out | Public | Status |
|---|---|---|---|---|---|
| v1 | 13 Okt | baseline: 176 fitur, LGB/XGB/Cat + TabPFN + Causilo, equal rank blend | 0,5051 | **0,53139** | disubmit |
| v2 | 13 Okt | +38 fitur residual (bz/ob/rq/sv/cf/ec2) + varian shallow trees | 0,5135 (greedy in-sample) | **0,54192** | disubmit |
| v3 | 13 Okt | TabPFN konteks penuh, LimiX-2 (hang, lalu dibuang), **TabM** | 0,5142 | **0,54545** | disubmit |
| v4 | 13–14 Okt | dibuat **Astra**: ablation spend 112–224 hari / semantic rate, 2 seed, tanpa FM | 0,5114 | tidak tercatat ("cukup jelek") | bahan analisis |
| v5 | 14 Okt | **price premium `pp_*`** + dec_month (+0,014). TabPFN di DFS. EXAONE gagal | 0,5274 | **0,56145** ← terbaik | disubmit |
| v6 | 14 Okt | referensi harga aman-waktu, nn_prep per fold, TabPFN di fitur manual, tabm_p | 0,5283 in-sample / 0,5263 held-out | **0,56123** | disubmit |
| v7 | 14–15 Okt | Kaggle T4x2: Causilo jalan, TabPFN-Rel native, EXAONE crash, bagged greedy | 0,5270 / 0,5268 held-out | — | tidak disubmit (Spearman 0,9995 vs v6) |
| v8 | 15 Okt | augmentasi 81.702 pinjaman historis, TabICL (crash ±4 jam), blend lokal | 0,5270 (bobot aktual); bagged greedy held-out 0,5277 | sekitar 0,561 ("sama aja") | disubmit |
| v8-stack | 15 Okt | logit stacking lokal 9 model v8 (`results/v8/submission_v9_stack.csv`) | 0,5277 cross-fit; held-out halves 0,5287 | belum tercatat | disarankan submit |
| v9 | 15 Okt | tanpa histori/TabICL/EXAONE, TabM 5 seed, RealMLP, logit stacking | 0,5261 in-sample / 0,5272 held-out | belum tercatat | disarankan submit supaya bisa dipilih (dihasilkan notebook utuh) |

## Top-1 public (laporan user)
0,57059 (13 Okt) → 0,57184 (14 Okt) → 0,57458 (14 Okt malam) → 0,57493 (15 Okt).

## Pola penting
- **Public − OOF stabil sekitar +0,026 sampai +0,034** di semua versi. Lihat `07_validasi_dan_leaderboard/`.
- Dua lompatan nyata: v1 → v2 (fitur perilaku, +0,011 public) dan v3 → v5 (fitur harga, +0,016 public). Selain itu semua perubahan berada di dalam noise.
- Selisih public antar versi yang lebih kecil dari ±0,005 adalah noise. Sd selisih public v6−v5 sekitar 0,0026.
- Korelasi Spearman submission antar versi sejak v5: 0,998–0,9995. Prediksinya praktis sama.

## KS per model (OOF, versi terakhir yang menjalankannya)
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
| realmlp | — | — | — | (tidak jalan) | 0,5223 |
| tabpfn (fitur manual) | — | 0,5186 | 0,5189 | 0,5186 | 0,5187 |
| tabpfn di DFS / TabPFN-Rel | 0,4912 | — | 0,4917 | — | — |
| causilo | — | — | 0,5226 | 0,5241 | 0,5239 |
| EXAONE | gagal | probe 0,524 (3k) | crash 56k | tidak jalan | dibuang |

(Angka v5/v6 trees dari review Astra; nilai yang tidak tercatat dibiarkan kosong.)
