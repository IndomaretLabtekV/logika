# Eksperimen 3 — Teknik dari write-up juara dan paper (24 Okt siang)

Basis semua uji LGB: v22 params (`valid2.LGB`, 4 thread), 232 fitur main (`data_pq/features.parquet`), outer seed 42,
inner 1/7 untuk budget, refit ×1,15; pembanding `lgb_base4` pooled **0,52074**; evaluator `eda/238_series_oof_eval.py`
(fitur baru dari `results/v66/series_oof/<tabel>_f<k>.parquet`, base dihitung ulang dengan thread sama, bootstrap 500).

## 3.1 LightGBM DART (kode juara 1 AmEx, jxzly)
- Param: boosting dart, num_leaves 64, lr 0,035, feature_fraction **0,05**, bagging 0,75/5, min_data_in_leaf 256,
  max_bin 63, l1 0,1, l2 30; ronde dipilih di inner-stop tiap 250 (≤4.500). `eda/236_dart_screen.py`.
- Fold 0: **0,51276 vs 0,51892 (−0,0062)** → null (ff 0,05 cocok untuk 1.000+ fitur AmEx, bukan 232).

## 3.2 Series OOF / distant supervision per baris event (AmEx & Home Credit juara 1)
- `eda/237_series_oof.py`: setiap baris riwayat diberi label customer; LGB baris (bobot 1/n per customer), GroupKFold
  per customer **di dalam** outer-train untuk OOF baris; model seluruh outer-train memprediksi baris outer-val/test;
  agregasi per customer (mean, max, p90, std, n, mean90, recency-weighted).
- ecom (4,36 juta baris) fold 0: 0,51860 (−0,0003). **cicilan fold 0: 0,52441 (+0,0055)** → konfirmasi 5-fold:
  **0,51940 vs 0,52074 (−0,0013, CI −0,0040…+0,0027, P 0,32)**; per fold +0,0055/−0,0046/−0,0003/+0,0003/−0,0027.
  **Gain fold 0 adalah kebetulan** (sd satu fold ~0,008) → null.

## 3.3 KNN target mean (Home Credit juara 1: "neighbors_target_mean_500")
- `eda/239_knn_target_mean.py`: nested per outer fold — top-5 fitur numerik menurut KS univariat di outer-train (terpilih:
  pp_mean, pp_trim, pp_done, pp_90, rp_frac_dpd1), rank-transform, rata-rata target k=100/500 tetangga (OOF 5-fold di
  dalam outer-train). Fold 0: **0,51495 (−0,0040)** → null (fitur dasar redundan).

## 3.4 CoLES — lihat `02_semi_supervised.md` (null).

## 3.5 GRU-NN (NN juara 1 AmEx)
- `eda/242_gru_series_nn.py`: GRU di urutan ecom + cicilan (encoder dari 240) + MLP 232 fitur (quantile-normal fit
  outer-train + mask missing), BCE, early stopping inner 1/7, tanpa refit.
- NN sendiri pooled **0,51409** (AUC 0,8377; fold 0,5111/0,5156/0,5165/0,5121/0,5376).
- Blend terkunci 0,75 LGB + 0,25 NN: **0,52266 vs LGB 0,52074 (+0,0019, CI −0,0007…+0,0043, P 0,93, 3/5)** — positif
  di atas LGB tunggal.
- Di atas blend terbaik (`eda/243_gru_blend_check.py`): B1 = 0,8 REF + 0,2 NN 0,52577 vs REF 0,52625 (−0,0005);
  **B2 = 0,8 K64 2-seed + 0,2 NN 0,52730 vs 0,52740 (−0,0001)** → keragaman GRU sudah tertangkap Kumo/Causilo → null.

## 3.6 Kumo Relational (diperiksa, tidak dijalankan)
- sdm 0.2.1 punya `KumoRelational` (bobot `nvidia/Kumo-Relational` v1.0.1, classifier 120 MB, tidak di-gate, belum
  diunduh). Backbone = **TabICLv2** (row embedding + ICL block), keluarga yang di data ini hanya 0,508–0,513; teman
  sempat menjalankan fold 0 di GPU 24 GB (OOM awal 10,35 GiB) tanpa skor tercatat. Prior rendah + ongkos tinggi.

## Sumber literatur
- Kode juara 1 AmEx: https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution (S3 series LGB,
  S5 LGB DART, S6 GRU, S7 ensemble 0,3/0,35/0,15/0,1).
- Ringkasan juara 1 Home Credit (fitur neighbor target mean EXT_SOURCE + credit/annuity): Medium "Home Credit Default
  Risk — An End to End ML Case Study".
- CoLES: https://arxiv.org/abs/2002.08232 (tabel 6, 10, 11).
