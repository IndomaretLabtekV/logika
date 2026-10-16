# Status semua rekomendasi lama (dari handoff 15 Okt dan review Astra)

| # | Rekomendasi | Status per 16 Okt | Hasil |
|---|---|---|---|
| A1 | Tuning hyperparameter GBDT | ✅ v10 (LGB, 12 konfigurasi, nested) | +0,0022 tapi 3/5 fold, tidak dipromosikan |
| A2 | Tuning TabM | ✅ v11 (optimizer) + v12 (kapasitas) | +0,0024 (3/5 fold) / +0,0005; null |
| A3 | 9 fitur selected-history Astra | ✅ v21 + cek LR lokal | −0,0011 OOF LGB; −0,0007 di stack |
| A4 | Rasio DPD exposure-matang, delta skor provider sama | ❌ tidak diuji di model | screening 0/15; skala provider simetris (cek 16 Okt); ekspektasi ≈ 0 |
| A5 | eb_k1 / eb_k30 | ❌ tidak diuji | chi2 ~2, lemah; ekspektasi ≈ 0 |
| A6 | Seleksi / pruning fitur | ✅ v21 | −0,0023 OOF LGB |
| B1 | Objective ranking | ✅ v15 (LGB rank_xendcg, XGB pairwise) | 0,5178 / 0,5001; null |
| B2 | Kebijakan checkpoint (AUC vs logloss vs KS) | ⚠ tidak diuji langsung, tapi ditemukan masalah yang lebih besar: checkpoint dipilih di fold validasi (bias OOF) | lihat `07_validasi_dan_leaderboard/bias_oof_early_stopping.md` |
| B3 | Stacking lebih rapi (L2, nonnegatif, seleksi) | ✅ sebagian: L2 LGB stacker (v15), bobot optimasi-KS, stack per segmen | semua null; stack LR bobot negatif kalah di public (v9, S3) |
| B4 | Pseudo-labeling | ✅ v19 | null |
| C1 | EXAONE 8–12k dengan query di-chunk | ❌ tidak dicoba | probe lama 12k 0,512 vs LGB 0,519; ekspektasi rendah |
| C2 | TabICLv2 dengan query di-chunk | ✅ v16 (ctx 40k) + v17 (ctx penuh) | 0,5079 / 0,5133; gain stack hanya lewat bobot negatif, public −0,0037 (S3) |
| C3 | Xiaomi-TabLDM / TabFM / Mitra | ❌ tidak dicek | – |
| Astra | DFS native dengan lineage | ❌ | – |
| Astra | Simpan bobot, hash, versi per run | ✅ untuk v10–v13 (manifest.json); v14–v21 menyimpan OOF/test dengan ID | – |
| baru | Model baris mentah (set-encoder) | ✅ v14 | 0,5187 / 0,5107; null |
| baru | Bias inductive lain (linear_tree, monotone, ExtraTrees), EBM | ✅ v13, v18 | null |
| baru | Causilo/TabPFN 16 estimator | ❌ v20 dibuat, tidak jalan (kuota GPU) | – |
