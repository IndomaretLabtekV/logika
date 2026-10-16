# Blending dan stacking

## Metode yang pernah dipakai
| Metode | Versi | Catatan |
|---|---|---|
| Equal rank average | v1 | 0,5051; FM yang lebih lemah ikut menurunkan |
| Simplex grid + cross-check separuh | v1–v2 | dipakai hanya kalau gain > 0,002 |
| Greedy (Caruana) + gate held-out | v3–v6 | in-sample terlihat bagus (v6 0,5283), held-out cuma 0,5263; bobot tidak stabil antar separuh |
| 80% trees + 20% TabM tetap | v6+ (kandidat), safety-net | held-out 0,5247–0,5260; hampir sama dengan greedy |
| Bagged greedy (20 bag) | v7–v8 | held-out 0,5268 (v7), 0,5277 (v8) |
| **Logit stacking** (logistic regression di logit prediksi, Newton numpy, L2 = 1) | v9 | held-out terbaik: 0,5287 (9 model v8), 0,5272 (11 model v9) |

## Temuan
- **Pilih blend berdasarkan held-out halves.** Bagi OOF jadi 2, fit di satu separuh, nilai di separuh lainnya, lalu sebaliknya. Angka in-sample selalu terlalu optimis sekitar +0,001–0,002.
- Logit stacking menang tipis karena risiko aditif di log-odds dan model-model mengukur faktor laten yang sama dengan noise berbeda (`03_data_dan_fitur/temuan_eda_utama.md`). Gain-nya kecil: +0,0008 sampai +0,0017.
- Menambah kolom yang sangat berkorelasi (tabm_p, realmlp, model `_aug`) bisa **menurunkan** stacking (v9 11 model −0,0019 vs 9 model). Koefisien negatif muncul karena kolinearitas. L2 yang lebih besar atau seleksi kolom bisa dicoba, tapi gain-nya tetap kecil.
- Rekalibrasi per segmen (history, bulan, rate, purpose, employment) selalu merugikan (−0,0005 sampai −0,0026). Stacking × interaksi history: −0,0007.
- Skala agregasi (80/20 tetap): prob mean 0,5250, rank 0,5246, logit 0,5245. Selisihnya noise.
- Rank per fold sebelum blend: merugikan (Astra).
- Band di sekitar cut KS: semua model AUC sekitar 0,53 di band ±5%. Reorder khusus band −0,003.

## File
- `results/v8/submission_v9_stack.csv`: stacking 9 model v8 (lgb, lgb_s, xgb, xgb_s, cat, cat_s, tabm, tabpfn, causilo).
- Notebook v9 sel blend: fungsi `fit_stack`. Kandidat disimpan sebagai `(held score, (oof, test))`.
- Script lokal pembanding v8 vs v9 dibuat di scratchpad sesi lama (kemungkinan sudah hilang). Logikanya: `logreg` Newton seperti di `eda/96_segment_calibration.py`, dengan 20 pengulangan split separuh.

## Revisi 16 Okt (setelah probe public)
- **Stack logit-LR kalah di public dua kali**: v9 (11 model) 0,55917 dan S3 (9 model + TabICL17) 0,55778, vs blend
  rank v8 0,56145. Keduanya punya bobot negatif. Kesimpulan "logit stacking menang tipis" di atas **hanya berlaku di OOF
  yang bias**; jangan dipakai sebagai submission tanpa validasi bersih.
- Blend v8 yang disubmit = rank-average greedy (bagged) dengan bobot positif semua, 46,7% GBDT / 53,3% NN/FM.
- Dekomposisi di public: bagian GBDT saja 0,56133, bagian NN/FM saja 0,55866. Porsi TabM di blend v8 kemungkinan terlalu
  besar karena OOF TabM paling menggelembung.
- Aturan baru: pilih bobot di **OOF bersih**, prefer rank-average bobot positif, nilai di held-out halves **dan** bootstrap
  baris; curigai setiap gain yang datang dari bobot negatif.
- `results/v8/submission_v9_stack.csv` (stack LR 9 model v8) tidak pernah disubmit; dengan bukti di atas, kemungkinan
  besar juga di bawah v8 di public.
- Tool lokal: `02_proyek_dan_tools/tools_bench_lokal.md`.
