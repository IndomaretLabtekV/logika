# Semua model yang sudah dicoba

## GBDT (tulang punggung)
- **LightGBM** (CPU), **XGBoost** (hist, cuda), **CatBoost** (GPU). Masing-masing punya varian normal dan **shallow** (`_s`), 3 seed × 5 fold StratifiedKFold, early stopping AUC 200 ronde.
- OOF sekitar 0,521–0,525 sejak v5. Korelasi antar tree sangat tinggi.
- **Hyperparameter tidak pernah di-tune** (nilainya ada di `02_proyek_dan_tools/notebook_v9_dan_setup_kaggle.md`). Ini tuas yang masih terbuka.
- Seed tambahan tidak berguna (`eda/68`).

## TabM (NN terbaik)
- Library `tabm`. Input `nn_prep` (ECDF → normal + indikator missing), embedding PLE (`TABM_BINS` 48), dan varian periodic (`tabm_p`).
- Epoch 60, patience 8, batch 512, lr 2e-3, wd 3e-4.
- OOF sekitar 0,5248–0,5257, **model tunggal terbaik** di semua versi sejak v3. 5 seed di v9 (0,5253) tidak lebih baik dari 3 seed.
- Selalu mendapat bobot terbesar di blend.
- Hyperparameter TabM juga belum di-tune (k, d, depth, dropout, epoch).

## RealMLP-TD (pytabkit) — v9
- Input median + flag missing per fitur + kategori, early stopping cross-entropy. 2 seed, sekitar 36 menit per seed.
- OOF **0,5223**. Tidak menambah apa-apa di blend maupun stacking.

## Foundation model tabular
| Model | Hasil | Catatan |
|---|---|---|
| **TabPFN-3.5** (Prior-Labs, gated HF) | v1 0,4956 (ctx 30k / 60 fitur) → v3 0,4995 (ctx penuh 56k, 100 fitur, `fit_with_cache`) → v6+ **0,5186** (fitur manual termasuk pp) | 4 estimator, fold dibagi ke 2 GPU, sekitar 14 menit. Korelasi tinggi dengan Causilo (0,99). Bobot di blend kecil dan tidak stabil |
| TabPFN di DFS buatan sendiri (v5) | 0,4912 | DFS tidak punya fitur harga |
| **TabPFN-Rel native** (PriorLabs/tabpfn-rel, relarena_core DFS depth 3) | v7 **0,4917** | butuh Python 3.12 (venv uv terpisah); kategori dan lineage hilang di pipeline kita (catatan Astra) |
| **Causilo** (nums-ai) | v1 0,4953, v2 0,4848 → v7 **0,5226** → v8/v9 **0,524** (pandas categorical) | dipasang `--no-deps` (deklarasinya torch ≥ 2.13, tapi jalan di 2.11); sekitar 17 menit |
| **LimiX-2** (stable-ai, model China) | **hang lebih dari 3 jam** di fold 0 (v3), dibuang | butuh flash-attn (tidak didukung T4) / Python 3.12; fallback SDPA OOM |
| **EXAONE-Tabular** (LG AI) | v5 "No available kernel" (FS internal); v6 probe 3k **0,524** (42 detik), di-skip oleh budget; v7 probe 4k 0,5437 vs LGB 0,5467 dan 12k 0,5120 vs LGB 0,5189 (query beda), lalu **crash di 56k** ("cannot fit", reserved 25,9 GB > 13 GB); v8 tidak sempat jalan; v9 dibuang | maks 100 kolom (lebih dari itu memicu FS berbasis attention); `max_vram_bytes`; support encoded per predict, biaya sekitar ens × (ctx² + query × ctx); flash hanya di Ampere+. **Belum pernah punya OOF lengkap** |
| **TabICLv2** (soda-inria) | v8: OOM di semua context 60k → 7,5k, kernel mati | `whole_query=True` mengirim sekitar 44k query sekaligus. Yang menentukan memori adalah query, bukan context. Belum dicoba dengan query di-chunk |
| Mitra-v2 (autogluon) | tidak dicoba | 75,7M parameter; resepnya fine-tuning 50 step + bagging 8 fold, jadi bukan drop-in murah |
| RT-PluRel | tidak dicoba | sistem relasional (RelArena), butuh adapter task dan cutoff |

## Kesimpulan model
- Semua model "melihat" informasi yang sama. Korelasi rank OOF antar model 0,97–0,997; TabPFN dan Causilo 0,990.
- Model tunggal terbaik (TabM) sekitar 0,525, dan blend terbaik sekitar 0,528.
- Foundation model tidak mengalahkan GBDT/TabM di data ini, bahkan dengan konteks penuh.

## Tambahan 16 Okt (protokol bersih: early stopping di inner split)
| Model | OOF KS bersih | Catatan |
|---|---|---|
| LightGBM param v8 (v10 control) | 0,5191 | 1 seed; v21 base 2 seed 0,5214; v19 base 0,5200 |
| LightGBM tuned per fold (v10) | 0,5214 | konfigurasi tidak stabil antar fold |
| TabM config v8 (v11 control) | 0,5175 | epoch terbaik ~6, overfit dini |
| TabM tuned (v11) / kapasitas (v12) | 0,5199 / 0,5192 | |
| EBM aditif / 20 interaksi (v13) | 0,5172 / 0,5185 | |
| Set-encoder hybrid / events (v14) | 0,5187 / 0,5107 | events korelasi 0,95 dengan TabM |
| LGB rank_xendcg / XGB pairwise (v15) | 0,5178 / 0,5001 | |
| L2 LGB di atas logit v8 (v15) | 0,5248 / 0,5229 | input OOF v8 yang bias |
| TabICLv2 ctx 40k, 4 est / ctx penuh, 8 est, 150 fitur (v16/v17) | 0,5079 / 0,5133 | query di-chunk 2.000; ~16 menit / ~1 jam per fold di T4 |
| LGB linear_tree / monotone / ExtraTrees (v18) | 0,5197 / 0,5177 / 0,5001 | |
| LGB pseudo-label (v19) | 0,5197–0,5208 | |
| LGB + mature-history / pruned (v21) | 0,5203 / 0,5191 | |

Catatan penting: angka di atas **tidak bisa** dibandingkan langsung dengan tabel v1–v9 (yang bias). Contoh: TabM v8
0,5257 vs TabM bersih 0,5175. Di protokol bersih, LGB (~0,520) justru **setara atau lebih baik** dari TabM (~0,518–0,520),
kebalikan dari kesan v3–v9 bahwa TabM model tunggal terbaik.
