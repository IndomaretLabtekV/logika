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
