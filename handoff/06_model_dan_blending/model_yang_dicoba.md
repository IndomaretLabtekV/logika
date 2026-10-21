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

## Tambahan 17–18 Okt (validation_v2: stop di inner 1/7, refit budget tetap, bobot cross-fit)
| Model | OOF KS bersih | Catatan |
|---|---|---|
| **Kumo-Tabular large, context penuh 56k** (NVIDIA `structured-data-models`) | **0,52665** | model tunggal terbaik; 55–65 menit/fold lokal, ~2,3 jam/fold T4 |
| Kumo large, 2 draw context 20k | 0,52484 | varians draw context menekan ~0,002 |
| Kumo large, context 70k (seluruh train), test sekali | – (tanpa OOF) | Spearman 0,99869 vs rata-rata 5 fold; 4,5 jam T4, 6,5 GB |
| Causilo (4 est, 100 fitur, pandas categorical) | 0,52583 | model terbaik di v22; 115–145 s/fold lokal |
| seg_causilo (per segmen has_history) | 0,52409 | null |
| lgb / xgb / xgb_s / lgb_s / cat_s / cat (param v8, 3 seed) | 0,52316 / 0,52310 / 0,52285 / 0,52118 / 0,52077 / 0,51975 | hyperparameter v8, tidak di-tune (tuning v10 null) |
| TabM (config v8, 3 seed, epoch dipilih di inner lalu refit) | 0,52100 | terlemah; epoch terbaik 5–6 |
| LR rank-gauss + one-hot (`lr_q`) | 0,52119 | ≈ TabM → risiko hampir linear |

Zoo fold 0 (detail `05_katalog_eksperimen/screening_lokal_17okt.md`): TabPFN-3.5 0,5175, TabICLv2 0,5163, RealMLP 0,5157,
iLTM 0,5216, TabFM 0,5173, AutoGluon CatBoost 0,5237, Mitra 0,4488; LimiX-2/TabDPT/Nori OOM atau gagal. Hanya Kumo dan
Causilo yang menambah.

### Kesimpulan model (revisi 18 Okt)
- Di protokol bersih urutannya: **Kumo ctx penuh > Causilo > LGB/XGB > TabM ≈ LR**. Kesan v3–v9 (TabM terbaik) adalah
  artefak bias OOF.
- Keluarga ICL (Kumo, Causilo) menambah informasi yang tidak ditangkap pohon (+0,002 CV, +0,0026 public di v31), tapi
  **hanya kalau dicampur pohon** (lihat `blending_dan_stacking.md`).
- Korelasi rank: Kumo–Causilo 0,989, Kumo–trees ~0,978, antar pohon 0,97–0,999, TabFM vs Kumo 0,977.
- Kapasitas tambahan (fitur lebih banyak, model lebih besar, estimator lebih banyak) tidak membantu: Kumo f154 < f100,
  medium < large, e8 ≈ e4 dalam noise; Causilo f150/f300 ≤ f100.

## Tambahan 18 Okt sore – 19 Okt (validation_v2)
| Model / varian | OOF / hasil | Keterangan |
|---|---|---|
| Kumo fine-tune blok ICL (203M param), lr 1e-5 / 1e-4 | inner-stop KS tidak naik / turun | null; semua parameter OOM di 8 GB (AdamW) |
| Stacked ICL (Causilo/Kumo + skor LGB bersarang) | Causilo 0,5224 vs 0,5249 (fold 0); Kumo 0,5258 vs 0,5256 | null |
| Distilasi Causilo → LGB (target lunak α 0,3/0,1) | 0,52343 / 0,52239 (5 fold) | null di keluarga |
| Causilo feature bagging / seleksi gain | bag 0,52345 / gain 0,52390 | null |
| TabR (pytabkit, retrieval NN) | 0,51217 (3 seed) | terlemah; menambah ke v31 merugikan |
| LGB nested config selection (10 config) | 0,52281 | null |
| LGB extra_trees / XGB depth 8 / Cat depth 8 | 0,52211 / 0,52397 / 0,51890 | keluarga 9 null |
| LGB custom objective surrogate-KS | −0,004 (fold 0) | null |
| LGB spesialis pita titik potong | −0,0008…−0,0053 | null |
| KumoRelational | tidak dijalankan | butuh embed 4,36 juta item, tidak muat 8 GB |
| VIP-COP context optimisation untuk Kumo | belum (dibatalkan) | lihat `10_ide_belum_dicoba/` |

### Kesimpulan model (revisi 19 Okt)
- **Di public**: pohon (0,56337) > Kumo (0,56056) > LR (0,55835) > Causilo (0,55333), terbalik dari CV
  (Kumo 0,5267 > Causilo 0,5258 > pohon 0,5245 > LR 0,5212). CV hanya bisa dipercaya untuk pohon dan LR.
- Pohon jenuh: fitur, hyperparameter, seed, keragaman, agregasi tidak menaikkan keluarga 6 GBDT.
- Info luar: tim 0,57065 memakai ensemble Kumo 3 seed → kemungkinan setup Kumo kita (fitur `fm_select` 100 numerik
  top-KS + 4 kategori sebagai kode, context train fold) jauh dari optimal untuk public.

21Okt, klarifikasi audit gabungan GNN: baris KumoRelational tidak dijalankan
di atas berlaku untuk repo utama. Snapshot teman menyimpan percobaan
KumoRelational ber-GNN: preflight/parity, kendala memori, lalu readouthemat
dengan1/5fold tersimpan. Tidak ada pooled5fold×3seed yang terbukti;status
training pada metadata19Okt adalah historis, bukan proses aktif saat ini.
Bukti:artifacts/optimization/goal054/relational_queue_status.json dan
relational_initial_oom.json. GNN mandiriGCN/GraphSAGE belum ditemukan.
