# Kode validation_v2 dan tool sesi 17–18 Okt (cara pakai)

Semua script dijalankan dari `eda/` dengan `../.venv/bin/python` (kecuali disebut `.venv-zoo`). Protokolnya dijelaskan di
`07_validasi_dan_leaderboard/validation_v2.md`; file ini fokus ke cara memakai.

## Modul inti
| File | Isi |
|---|---|
| `eda/valid2.py` | Protokol: `frames(features)` (X/Xt pandas dengan kategori dari train, y, ID), `outer(y, seed)` (StratifiedKFold 5), `inner(y_tr, seed, fold)` (split 1/7 untuk stopping), `fm_select(X, y, tr, n_num)` (100 numerik teratas menurut KS train fold + filter \|Spearman\| < 0,95 + 4 kategori), fungsi fit per model (`lgb/xgb/cat/tabm/causilo_fit/tabpfn_fit/tabdpt_fit/kumo_fit/lr_q_fit/woe_lr`), `seg_fit(base)` (fit model per segmen has_history), `run()` (loop seed × fold, tulis `oof_<model>.parquet`, `test_<model>.parquet`, `log_<model>.csv`, `manifest.json` setelah **setiap** fold) |
| `eda/run_valid2.py` | Driver. Lokal: `python run_valid2.py OUT_DIR lgb,tabm --folds 0 --seeds 42`. Full: tanpa `--folds/--seeds` (5 fold × 3 seed). Opsi `--extra file.parquet` (kolom tambahan dengan application_id), `--drop`, `--deadline` |
| `eda/features.py` | Sumber fitur tunggal (`build_all`, 232 prediktor + ID). Cache: `data_pq/features.parquet` lewat `common.features()` |
| `eda/nn_prep.py` | Input TabM / `lr_q` (ECDF → normal per fold, indikator pola missing) |

### Kunci nama model di `run_valid2.py`
- GBDT dan NN dari `MODELS`: `lgb`, `lgb_s`, `xgb`, `xgb_s`, `cat`, `cat_s`, `tabm`, `lr_q`, `woe_lr`, `realmlp`.
- ICL dengan sufiks parameter: `causilo`, `tabpfn`, `tabdpt`, `kumo`. Token sufiks: `e`=n_est, `f`=n_num (jumlah fitur),
  `c`=max_context, `x`=context, `n`=n_ens, `b`=n_bag (jumlah draw context yang dirata-rata), `g`=g_off (seed generator
  estimator lain, context sama). Kumo menerima ukuran `small|medium|large` sebagai token pertama.
  Contoh: `causilo` (4 est, 100 fitur), `causilo_e16_f150`, `kumo_large_c20000_b2`, `kumo_large_c56000` (context penuh),
  `kumo_large_c56000_g1`.
- Prefix `seg_` → `seg_fit` (contoh `seg_causilo`, fit terpisah thin-file vs history; null di v29).

## Blend dan kandidat
| Script | Fungsi | Contoh |
|---|---|---|
| `eda/blend2.py` | Blend cross-fit di OOF bersih: bobot dipilih di 4 fold (penugasan fold seed 42), dinilai di fold ke-5; pembanding = resep v8 (bobot v8 direkonstruksi, `_aug` dilipat ke model dasar, TabPFN dibuang) di model bersih; paired bootstrap selisih pooled KS + fold menang. Aturan tetap: equal trees, trees_causilo_50, greedy_bagged, causilo saja; kalau ada kolom `kumo*`: kumo_causilo_50, trees_causilo_kumo (1/3 per keluarga), v8_causilo_to_kumo, v8_plus_kumo_25 | `python blend2.py ../results/v22/v22 ../results/v28m > out/blend2_v28.txt` |
| `eda/write_candidate.py` | Tulis CSV rank-average dari test bersih + validasi format (30.000 baris, ID = sample_submission, [0,1]) | `python write_candidate.py ../results/vNN/nama.csv lgb=1/18 ... causilo=1/3 kumo_large_c56000=1/3 --dirs ../results/v22/v22 ../results/v28m` |
| `eda/merge_runs.py` | Gabung run parsial satu model (fold disjoint, seed sama): OOF ambil non-NaN, test rata-rata berbobot jumlah fold, log digabung; assert tidak overlap | `python merge_runs.py ../results/v28m kumo_large_c56000 ../results/v28_local_f2 ../results/v28_local` |
| `eda/107_public_noise.py` | Distribusi selisih KS kandidat vs resep v8 di subset acak 30% (perkiraan ± sd sebelum submit) | lihat docstring |

## Screening fold 0
| Script | Fungsi |
|---|---|
| `eda/screen2.py` | Kandidat (OOF fold 0) vs resep v8 di fold 0: KS sendiri dan +20% bobot tetap ke referensi (trees 50% + causilo 50%), bootstrap 1.000×. Referensi di `results/screen/{ref_f0,causilo_f0}` |
| `eda/zoo_screen.py` | Zoo TabArena lewat AutoGluon di `.venv-zoo` (tuning_data = inner 1/7, default TabArena, tanpa bagging). Output `results/screen/zoo_f0/` |
| `eda/sdm_screen.py` | NVIDIA structured-data-models (KumoTabular small/medium/large, TabFM, TabICLv2) di fold 0; `--ctx`, `--est`, `--chunk`, `--fp32` (wajib untuk TabFM: fp16 → NaN) |
| `eda/kumo_fulltrain.py` | Kumo large, seluruh 70k train sebagai context, prediksi 30k test sekali (opsi notebook final). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--chunk 500` default. T4: 16.172 s, peak 6,5 GB |
| `eda/kumo_finetune.py` | Fine-tune Kumo di fold 0 (episode context 2048 + query 512 dari inner-fit, BCE, AdamW 1e-5, early stop di inner-stop). **Ada bug: KS baseline 0,298**; lihat `09_langkah_berikutnya/rekomendasi.md` |

## Pola output `results/`
- Run validation_v2: `results/<run>/{oof,test,log}_<model>.parquet|csv` + `manifest.json` (versi paket + config).
  OOF bernilai NaN di fold yang tidak dijalankan; test = rata-rata model fold yang dijalankan.
- Run Kaggle: `results/vNN/vNN/<subdir>/...`, log kernel `results/vNN/<slug>.log`, salinan kode yang dipakai saat run
  `results/vNN/code/` (provenance; kode di `eda/` bisa sudah berubah).
- Gabungan run parsial diberi akhiran `m` (`v28m`, `v29m`, `v33m`).

## Environment
- `.venv` (Python 3.13): polars 2.0.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, torch 2.14.1 (CUDA 13),
  tabm 0.0.3, causilo 1.0.3, scikit-learn 1.9.1, structured-data-models 0.2.1. Semua run validation_v2 lokal, termasuk Kumo.
- `.venv-zoo`: autogluon.tabular 1.6.3, tabarena 0.1.0, tabpfn 9.0.0, tabicl 2.2.0, pytabkit 1.7.3,
  structured-data-models 0.2.1, torch 2.9.1. Hanya untuk `zoo_screen.py` (dan screening sdm awal). 11 GB, di-.gitignore;
  boleh dihapus kalau zoo tidak dipakai lagi.
- Kaggle (manifest v22/final): numpy 2.1.3, pandas 2.3.3, polars 2.0.0, scikit-learn 1.6.1, scipy 1.16.3,
  lightgbm 4.6.0, xgboost 3.4.1, catboost 1.2.10, torch 2.11.0+cu128, tabm 0.0.3, rtdl_num_embeddings 0.0.12, causilo 1.0.3.

## Tambahan 18–19 Okt (semua di `eda/valid2.py` / `eda/run_valid2.py`; jalur default resep tidak berubah)
| Fitur | Cara pakai | Catatan |
|---|---|---|
| `run(..., test=False)` / `run_valid2.py --no-test` | screening tanpa prediksi 30k test | hemat ~2/3 waktu Kumo/ICL |
| `stack_fit(base)` / prefix `stk_` | ICL diberi kolom `gb_stk` = skor LGB OOF bersarang (query = rata-rata 5 model inner) | null (siklus 4) |
| `stack_lr_fit(base)` / prefix `slr_` | kolom `lr_stk` = logit lr_q bersarang | null (siklus 25) |
| `pca_fit(base, k=20)` / prefix `pca_` | 20 komponen PCA rank-gauss (fit di train fold) ditambahkan | null |
| `fm_select(..., rand=k)` / token `r<k>` | feature bagging: 20 fitur KS teratas + 80 acak dari yang lolos filter Spearman | null (siklus 8) |
| `fm_select(..., rand=-k)` / token `r-1` | 100 fitur menurut **gain LGB** (fit di train fold) | Causilo null (siklus 16) |
| `realmlp_fit(model="TabR_S_D_Classifier")` / model `tabr` | pytabkit TabR; kategori di-one-hot; metrik val default; index faiss GPU diganti `_torch_faiss_gpu` (cdist+topk) | butuh `skorch` (dipasang di `.venv`); null (siklus 15) |
| `lgb_nested_fit(LGB_GRID)` / model `lgb_nested` | 10 konfigurasi LGB, pilih per outer fold dari KS inner-stop | null (siklus 21) |
| `LGB_X`, `XGB_D8`, `CAT_D8` / model `lgb_x`, `xgb_d8`, `cat_d8` | varian pohon (extra_trees, depth 8) | null (siklus 34) |
| `kumo_fulltrain.py --g_off K` | Kumo-70k dengan draw estimator lain (context sama); output `test_kumo_full70k_g<K>.parquet` | `results/v41` |
| `kumo_finetune.py` (diperbaiki) | `p1(out)` memilih kolom kelas "1" per nama; baseline di-cache `eda/out/kumo_finetune_f0_base.parquet` | fine-tune null |

### Evaluator baru (pakai ini, jangan menulis ulang)
| Script | Fungsi |
|---|---|
| `eda/screen2.py` | fold 0: kini juga vs v31 fold 0 (kandidat menggantikan Kumo, atau ditambah 1/4) |
| `eda/120_swap_eval.py FAM w:path …` | ganti satu keluarga v31 dengan OOF lain (multi-seed dirata-rata probabilitas), vs v31 dan resep v8 |
| `eda/124_add_component.py path` | tambah keluarga ke v31 dengan bobot 0,1/0,2/0,25 (versi ICL seed 42 dan multi-seed) |
| `eda/116_fbag_eval.py`, `eda/119_distill_eval.py`, `eda/143_snapd_eval.py` | evaluasi bag / distilasi / delta di level keluarga pohon |
| `eda/121_trees_only_estimate.py`, `eda/131_public_family.py` | perkiraan public probe dan rangkuman offset per keluarga |
| `eda/129_lowvar_candidate.py OUT.csv` | kandidat v31 varian rendah (pohon + Causilo multi-seed + Kumo-70k multi-draw) |
| `eda/139_cutoff_audit.py` | audit baris setelah cutoff train vs test |

### Prinsip evaluasi yang ditambahkan 19 Okt
- **Bandingkan di level keluarga**, bukan anggota: gain anggota (LGB +0,002–0,004) hilang di rank-mean 6 GBDT.
- **Rata-rata seed outer** sebelum percaya selisih < ~0,002 (sd seed ±0,0015; bootstrap baris tidak menangkapnya).
- **Seleksi bersarang** untuk fitur yang dipilih dari banyak kandidat memakai label (`eda/146` sebagai pola).
