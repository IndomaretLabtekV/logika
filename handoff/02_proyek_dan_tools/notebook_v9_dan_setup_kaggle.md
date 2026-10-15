# Notebook v9 (versi aktif) dan setup Kaggle

## Konvensi skill `ref/notebook/`
- Banner 3 sel (judul, tim, daftar isi), lalu **Introduction** (Overview, Aim, Metric, Dataset, Approach), lalu **Initialization** (Environment Setup dengan `!nvidia-smi` dan `%pip`, Import Libraries, Seed Everything, Settings, Load Dataset).
- Import hanya di sel Libraries. Tidak ada docstring atau komentar pembatas di sel kode.
- Ada paragraf penghubung antar sel, dan blockquote "#### Insights" setelah setiap analisis.
- Builder (`build_notebook.py`) memakai helper `C()`/`M()`. Sel fitur diambil dari `eda/features.py` (docstring dan import dibuang lewat regex), sel nn_prep diambil dari `eda/nn_prep.py`.

## Isi v9 (109 sel)
1. **Introduction**
2. **Initialization**:
   - install `polars==2.0.0`, lightgbm, xgboost, catboost, tabpfn, tabm, pytabkit (RealMLP), dan causilo (`--no-deps`);
   - pembacaan secret;
   - Settings (lihat bawah).
3. **EDA**:
   - utilitas KS;
   - target per waktu;
   - rate tier (KS 0.23);
   - burst belanja sebelum aplikasi (rasio bad/good 2.34× di minggu terakhir);
   - keparahan keterlambatan;
   - price premium (KS 0.368);
   - cakupan tabel;
   - tabel "Offline Audit of Versions 6 to 8".
4. **Feature Engineering**:
   - `features.py` di-embed, `build_all` (100.000 × 237, sekitar 13 detik);
   - leak test 2 bagian (injeksi baris masa depan, dan baris setelah cutoff Desember tidak menggeser fitur, termasuk referensi harga);
   - tabel KS/PSI fitur;
   - "Signal Structure Within Classes" (korelasi di dalam kelas).
5. **Validation Strategy**: StratifiedKFold 5 fold per seed, `report()` (OOF KS, AUC, fold KS, simulasi gap), dan penyimpanan npy.
6. **GBDT**: lgb, lgb_s, xgb, xgb_s, cat, cat_s × seed [42, 7, 2026], plus ablation v3 vs v3+v5.
7. **NN dan FM**:
   - TabM 5 seed;
   - **safety-net submission** (80% trees + 20% TabM);
   - tabm_p (TabM periodic) 3 seed;
   - RealMLP-TD 2 seed;
   - TabPFN-3.5 (konteks 56k, 100 fitur + 4 kategori, `fit_with_cache`, fold dibagi ke 2 GPU);
   - Causilo (kategori pandas bertipe).
8. **Ensemble**:
   - lima kandidat dinilai di *held-out halves*: equal trees, 80/20 trees+TabM, greedy, bagged greedy (20 bag), dan **logit stacking** (Newton numpy, L2 = 1);
   - yang terbaik dipilih;
   - cek stabilitas: fold KS, simulasi gap 30% dan 50%, thin-file vs history, KS per bulan, plot CDF.
9. **Submission**: assert format, tulis `submission.csv` dan `submission_gbdt_only.csv`, simpan `oof_all.parquet` / `test_all.parquet` beserta ID.

## Settings penting
```
SEED=42  GBDT_SEEDS=[42,7,2026]  N_FOLDS=5
RUN_LIMIT_MIN=12*60-45  BLEND_RESERVE_MIN=20
FM_N_FEATURES=100  FM_MAX_CONTEXT=60000  FM_N_EST=4  FM_CHUNK=10000
TABM_SEEDS=[42,7,2026,11,99]  TABM_EPOCHS=60  TABM_PATIENCE=8  TABM_BATCH=512  TABM_BINS=48  TABM_LR=2e-3  TABM_WD=3e-4
REALMLP_SEEDS=[42,7]  ENS_ITERS=30  ENS_BAGS=20
LGB: lr .02, leaves 31, min_child 100, ff .5, bagging .8, l2 5   | LGB_SHALLOW: leaves 7, depth 3, min_child 200, ff .4, l2 10
XGB: hist cuda, depth 5, lr .02, mcw 5, subsample .8, colsample .5, lambda 5 | XGB_SHALLOW: depth 3, mcw 20, colsample .4, lambda 10
CAT: GPU, depth 6, lr .03, l2 5, od_wait 200 | CAT_SHALLOW: depth 4, l2 10
N_ROUNDS=10000  EARLY_STOP=200  (metric early stopping: AUC)
```
**Catatan:** hyperparameter GBDT ini praktis tidak berubah sejak v1 dan **tidak pernah di-tune**.

## Setup Kaggle untuk menjalankan notebook
1. Upload `notebook/v9.ipynb`, atau notebook baru dari builder.
2. Add Data: data kompetisi `data-science-competition-logika-ui-2026`.
3. Accelerator **GPU T4 x2**, Internet **On**.
4. Secrets: `HF_TOKEN` (akses TabPFN-3.5 gated, lisensi sudah di-accept) dan opsional `TABPFN_TOKEN`.
5. **Save Version → Save & Run All (Commit).** Sesi interaktif bisa mati kalau ditinggal.
6. Setelah selesai, unduh seluruh `/kaggle/working` ke `results/vN/` dan notebook hasil run ke `notebook/vN.ipynb`.
7. Nama notebook resmi: `ParticipantNumber_TeamName_Notebook`. Isi juga banner tim di builder.
