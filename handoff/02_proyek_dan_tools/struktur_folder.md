# Struktur proyek `/home/rian/Projects/logika` (per 16 Okt 2026)

```
data/                 CSV asli + DATA_DICTIONARY.md + sample_submission.csv (di-.gitignore, jangan diubah)
data_pq/              cache parquet (di-.gitignore): tabel mentah, features.parquet (matriks fitur 100k, ada has_history),
                      dfs.parquet, rel_dfs_d2/d3.parquet, hist_labels.parquet / hist_feats.parquet (augmentasi v8),
                      v1_train_order.parquet
docs/context.md       deskripsi, dataset, dan rules kompetisi
docs/data_guide.md, docs/reports/   dari repo teman (baseline awal)
ref/notebook/         SKILL pembuatan notebook (SKILL.md, references/, build_template.py, template.ipynb)
research/             indeks paper (PDF di-.gitignore)
eda/                  script EDA bernomor 00–96 + modul inti (lihat bawah) + tool bench s01–s04
eda/out/              output script EDA (*.txt, *.csv, *.png)
eda/deep/, eda/v5_review/ … eda/v8_review/   audit GPT-6 Astra (README.md + RESULTS.md + out/)
eda/baseline_duckdb/, eda/risk_exploration/  dari repo teman
notebook/             builder + notebook (lihat bawah)
results/v1 … v9/      output /kaggle/working pipeline penuh: oof_*.npy, test_*.npy, train_ids.npy/test_ids.npy (v7+),
                      submission.csv, submission_gbdt_only.csv, figures/
results/v8/           + submission_v9_stack.csv (stack LR 9 model, tidak pernah disubmit) + v8_run.ipynb (notebook yang di-run)
results/v10 … v21/    output eksperimen 16 Okt: vNN/vNN/oof.parquet (+ test.parquet, atau submission_*.csv untuk v10–v13
                      sebagai prediksi test), metrics.json, log; vNN/<slug>.log = log kernel
results/v14k … v21k/  kernel-metadata.json untuk push CLI (notebook staging sudah dihapus; buat ulang dengan builder)
submissions/00_scorecard_baseline.csv   scorecard tanpa model (KS CV ~0,41), tidak pernah disubmit
submissions/probe/    probe 16 Okt: S3/S4/S5 (yang disubmit), README.md, expectations.csv
outputs/04_risk_exploration/   dari repo teman
log/logika.log        log run v3 yang hang di LimiX
handoff/              folder ini
```

## Modul inti di `eda/`
| File | Fungsi |
|---|---|
| `features.py` | **Sumber kebenaran tunggal fitur** (`build_all`, 232 fitur + ID, 4 kategorikal). Di-embed ke notebook v9 oleh `build_notebook.py`, dan ke v10–v21 lewat sel 12 `notebook/v10.ipynb` |
| `nn_prep.py` | Input TabM: ECDF mid-rank → normal (fit di train fold), clip ±5, NaN → 0, indikator per pola missing |
| `hist_snapshots.py` | Label pinjaman historis dan snapshot fitur (v8). Tidak dipakai sejak v9 |
| `dfs.py` | DFS buatan sendiri (v5), kalah dari fitur manual |
| `common.py` | `load`, `ks`, `ks_nan`, `psi`, `features`, `resid_test`, `screen`, loader OOF v1–v6 (⚠ `qbin` punya bug nol/positif) |
| `common9.py` | `v8_ref`, `safe_bins` (bin aman-nol), `oe_test`, `crossfit_dks` |
| `s01`–`s04_*.py`, `review_remote_tuning.py` | tool bench prediksi tersimpan, lihat `tools_bench_lokal.md` |

## Notebook
| File | Keterangan |
|---|---|
| `notebook/build_notebook.py` | Builder pipeline penuh v9 (⚠ early stopping di fold validasi) |
| `build_notebook_v6.py`, `_v7.py`, `_v7_vast.py`, `_v8.py` | Builder versi lama |
| `dryrun_stub.py` | Dry run lokal pipeline v9 dengan semua model di-stub |
| `v1.ipynb … v9.ipynb` | Notebook hasil run Kaggle (`v7.ipynb` = output run, jangan ditimpa) |
| `build_v10.py … build_v21.py` + `vNN_experiment.py` + `vNN.ipynb` | Eksperimen terisolasi 16 Okt. Pola: sel 12 v10 (fitur) + `vNN_experiment.py` + `main_vNN(build_all)`. v12 = v11 dengan grid diubah (tanpa experiment.py sendiri). v20 belum pernah jalan |
| `0-initial-iteration/` | 3 notebook GBDT CPU dari repo teman |

## Yang sudah dihapus saat bersih-bersih 16 Okt (semua redundan / bisa dibuat ulang)
- Salinan notebook di `results/vNNk/*.ipynb` (sama dengan `notebook/vNN.ipynb`).
- `results/v15ds/` (staging dataset Kaggle `logika-v8-preds`; isinya prediksi v8 yang sudah ada di `results/v8/`).
- `results/stack/` (sub_v8_tabicl16.csv null; sub_v8_tabicl17.csv identik dengan probe S3).
- File per-fold `*fold*.npz` di `results/v10–v17` (isinya sudah ada di oof.parquet/test.parquet/submission_*.csv).
- `results/v14/v14/submission_*.csv` (ada test.parquet).
- `notebook/check_v10.py` (pemeriksa mock sekali pakai untuk v10).
- `log/download.txt` (identik dengan `log/logika.log`).
- `submissions/probe/S1_gbdt_only.csv`, `S2_nn_fm_only.csv` (tidak disubmit, diganti S4/S5).
