# Struktur proyek `/home/veinsan/Projects/logika` (per 18 Okt 2026)

```
data/                 CSV asli + DATA_DICTIONARY.md + sample_submission.csv (di-.gitignore; unduh ulang lewat Kaggle CLI)
data_pq/              cache parquet (di-.gitignore): tabel mentah + features.parquet (dibangun common.features())
docs/context.md       deskripsi, dataset, dan rules kompetisi (salinan halaman Kaggle)
docs/data_guide.md, docs/reports/   dari repo teman (baseline awal)
ref/notebook/         SKILL pembuatan notebook (SKILL.md, references/, build_template.py, template.ipynb)
research/             indeks paper (PDF di-.gitignore, unduh ulang dengan download_papers.py)
eda/                  script EDA bernomor 00–111 + modul inti + tool (lihat bawah)
eda/out/              output EDA (*.txt, *.csv, *.png; parquet di-.gitignore)
eda/deep/, eda/v5_review/ … eda/v8_review/   audit GPT-6 Astra
eda/baseline_duckdb/, eda/risk_exploration/  dari repo teman
notebook/             builder + notebook (lihat bawah)
results/              output run (lihat bawah)
submissions/00_scorecard_baseline.csv   scorecard tanpa model (KS CV ~0,41), tidak pernah disubmit
submissions/probe/    probe 16 Okt: S3/S4/S5, README.md, expectations.csv
outputs/04_risk_exploration/   dari repo teman
log/logika.log        log run v3 yang hang di LimiX
handoff/              folder ini
.venv/                environment utama (di-.gitignore)
.venv-zoo/            AutoGluon/TabArena (11 GB, di-.gitignore; boleh dihapus kalau zoo tidak dipakai lagi)
```

## Modul dan tool di `eda/`
| File | Fungsi |
|---|---|
| `features.py` | **Sumber kebenaran tunggal fitur** (`build_all`, 232 prediktor + ID, 4 kategorikal) |
| `nn_prep.py` | Input TabM / LR rank-gauss: ECDF mid-rank → normal (fit per fold), clip ±5, NaN → 0, indikator pola missing |
| `common.py` | `load`, `ks`, `ks_nan`, `psi`, `features()` (cache), `OUT`, loader lama (⚠ `qbin` bug nol/positif) |
| `common9.py` | `v8_ref`, `safe_bins`, `oe_test`, `crossfit_dks` (era v9) |
| `valid2.py`, `run_valid2.py` | **validation_v2**: protokol + driver (lokal dan Kaggle) |
| `blend2.py`, `write_candidate.py`, `merge_runs.py` | blend cross-fit, tulis kandidat CSV, gabung run parsial |
| `screen2.py`, `zoo_screen.py`, `sdm_screen.py` | screening fold 0 (pembanding, zoo AutoGluon, NVIDIA sdm/Kumo/TabFM) |
| `kumo_fulltrain.py` | Kumo context 70k untuk notebook final |
| `kumo_finetune.py` | fine-tune Kumo fold 0 (**bug belum ketemu**) |
| `100_*.py` … `111_*.py` | EDA sesi 17–18 Okt (katalog: `05_katalog_eksperimen/katalog_eda.md`) |
| `hist_snapshots.py`, `dfs.py` | augmentasi historis v8 / DFS v5 (tidak dipakai lagi) |
| `s01`–`s04_*.py`, `review_remote_tuning.py` | bench era 16 Okt di atas OOF v8 bias (usang) |

Cara pakai rinci: `validation_v2_dan_tool.md`.

## Notebook
| File | Keterangan |
|---|---|
| `notebook/build_final.py` → `final.ipynb` | **Notebook final** (lihat `notebook_final.md`) |
| `build_v22.py` … `build_v33.py` + `vNN.ipynb` | Kernel validation_v2 17–18 Okt (menyematkan file `eda/`). v25 dibangun dengan builder v23 (zoo ronde 2). `v28.ipynb` dibuat tapi kernel-nya dihapus |
| `build_v10.py … build_v21.py` + `vNN_experiment.py` + `vNN.ipynb` | Eksperimen terisolasi 16 Okt |
| `build_notebook.py` (+ `_v6/_v7/_v7_vast/_v8`), `dryrun_stub.py` | Pipeline penuh v1–v9 (⚠ early stopping di fold validasi) |
| `v1.ipynb … v9.ipynb` | Notebook hasil run Kaggle era lama |
| `0-initial-iteration/` | 3 notebook GBDT CPU dari repo teman |

## `results/`
| Folder | Isi |
|---|---|
| `v1` … `v9` | output pipeline lama (OOF/test npy, `oof_all.parquet` sejak v7, submission) |
| `v10` … `v21`, `v14k` … `v21k` | eksperimen 16 Okt + metadata kernel |
| `v22/v22/{gbdt,nn,fm}` | **OOF/test bersih resep v8** (6 GBDT × 3 seed, TabM 3 seed, Causilo) + manifest + log; `v22/code/` kode saat run |
| `v23`, `v25` | zoo Kaggle ronde 1/2 (log + OOF iLTM / Mitra) |
| `v24/` | kandidat lokal dari v22: `v8recipe_clean.csv`, `trees_causilo50.csv` (tidak disubmit) |
| `v26/`, `v26_local/` | Kumo 2 draw ctx 20k: Kaggle dan lokal (reproducibility); `v26/trees_causilo_kumo.csv` |
| `v27/` | TabFM fold 0 (log) |
| `v28_local_f2/`, `v28_local/` → `v28m/` | Kumo context penuh 5 fold (lokal), gabungan di `v28m` |
| `v29/` → `v29m/` | seg_causilo (Kaggle) |
| `v30/v30/test_kumo_full70k.parquet` | Kumo context 70k untuk test |
| `v31/` | `submission.csv` (public 0,56405), `kumo_causilo50.csv` (v31b), `kumo_only.csv` (v31c) |
| `v32/trees_kumo_1to2.csv` | public 0,56376 |
| `v33_local/` + `v33/` → `v33m/` | Kumo ctx penuh seed generator kedua |
| `v34/` | `submission.csv` resep final (public 0,56271) + `oof/test_kumo_full70k.parquet` |
| `v35/five_family.csv` | public 0,56298 |
| `v36/` | rata-rata Kumo 2 seed (≈ e8) |
| `screen/` | screening fold 0 (`ref_f0`, `causilo_f0`, `causilo_x_f0`, `icl_f0`, `kumo_f0`, `kumo_full_f0`, `lin_f0`, `lin_5f`, `nn_f0`, `sdm_f0`, `zoo_f0`) |
| `final_kaggle/` | output run Kaggle notebook final (`submission.csv` = v34, `run/{gbdt,fm}`, log, `pipeline/` kode saat run) |
| `v22k` … `v33k`, `finalk` | `kernel-metadata.json` untuk push (notebook staging dihapus; buat ulang dengan builder) |

## Yang dihapus saat bersih-bersih 18 Okt
- Semua salinan `features.parquet` di `results/*/` (9 × 64 MB; turunan data lomba, bisa dibangun ulang, tidak boleh
  didistribusikan) dan `eda/out/100_external_credit.parquet` (6,7 MB, turunan).
- Notebook staging `results/v22k … v33k, finalk/*.ipynb` (identik byte dengan `notebook/vNN.ipynb` / `final.ipynb`).
- `__pycache__/` dan `catboost_info/` di `eda/` dan `results/*/code/`.
- Log run lokal dipindah ke dalam foldernya: `results/v26_local/run.log`, `v28_local/run.log`, `v28_local/run_b.log`,
  `v33_local/run.log`.
- `.gitignore` ditambah: `.venv-zoo/`, `eda/out/*.parquet`, `results/**/features.parquet`.

## Yang sudah dihapus saat bersih-bersih 16 Okt
Salinan notebook di `results/vNNk/*.ipynb`, `results/v15ds/`, `results/stack/`, file per-fold `*fold*.npz` v10–v17,
`results/v14/v14/submission_*.csv`, `notebook/check_v10.py`, `log/download.txt`, `submissions/probe/S1/S2`.
