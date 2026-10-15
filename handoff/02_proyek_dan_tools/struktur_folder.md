# Struktur proyek `/home/rian/Projects/logika`

```
data/                 CSV asli dari Kaggle dan DATA_DICTIONARY.md (jangan diubah)
data_pq/              cache parquet: tabel mentah, features.parquet (matriks fitur 100k baris),
                      dfs.parquet (DFS buatan sendiri), rel_dfs_d2/d3.parquet (DFS native TabPFN-Rel),
                      hist_labels.parquet / hist_feats.parquet (augmentasi historis v8),
                      v1_train_order.parquet (urutan ID untuk OOF v1)
docs/context.md       deskripsi, dataset, dan rules kompetisi
ref/notebook/         SKILL pembuatan notebook (SKILL.md, references/, build_template.py, template.ipynb)
eda/                  script EDA bernomor 00–96 dan modul inti (lihat bawah)
eda/out/              output semua script: *.txt, *.csv, *.png
eda/deep/             audit Astra #1 (13 Okt), README.md + RESULTS.md
eda/v5_review/ … eda/v8_review/   audit Astra per versi, masing-masing README.md + RESULTS.md + out/
notebook/             builder notebook dan notebook hasil run (lihat bawah)
results/v1 … v9/      output /kaggle/working tiap run: oof_*.npy, test_*.npy, submission.csv,
                      submission_gbdt_only.csv, figures/, (v7+) oof_all.parquet / test_all.parquet + train_ids.npy / test_ids.npy
results/v8/submission_v9_stack.csv   logit stacking lokal 9 model v8 (lihat 06_model_dan_blending)
results/v8/v8_run.ipynb              notebook v8 yang benar-benar di-run (crash di TabICL)
submissions/00_scorecard_baseline.csv   scorecard tanpa model (KS CV sekitar 0.41), tidak pernah di-submit
log/                  log run v3 (logika.log dan download.txt) yang hang di LimiX
handoff/              folder ini

# dari repo teman (Nicholas, 10–12 Okt; baseline awal sebelum v1 kita)
docs/data_guide.md, docs/reports/          panduan dataset, EDA_RESEARCH_REPORT.md, RISK_EXPLORATION_REPORT.md
eda/baseline_duckdb/                       EDA dan 63 fitur SQL DuckDB awal (+ requirements.txt)
eda/risk_exploration/ + outputs/04_risk_exploration/   risk exploration lanjutan (124 kandidat fitur, deskriptif)
notebook/0-initial-iteration/              3 notebook GBDT CPU (pooled OOF KS LGB 0,4817 / XGB 0,4790 / Cat 0,4790)
research/                                  indeks paper metode (PDF di-.gitignore, unduh ulang dengan download_papers.py)
```
Catatan: `RISK_EXPLORATION_REPORT.md` menyebut share cicilan pernah ≥30 DPD (hanya cicilan dengan due ≥30 hari) KS 0,435. Angka itu dihitung pada support observed saja (customer berhistori), jadi tidak sebanding dengan KS kita yang memperlakukan NaN sebagai bin.

## Modul inti di `eda/`
| File | Fungsi |
|---|---|
| `features.py` | **Sumber kebenaran tunggal fitur.** Di-embed verbatim ke notebook oleh builder. Isinya `app_features`, `bureau_features`, `repayment_features`, `account_features`, `ecommerce_features`, `prev_features`, `v2_features`, `v5_features`, dan `build_all`. Total 237 kolom (232 fitur + ID), 4 di antaranya kategorikal. |
| `nn_prep.py` | Input untuk TabM: ECDF mid-rank → normal (fit hanya di train fold), clip ±5, NaN → 0, satu indikator missing untuk setiap pola missing yang unik. Diverifikasi di `eda/32`. |
| `hist_snapshots.py` | Label pinjaman historis dan snapshot fitur per tanggal T (v8). **Tidak dipakai lagi di v9.** |
| `dfs.py` | DFS buatan sendiri (resep TabPFN-Rel), dipakai v5. Kalah dari fitur manual. |
| `common.py` | Path, `load()` (cache parquet), `ks`, `ks_nan` (NaN di ujung), `psi`, `features()` (cache features.parquet), `analyze`, `calibrate`, `qbin`, `resid_test` (chi2/df + dKS cross-fit terhadap OOF terkalibrasi), `screen`, `v1_oof`, `v2_oof`, `v5_oof`, `v6_oof` |
| `common9.py` | `v8_ref()` (OOF blend v8 terkalibrasi dari `eda/v8_review/out/00_bound_oof.parquet`), `safe_bins` (nol struktural dan missing dipisah), `oe_test`, `crossfit_dks` |

## Notebook
| File | Keterangan |
|---|---|
| `notebook/build_notebook.py` | Builder aktif, menulis `notebook/v9.ipynb` dengan output ke `output_v9/` (lokal) atau `/kaggle/working` |
| `build_notebook_v6.py`, `_v7.py`, `_v7_vast.py`, `_v8.py` | Backup builder versi lama |
| `dryrun_stub.py` | Dry run lokal dengan semua model di-stub |
| `v1.ipynb … v9.ipynb` | Notebook hasil run Kaggle (sumber yang dikembalikan user). `v7.ipynb` adalah output run, **jangan ditimpa** builder |
