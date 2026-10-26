# Kode sumber DSC0152 ("kali ini beneran deh") — diintegrasikan 25 Okt 2026

Sumber ketiga repository (selain pipeline utama di `eda/` dan pipeline teman di `scripts/*.py`). Diunggah user sebagai
folder `dsc-logika-ui-2026-main/` lalu dipecah ke struktur repo. Peta file lengkap + checksum:
`docs/dsc0152/pemetaan_file.csv` (592 file identik SHA256, 16 `.DS_Store` dihapus). Ringkasan riset sumber ini:
`docs/dsc0152/README.md`.

## Struktur di sini
| Path | Isi |
|---|---|
| `features.py` | pipeline fitur DSC0152 (termasuk factor model `factor_inputs`, `factor_features`); membaca `data/` relatif CWD |
| `make_nb.py` | generator notebook: `python make_nb.py <out.ipynb> <N_FOLDS> <TREE_SEEDS> <FOLD_SEEDS> <LR>` (final: `10 1 42,7,13 0.01`) |
| `bag.py` | rata-rata blend beberapa run seed (`variants/bag_*.csv`) |
| `experiments/` | ±100 skrip eksperimen + log (harvest/planner public, luck search, factor, ec_deep, team_stack, v61_plan, …) |
| `work/` | `common.py`, `v61_recipe.py` |
| `shell/` | 31 skrip zsh (antrean submit terjadwal, training pass, battery guard) — **memakai path macOS** `/Users/yutaaa/Developments/dsc-logikaui`, arsip saja, tidak jalan di laptop ini |
| `agas/` | pipeline Agas per versi `v57…v71/{training,notebook,output}` + `shared/` (paths, feat_extra, build_extra, cv_harness, factor2, bootstrap) + dokumen (`README.md`, `progress.md` jurnal lengkap, `plan.md`, `limit.md`, `rules.md`, `overview.md`) + `OtherTries/` |
| `agas/v*/output`, `agas/shared/harness_output` | **symlink** ke `results/dsc0152/agas/...` (prediksi `preds_*.npz`, summary JSON, log; ±325 MB) |
| `dimaz/train_default_model.py` | model awal Dimaz (output di `results/dsc0152/dimaz/model_output/`, notebook di `notebook/dsc0152/dimaz_test.ipynb`) |
| `data` | **symlink** `../../data` supaya `features.py`/`agas/shared/paths.py` (ROOT = folder ini) menemukan dataset |

Notebook: `notebook/dsc0152/` (`DSC0152_kali ini beneran deh_Notebook.ipynb` deliverable lama 5 fold; `runs/<run>/` notebook
per konfigurasi run: run1, run10, run_fac, run_seeds, run_s7/s13/s21, run_v3_*, run_v4_*, run_lr005, run_final).
Log root: `results/dsc0152/logs/`. Dokumen root (PROGRESS.md, next_submission.txt, .gitignore asli): `docs/dsc0152/`.

## Cara menjalankan
- Jalankan dari folder ini: `cd scripts/dsc0152 && ../../.venv/bin/python agas/v66/training/<script>.py` (paths.py
  menghitung `ROOT = scripts/dsc0152`). Skrip root (`features.py`, `make_nb.py`) memakai `data/` relatif CWD → juga dari
  folder ini.
- Yang tidak ikut di upload (dikecualikan `.gitignore` asli): `variants/*.csv`, `*.pkl` cache, `features.parquet`,
  `models/`, `run_*/` output OOF. Beberapa skrip (mis. `bag.py`, planner) merujuk `run_*/` dan `variants/` di root
  aslinya → perlu dibuat ulang atau path disesuaikan sebelum dijalankan.
- Prediksi OOF/test yang **ada**: `agas/v6x/output/preds_*.npz` berisi `train_ids`, `test_ids`, `oof_<leg>`,
  `test_<leg>` (leg: fac_full, lgb_full, lgb_lean, lgbx_full, cat_full, xgb_full, linear, …) — siap di-join by ID untuk
  blend lintas pipeline.
