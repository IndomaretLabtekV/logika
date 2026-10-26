# Pembersihan repo 25 Okt 00:05 WIB

## Sudah dihapus (sampah, tidak ada informasi yang hilang)
- `eda/__pycache__/` (cache Python, dibuat ulang otomatis).
- 16 file **0 byte**: log guard kosong `eda/out/{220,237,240,241,242,244,245,246,248,249}_guard.log`; log percobaan kosong
  `results/v56/validation_logs/{features,model}_s7_f{1,2}_attempt02.log`, `results/screen/zoo_run3.log`; OOF rusak
  `results/v66/teammate_select/oof_lgb_tsel30.parquet` (crash hang #4; log fold tetap ada).
- 14 skrip antrean sekali pakai `eda/run_220_recipe.sh`, `run_220{b..g}_queue.sh`, `run_229.sh`,
  `run_230_main150.sh`, `run_231.sh`, `run_237.sh`, `run_237b.sh`, `run_240b.sh`, `run_244_seed2028.sh` — isinya
  diarsipkan di `skrip_antrean_dihapus.md`.

## Sengaja dipertahankan
- Semua kode `eda/*.py`, data (`data/`, `data_pq/`), hasil eksperimen, log dan telemetri (bukti angka).
- Folder karantina crash (`results/eda_23okt/row_vip/quarantine_crash_2026-10-23/`,
  `results/v60/strict_tfm/quarantine_crash_2026-10-24/`) sebagai bukti insiden (berisi 3 file 0 byte).
- `results/v56/teammate_features/outer_s42_f*` — **dibutuhkan untuk rencana merge** (fold sama dengan user).

## Kandidat hapus opsional (perlu keputusan user; belum dihapus)
| Path | Ukuran | Catatan |
|---|---:|---|
| `results/v56/teammate_features/outer_s7_f{0,1,2}`, `outer_s2026_f{0,1}` | ±7 GB | build fitur teman seed outer lain (v56 tak selesai); bisa dibangun ulang tapi makan waktu |
| `results/v50/teammate_features/` | 1,4 GB | versi lebih lama build fitur teman (v50 input families) |
| `results/v4/`, `v6–v9/`, `screen/` | ±360 MB | OOF bias era v1–v9 (tidak dipakai sebagai pembanding) |
| `notebook/build_notebook_v*.py`, `build_v10–v18.py` | kecil | builder notebook lama (resep lama diarsip di `notebook/archive/`) |
| `eda/out/*.log` lama (<200) | ±260 MB folder | log eksperimen lama; berguna untuk audit |

## Integrasi folder `dsc-logika-ui-2026-main/` (25 Okt 00:40 WIB)
- 608 file (346 MB) diinventaris dengan SHA256 (`docs/dsc0152/inventaris_awal.csv`), dipindah (mv) ke:
  `scripts/dsc0152/` (kode, experiments, work, shell, agas, dimaz), `notebook/dsc0152/` (notebook deliverable, runs/,
  dimaz_test.ipynb), `results/dsc0152/` (output agas ±325 MB, harness_output, dimaz model_output, logs),
  `docs/dsc0152/` (PROGRESS.md, next_submission.txt, kamus data Dimaz, .gitignore asli).
- Symlink: `scripts/dsc0152/agas/v*/output` dan `agas/shared/harness_output` → `results/dsc0152/agas/...`;
  `scripts/dsc0152/data` → `../../data` (agar `paths.py` ROOT tetap valid).
- Verifikasi: **592 file identik SHA256, 16 `.DS_Store` dihapus, 0 masalah** (`docs/dsc0152/pemetaan_file.csv`).
  Folder `dsc-logika-ui-2026-main/` sudah kosong dan dihapus. Tidak ada kredensial di skrip.
- Skrip shell sumber memakai path macOS (`/Users/yutaaa/...`) → arsip. `variants/*.csv`, cache pkl, `run_*/` output
  tidak ikut di upload.

## Status git
- Perubahan sesi ini **belum di-commit** (README, indeks, jurnal, status, validation_v2, features.py, build_final.py,
  final.ipynb, script 212–249, protocol 24 Okt, folder ini). `results/`, data, log dikecualikan `.gitignore`.
- Jangan `git push` (larangan user).
