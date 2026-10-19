# Cara kerja lokal dan jebakan teknis

> Bagian "Environment" dan "Dry run" di bawah berlaku untuk laptop lama (12–16 Okt, tanpa CUDA). Environment sekarang ada di
> bagian "Mesin baru 17 Okt" di akhir file dan di `validation_v2_dan_tool.md`.

## Environment
- `.venv`: Python 3.13, polars 2.0.0, numpy, scipy, dan matplotlib. **Tidak ada sklearn.** Untuk clustering pakai `scipy.cluster.vq.kmeans2`, untuk logistic regression pakai Newton numpy.
- `.venv-rel`: Python 3.12 berisi tabpfn-rel, relarena-core[dfs], dan relbench, **tanpa torch**. Hanya untuk featurizer DFS native (`eda/66a`).
- Jalankan script EDA dengan `cd eda && ../.venv/bin/python NN_nama.py`. Output masuk ke `eda/out/`.
- Clone repo eksternal (EXAONE-Tabular, tabpfn-rel) dulu ada di scratchpad sesi lama. Kemungkinan sudah hilang.

## Dry run notebook (wajib setiap kali notebook diubah)
```
cd notebook && ../.venv/bin/python build_notebook.py          # tulis v9.ipynb
../.venv/bin/python dryrun_stub.py v9.ipynb                    # semua model di-stub, KS yang tercetak tidak bermakna
rm -rf ../output_v9 ../output                                  # bersihkan
```
- Stub mencakup torch, lightgbm, xgboost, catboost, sklearn, tabpfn, tabm, exaone, Causilo, RealMLP, dan (opsional) TabICL. Stub juga menguji guard deadline dan `run_fm` paralel.
- Kalau menambah model baru, tambahkan juga stub-nya di `dryrun_stub.py`.

## Jebakan yang pernah kena
- **`ls` di shell user di-alias dengan `--icons`.** Pakai `command ls` atau `/bin/ls`.
- **Jangan `pkill -f` pola yang muncul di command line sendiri**, karena shell sendiri ikut terbunuh.
- Polars 2.0: `Expr.describe` tidak ada (pakai `Series.describe`); nama kolom `len` bentrok (pakai `n`); `qcut` deprecated.
- **Tie timestamp bikin sort nondeterministik.** Ada 170 order dan 94 laporan bureau dengan timestamp kembar, dan `due_date` kembar antar pinjaman paralel. Selalu sort dengan tiebreak ID: `("report_datetime","bureau_observation_id")`, `("application_datetime","application_id")`, `("due_date","loan_id")`.
- **Leak test butuh toleransi float**, dan harus menyertakan `previous_applications` (nama kolom `decision_datetime` bentrok).
- Stub torch merusak scipy kalau scipy di-import belakangan. Import `scipy.stats` sebelum stub dan sediakan `torch.Tensor`.
- File `.npy` di `results/v1`–`v6` **tidak membawa ID**. Urutannya sama dengan urutan train di notebook. v1 diselaraskan lewat `data_pq/v1_train_order.parquet`. Sejak v7 sudah ada `oof_all.parquet` dan `train_ids.npy`.
- `cudf-polars` di Kaggle mengeluh soal polars 2.0. Itu cuma warning karena kita tidak memakai cudf.
- Fold KS rata-rata **lebih tinggi** daripada pooled OOF KS (sekitar +0.003), karena KS dimaksimalkan per subset. Selalu laporkan **pooled OOF KS** sebagai angka utama.

## Fakta Kaggle (dari log run)
- Python 3.13.15, torch 2.11.0+cu128, 2× Tesla T4 15 GB, polars bisa di-pin ke 2.0.0.
- Sesi maksimal 12 jam. Notebook memakai `DEADLINE = 12h − 45 menit` dan `minutes_left()` yang dicek sebelum setiap fold.
- Secret yang dipakai: `HF_TOKEN` dan `TABPFN_TOKEN` (TabPFN-3.5 gated, lisensi harus di-accept di HF).
- Data ada di `/kaggle/input/competitions/data-science-competition-logika-ui-2026` dan dicari otomatis lewat `rglob('train_applications.csv')`.
- Waktu yang terukur di T4x2:

| Model | Waktu |
|---|---|
| Trees (6 varian × 3 seed × 5 fold) | sekitar 30–35 menit |
| TabM | sekitar 10 menit per seed |
| TabM periodic (tabm_p) | sekitar 17 menit per seed |
| RealMLP | sekitar 35 menit per seed |
| TabPFN-3.5 (konteks penuh 56k, 104 fitur, 2 GPU paralel) | sekitar 14 menit |
| Causilo | sekitar 17 menit |
| Total v9 | sekitar 4 jam |

## Jebakan tambahan 16 Okt
- `ls` di-alias (`--icons`, dan `ls data` bisa error "invalid value 'data' for '--icons'"). Pakai `command ls`.
- Python sistem (`python3`) tidak punya polars; selalu `.venv/bin/python`.
- Perintah > 2 menit di Bash pindah ke background; bootstrap KS ribuan kali bisa > 2 menit, beri timeout besar atau
  kurangi iterasi.
- Builder notebook (`notebook/build_vNN.py`) hanya `compile()` sel; aman dijalankan lokal karena tidak melatih apa pun.
- Kredensial Kaggle CLI kedaluwarsa diam-diam (lihat `kaggle_cli_kernel_dan_submit.md`).
- File di scratchpad sesi (`/tmp/claude-…/scratchpad`) hilang antar sesi; simpan apa pun yang penting di repo.

## Mesin baru 17 Okt
- CachyOS, i7-12700H (20 thread), RAM 32 GB, RTX 3070 Ti Laptop 8 GB. Repo `/home/veinsan/Projects/logika`.
  Shell user **fish** (bukan bash/zsh).
- `.venv` (Python 3.13) sekarang lengkap: torch CUDA, lightgbm, xgboost, catboost, tabm, causilo, sklearn,
  structured-data-models. `.venv-zoo` untuk AutoGluon/TabArena. Versi: `validation_v2_dan_tool.md`.
- Data diunduh ulang lewat Kaggle CLI ke `data/`; cache `data_pq/` dibangun ulang otomatis oleh `common.features()`.
- Model boleh dilatih lokal (screening dan full fold), VRAM < 8 GB, satu job berat sekaligus.

## Jebakan baru 17–18 Okt
- **Laptop hang** (17 Okt ~15:00): TabICLv2 (AutoGluon) jatuh ke CPU (~500% CPU) bersamaan ablation LR yang memakai semua
  thread. Aturan: satu job berat sekaligus, `OMP_NUM_THREADS=4` (maks 8), `nice -n 5` untuk job panjang.
- **`pgrep -f` / `pkill -f` mencocokkan baris perintah sendiri.** Loop `until ! pgrep -f "<pola>"` tidak pernah selesai
  (diagnostik `eda/106` tertunda ~1 jam), dan `pkill -f` pernah membunuh shell sendiri (exit 144). Ambil PID dengan
  `ps -eo pid,args | grep "[p]ola"` lalu `kill <PID>`.
- **Word-splitting argumen dalam loop**: di zsh/fish `"$a"` berisi beberapa argumen tidak dipecah seperti bash. Untuk loop
  screening pakai script bash terpisah.
- `statsmodels` tidak terpasang → log-likelihood LR pakai `sklearn.linear_model.LogisticRegression(C=1e6)` (lihat `eda/111`).
- **Run parsial menimpa output.** `run_valid2.py` dengan `--folds` menulis file bernama sama; dulu fold lain bisa hilang.
  Sekarang `valid2.run` menyimpan setelah setiap fold, tapi tetap pakai direktori output berbeda per run parsial lalu
  gabung dengan `merge_runs.py`.
- **OOM Kumo context besar**: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` dan chunk query 500–1000.
  Fragmentasi (7,3 GB reserved-unallocated) adalah penyebab OOM v30 v1.
- **TabFM di fp16 → prediksi NaN**; wajib `--fp32`. Context 20k OOM di T4 (12,4 GB).
- **Estimasi waktu Kumo lokal**: ~55–65 menit per fold context penuh (termasuk 30k test), bukan 18 menit (angka 18 menit
  dari screening tanpa test).
- Python sistem tidak punya polars; `python3 -I` tidak melihat `.venv`. Selalu `.venv/bin/python`.

## Jebakan baru 18–19 Okt
- **Laptop hang lagi (18 Okt ~21:50)** saat TabR (pytorch-lightning) fold 2; tidak ada OOM di journal, hanya "system too
  slow". Aturan: job berat `taskset -c 0-11 nice -n 10`, OMP/MKL 8. Setelah itu tidak hang lagi.
- **Kumo `out.numerical[..., 1]` bukan P(kelas 1)**: urutan kolom kelas berubah per panggilan (ecoc). Selalu pilih per nama
  (`to_pandas()["1"]` atau `out.columns[sdm.Stype.numerical].index("1")`).
- **pytabkit TabR**: encoder kategori gagal di sklearn 1.9 bila grup kosong → one-hot sendiri; `1-auc_ovr` tidak
  didukung; faiss CPU-only → stand-in index torch; butuh `skorch`. Checkpoint lightning menumpuk di `eda/checkpoints/`
  (dihapus, sekarang di-.gitignore).
- **Snapshot cutoff mundur** (`build_all` dengan decision_datetime −k hari) kehilangan kolom pivot `bp_*` (laporan bureau
  aplikasi terpilih belum ada) → lewati kolom yang tidak ada.
- **Notebook lokal**: `OUTPUT_DIR` relatif + subprocess `cwd=CODE_DIR` → path ganda (diperbaiki `.resolve()`); `%pip` versi
  pin akan men-downgrade `.venv` (diganti instalasi hanya bila `/kaggle/input` ada); < 2 GPU → rantai berurutan.
  Eksekusi lokal: `nbclient` + `ipykernel` (dipasang di `.venv`), runner di scratchpad (`NotebookClient(...).execute()`).
- `pkill -f` lagi-lagi membunuh shell sendiri (exit 144) saat menghentikan poller → pakai PID dari `ps … | grep "[p]ola"`.
- Bash tool foreground dibatasi ~10 menit; tunggu job panjang dengan `run_in_background` + `until …; do sleep …; done`.
