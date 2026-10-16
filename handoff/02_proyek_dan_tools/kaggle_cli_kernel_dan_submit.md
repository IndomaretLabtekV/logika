# Kaggle CLI: push kernel, ambil hasil, submit (alur sejak 16 Okt)

Sejak 16 Okt assistant menjalankan model **langsung lewat Kaggle API** (bukan user upload manual). Tidak ada model atau
notebook yang dijalankan lokal.

## Akun dan auth
- Akun Kaggle: `riantamaputra`. Kaggle CLI 2.2.4, kredensial OAuth di `~/.kaggle/credentials.json`.
- **Access token kedaluwarsa** (contoh: 16 Okt 14:58 UTC) dan CLI 2.2.4 tidak me-refresh otomatis. Gejala: semua
  perintah membalas "Authentication required". `kaggle auth login` hanya bilang "already logged-in" karena cuma
  mengecek file. Solusi: user menjalankan `! kaggle auth login --force`. Cek kedaluwarsa tanpa membuka secret:
  `python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/credentials.json')))['access_token_expiration'])"`.
- Kompetisi: `data-science-competition-logika-ui-2026`.

## Membuat dan menjalankan kernel
1. Builder `notebook/build_vNN.py` menyusun notebook dari: markdown judul, `%pip install`, import, **sel 12
   `notebook/v10.ipynb`** (= `eda/features.py`, fungsi `build_all(tables)`), isi `notebook/vNN_experiment.py`, lalu
   `main_vNN(build_all)`. Builder hanya `compile()` setiap sel (cek sintaks), tidak mengeksekusi.
2. Salin notebook ke folder staging `results/vNNk/<slug>.ipynb` dan tulis `results/vNNk/kernel-metadata.json`
   (contoh di folder `results/v1Xk/`; notebook staging sudah dihapus, buat ulang dengan builder):
   ```json
   {"id": "riantamaputra/logika-vNN-xxx", "title": "Logika vNN xxx", "code_file": "logika-vNN-xxx.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": true, "enable_gpu": true, "enable_tpu": false,
    "enable_internet": true, "keywords": [], "dataset_sources": [], "kernel_sources": [],
    "competition_sources": ["data-science-competition-logika-ui-2026"], "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"}
   ```
   Untuk CPU: `"enable_gpu": false` dan hapus `machine_shape`.
3. `kaggle kernels push -p results/vNNk` → jalan otomatis (Save & Run All).
4. Poll di background: `kaggle kernels status riantamaputra/<slug>` sampai bukan RUNNING/QUEUED.
5. `kaggle kernels output riantamaputra/<slug> -p results/vNN` → file `/kaggle/working/vNN/` masuk ke
   `results/vNN/vNN/` dan log ke `results/vNN/<slug>.log`.
6. CLI **tidak bisa membatalkan** kernel yang sedang jalan; stop lewat UI ("Stop session").

## Dataset private yang dibuat
| Dataset | Isi | Dipakai |
|---|---|---|
| `riantamaputra/logika-v8-preds` | `v8_oof.parquet`, `v8_test.parquet` (prediksi 9 model v8 dengan application_id) | v15, v18 |
| `riantamaputra/logika-mature-history` | `mature_history.parquet` (= `eda/v8_review/out/03_selected_history_features.parquet`) | v21 |
Buat dataset: folder berisi file + `dataset-metadata.json` (`{"title":…, "id":"riantamaputra/<slug>", "licenses":[{"name":"CC0-1.0"}]}`), lalu `kaggle datasets create -p <folder>`.

## Kuota dan batasan
- GPU: **30 jam per minggu**, habis 16 Okt setelah v11/v12/v14/v16/v17 (v17 sendiri ~3 jam). Push GPU ditolak dengan
  "Maximum weekly GPU quota of 30.00 hours reached". CPU tidak terkena kuota itu.
- Sesi maksimal 12 jam; experiment memakai `DEADLINE` di dalam loop.
- Secret (`HF_TOKEN`/`TABPFN_TOKEN`, untuk TabPFN-3.5 gated) dipasang per notebook di UI. Kernel baru hasil push CLI
  kemungkinan tidak punya secret → TabPFN gagal. Causilo dan TabICL tidak butuh token.

## Slug kernel yang sudah ada
logika-v10-nested-tuning (CPU), logika-v11-tabm-tuning (T4), logika-v12-tabm-capacity (T4), logika-v13-ebm (CPU),
logika-v14-hybrid-events (T4), logika-v15-rank-l2 (CPU), logika-v16-tabicl (T4×2), logika-v17-tabicl-full (T4×2),
logika-v18-bias (CPU), logika-v19-pl (CPU), logika-v21-feat (CPU). logika-v20-fm16 tidak pernah ter-push.

## Submit
- Batas **3 submit per hari**.
- Perintah: `kaggle competitions submit data-science-competition-logika-ui-2026 -f <dir>/submission.csv -m "<deskripsi>"`.
  Preferensi user: nama file **selalu `submission.csv`** (salin ke folder sementara), penjelasan isi di `-m`.
- Cek skor: `kaggle competitions submissions data-science-competition-logika-ui-2026` (baris paling atas = terbaru).
- Validasi sebelum submit: kolom `application_id,default_probability`, 30.000 baris, ID unik dan sama dengan
  `data/sample_submission.csv`, tanpa null, nilai di [0, 1].
- Submit dari assistant bisa ditolak classifier izin Claude Code ("Real-World Transactions") kalau user belum
  eksplisit meminta. Setelah user bilang "coba submit aja sendiri", submit berjalan.
