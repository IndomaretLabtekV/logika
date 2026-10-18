# Kaggle CLI: push kernel, ambil hasil, submit (alur sejak 16 Okt)

Sejak 16 Okt assistant menjalankan model **langsung lewat Kaggle API** (bukan user upload manual). Sejak 17 Okt model
juga boleh dijalankan lokal (GPU 3070); Kaggle dipakai untuk paralel, VRAM besar, dan notebook final.

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

## Tambahan 17–18 Okt
### Auth: access token kedaluwarsa tanpa login ulang
- 18 Okt 01:29 WIB semua perintah membalas "Authentication required". `kaggle auth print-access-token` mencetak access
  token baru dari refresh token (tanpa menyimpannya), jadi cukup:
  ```
  KAGGLE_API_TOKEN="$(kaggle auth print-access-token 2>/dev/null)" kaggle <perintah>
  ```
  Praktisnya dibuat wrapper kecil di scratchpad sesi (`kg`), isinya hanya baris itu + `exec kaggle "$@"`. Token **tidak**
  ditulis ke file mana pun. Kalau `print-access-token` juga gagal, baru minta user `! kaggle auth login --force`.

### Pola kernel v22+ (berbeda dari v10–v21)
- Builder `notebook/build_vNN.py` menyematkan file repo (`features.py`, `nn_prep.py`, `valid2.py`, `run_valid2.py`,
  dan script lain bila perlu) sebagai string → ditulis ke `/kaggle/working/code/` → dijalankan dengan
  `subprocess.Popen` per GPU (`CUDA_VISIBLE_DEVICES=0/1`). Jadi kernel Kaggle menjalankan kode yang sama dengan lokal.
- Builder menulis `notebook/vNN.ipynb`; salin ke `results/vNNk/<slug>.ipynb` lalu `kaggle kernels push -p results/vNNk`.
  Notebook staging sudah dihapus saat bersih-bersih 18 Okt (duplikat `notebook/vNN.ipynb`); `kernel-metadata.json` tetap ada.
- `kaggle kernels output` menaruh file di `results/vNN/vNN/...` dan log di `results/vNN/<slug>.log`. Salinan kode yang
  dipakai saat run ada di `results/vNN/code/`.

### Batasan yang ditemukan
- **Maksimal 2 sesi GPU batch bersamaan.** Push ke-3 ditolak "Maximum batch GPU session count of 2 reached". Kernel v28
  (duplikat run lokal) dihapus (`kaggle kernels delete`) supaya v29 bisa jalan.
- **Status CANCEL_ACKNOWLEDGED bisa sementara.** 17 Okt 18:58 v25/v26 terbaca CANCEL_ACKNOWLEDGED, disimpulkan
  "terblokir", padahal 21:30 v25 COMPLETE dan v26 RUNNING. Cek `kaggle kernels status` + `kaggle kernels list --mine`
  minimal 2 kali dengan jeda sebelum menyimpulkan.
- Kernel CLI tidak punya secret → TabPFN (gated) tidak bisa di Kaggle; dataset berisi token ditolak (jangan dicoba).
- Paket `structured-data-models` (Kumo) dan bobotnya diunduh saat runtime (Internet on), tidak butuh token.

### Slug kernel 17–18 Okt
| Slug | Isi | Hasil |
|---|---|---|
| logika-v22-clean | rerun bersih resep v8 (6 GBDT × 3 seed, TabM 3 seed, Causilo) | COMPLETE, fondasi validation_v2 |
| logika-v23-zoo | zoo TabArena ronde 1 (TabICL/v2, iLTM, TabDPT, Nori, LimiX, Mitra) | iLTM 0,5216; lainnya OOM/ditolak |
| logika-v25-zoo2 | zoo ronde 2 (LimiX ctx 15k, TabDPT, Nori, Mitra, TabICLv2) | LimiX OOM, TabDPT gagal load, Nori gagal, Mitra 0,4488 |
| logika-v26-kumo | Kumo large 2 draw ctx 20k, 5 fold | COMPLETE, = run lokal (reproducible) |
| logika-v27-tabfm | TabFM fold 0 (v2: fp32, ctx 8k/4k) | null (0,5173 / 0,5169) |
| logika-v28-kumo56k | Kumo context penuh 5 fold | **dihapus** saat jalan (duplikat run lokal, batas 2 sesi) |
| logika-v29-segcausilo | Causilo per segmen has_history | null |
| logika-v30-kumo70k | Kumo context 70k, test sekali (ukur waktu/VRAM) | 16.172 s, 6,5 GB |
| logika-v33-kumo-g1 | Kumo ctx penuh seed generator kedua, fold 3–4 | null (bagian dari v33m) |
| logika-final | notebook final penuh | COMPLETE, = v34 |

### Submit 17–18 Okt
Semua lewat `kaggle competitions submit ... -f <dir>/submission.csv -m "<isi + pertanyaan riset + perkiraan>"`.
File kandidat disalin ke folder sementara sebagai `submission.csv`. Daftar dan hasil: `04_riwayat_versi/ringkasan_skor.md`.
