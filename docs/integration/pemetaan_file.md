# Pemetaan sumber ke proyek utama

Pemetaan **seluruh508 file**, checksum sebelum/sesudah, ukuran, dan asal:
[pemetaan_file.csv](pemetaan_file.csv). Sumber sebelum dipindahkan:
[inventaris_awal.csv](inventaris_awal.csv). Tidak ada collision tujuan.

| Asal snapshot | Lokasi sekarang | Alasan |
|---|---|---|
| `Logika-main/scripts/*.py` | `scripts/*.py` | Entry point riset alternatif,69 script; belum ada folder scripts root sebelumnya |
| `Logika-main/notebooks/*.ipynb` | `notebook/*.ipynb` | Satu folder notebook; nama01–12 tidak bertabrakan dengan vN |
| `Logika-main/analisis_logika.ipynb` | `notebook/analisis_logika.ipynb` | Notebook pembaca laporan |
| `Logika-main/notebooks/README.md` | `notebook/README_teman.md` | Indeks historis, asal jelas |
| `Logika-main/artifacts/**` | `artifacts/**` | Laporan, metrik, checksum, resep terkunci; dipertahankan |
| `Logika-main/third_party/**` | `third_party/**` | Kode sumber beserta NOTICE/LICENSE byte-identik |
| `Logika-main/requirements*.txt` | `environments/teman/requirements*.txt` | Dependency historis terpisah dari runtime utama |
| `Logika-main/submission*.csv` | `results/teman/submission*.csv` | Checkpoint historis asal teman; tidak menyamakan vN kedua sumber |
| `Logika-main/README.md` | `docs/integration/riset_teman.md` | Catatan riset sumber, ditautkan dari README utama |
| `Logika-main/.gitignore` | `docs/integration/gitignore_teman.txt` | Bukti asal; pola relevan digabung ke root.gitignore |

## Penyesuaian setelah byte terverifikasi

- Input kedua pipeline sekarang konsisten **ROOT/data**. Pengaturan path proyek
  tetap melalui `ROOT`; notebook teman juga mendukung `LOGIKA_PROJECT_DIR` dan
  `LOGIKA_FEATURE_DIR`. Tidak membuat salinan data/raw. Parquet teman memakai
  `data/processed`, cache utama tetap `data_pq`; keduanya hasil turunan yang berbeda.
- Kode builder/sel notebook: `notebooks/`→`notebook/`, `data/raw/`→`data/`.
  Output submission root teman→`results/teman/`; requirements→`environments/teman/`.
  Link README dipetakan relatif terhadap lokasi tujuan.
- Perubahan mekanis dapat direkonstruksi dari [adaptasi_path.patch](adaptasi_path.patch).
  Metadata JSON dan output historis notebook tidak dianggap run pascamigrasi.
- `scripts/reporting.py` ditindaklanjuti agar notebook EDA yang dibuat ulang juga
  ditulis ke folder tunggal. Patch tindak lanjut disimpan terpisah.
- Kode resep baseline utama152–157,features/valid2/common,builder submission,
  cache features,dan submissionv31 dijaga byte-identik; bukti pada
  [verifikasi159](../../eda/out/159_integration_verify.json).

Sesudah integrasi, user kembali mengizinkan SSL. Source156 ditambah threshold
kelas opsional; versi sebelumnya dipertahankan byte-identik dalam
`research/locked_recipes/156_kumo_ssl_context_pre_class_threshold.py`. Ini perubahan
perilaku riset yang disengaja dan terpisah dari migrasi. Default selector diperiksa
terhadap arsip; modul/core resep produksi tidak diganti.

Pemetaan bukan klaim model membaik. Menjalankan kode teman dengan paket baru
tidak otomatis mereproduksi hasil lama; environment historis dan split tetap
harus diikat ke manifest eksperimen.


### Penyimpanan GitHub setelah cleanup

Pemetaan508file di atas menjelaskan integrasi fisik lokal. Untuk publikasi,
`results/` kini diabaikanGit: CSV historis teman dan seluruh hasil model tetap
lokal, tidak hilang. Snapshot implementasi utama serta ringkasan penting disalin
ke `research/snapshots/` dan `artifacts/history_utama/` dengan checksum identik.
Builder/notebookv34 yang dulu dijaga tidak berubah sudah diarsipkan byte-identik
pada `notebook/archive/production_v34_21okt/`; builderaktif sekarang untukv59.
Rincian publikasi ada di [github_archive_manifest.json](github_archive_manifest.json).
