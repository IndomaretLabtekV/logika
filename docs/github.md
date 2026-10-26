# Menyiapkan commit GitHub

Repository mempublikasikan kode, notebook tanpa output runtime, protokol eksperimen, dokumentasi, laporan agregat, dan lisensi. Dataset, environment, bobot model, OOF, prediksi test, CSV submission, cache, log dan ZIP disimpan lokal.

## Pemeriksaan dari root

Gunakan Python 3.13 atau lebih baru, tanpa menginstal dependensi ML:

```bash
python3 tools/check_repository.py
python3 -m unittest discover -s tools/tests
git diff --check
git status --short
```

Checker memeriksa sintaks Python/notebook, pola kredensial, nama file sensitif, artefak lokal, dan symlink relatif. Symlink DSC0152 ke data/results diperiksa sebagai teks target; dataset dan hasil yang dirujuk tidak dibaca. Sel notebook dengan magic IPython tidak diparse sebagai Python biasa dan jumlahnya dilaporkan.

Mode default membaca file kerja yang akan dipublikasikan oleh `git add -A`, termasuk file baru yang tidak diabaikan dan penghapusan file kerja. Mode `--staged` membaca byte index, sehingga perubahan yang belum di-stage tidak dianggap masuk commit. Laporan lokal ditulis ke `.repo-checks/`.

## Meninjau dan membuat commit

Cleanup disiapkan sebagai perubahan working tree. Setelah meninjau perubahannya:

```bash
git add -A
python3 tools/check_repository.py --staged
git diff --cached --check
git diff --cached --stat
git commit -m "Clean repository layout and publication artifacts"
```

Push dapat dilakukan setelah commit ditinjau. Pemeriksaan ini tidak menjalankan training, submission, commit, atau push.

## Menjaga repository tetap rapi

- Bersihkan output sel, execution count dan state widget notebook kerja sebelum commit. Salinan audit dalam `research/snapshots/` dan `notebook/archive/` mempertahankan byte historis untuk checksum/provenance.
- Simpan hasil model dan eksperimen besar di `results/`, dan dataset di `data/` atau `data_pq/`.
- Folder output runtime `eda/**/out/` dan `outputs/` mengabaikan file baru. Laporan agregat yang sudah tercatat tetap dipertahankan; pilih laporan baru satu per satu dengan `git add -f path/ke/laporan.json`, setelah memastikan tidak berisi data per baris atau kredensial.
- Gunakan `artifacts/` untuk protokol, konfigurasi dan laporan kecil; tempatkan konteks sesi bertanggal di `handoff/`.
- Jangan menaruh kredensial di source/notebook. Pemindaian pola membantu menemukan bentuk yang dikenal, tetapi tidak menjamin seluruh bentuk secret terdeteksi.

Peta folder dan entry point tersedia di [README](../README.md). Pada clone baru, dataset, prediksi replay, cache dan bobot perlu disediakan terpisah. Builder notebook merge membutuhkan manifest lokal; notebook hasil build sudah tersedia.

## History dan backup lokal

Cleanup checkout tidak mengecilkan history Git yang sudah dipush. Audit 21 Oktober mencatat sekitar 559 MiB objek Git, termasuk blob ZIP lama sekitar 85 MB. Itu temuan historis, bukan pengukuran ulang pada cleanup ini. History tidak ditulis ulang.

Cleanup 26 Oktober menyimpan backup notebook beroutput serta cache yang dipindahkan di `.repo-checks/cleanup_26okt/`, yang diabaikan Git. Ringkasan perubahan ada di [cleanup_26okt.md](integration/cleanup_26okt.md), dengan [manifest](integration/cleanup_26okt.json). Cleanup sebelumnya tetap tercatat di [cleanup.md](integration/cleanup.md).
