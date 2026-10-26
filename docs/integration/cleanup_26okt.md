# Cleanup repository — 26 Oktober 2026

Cleanup disiapkan dari commit `275903a`, dengan working tree awal bersih. Pekerjaan ini menata file publikasi dan entry point; tidak menjalankan model, mengubah dataset, menulis ulang history, commit, atau push.

## Perubahan

| Bagian | Hasil |
|---|---|
| Cache Matplotlib | Lima fontlist cache dipindahkan dari `eda/**/out/mpl_cache/` ke backup lokal; 2.334.415 byte dikeluarkan dari checkout publik |
| Notebook kerja | Output, execution count dan metadata eksekusi/widget dibersihkan pada 51 notebook; berkurang 9.400.614 byte, source tiap sel dipertahankan |
| Snapshot audit | `research/snapshots/` dan `notebook/archive/` tidak dibersihkan ulang agar checksum bukti historis tetap cocok |
| Handoff | 29 dokumen dari `serah_terima_25okt/` dikumpulkan di `handoff/2026-10-25/`; rujukan Markdown diperbarui |
| Navigasi | README root, indeks handoff, dan panduan `eda/`, `scripts/`, `notebook/` diperjelas berdasarkan jurnal merge 25 Oktober |
| Git ignore | Aturan environment, output, cache, model, kredensial, ZIP dan file editor dikelompokkan; output baru harus dipilih secara sadar |
| Runner DSC0152 | Skrip yang tadinya hanya ada di `results/run_dsc_s42.sh` disediakan di `scripts/run_dsc_s42.sh`, dengan root relatif dan Python yang dapat dikonfigurasi |
| Resep dan notebook merge | Perintah famili D memakai runner yang ikut Git; builder mengembed runner dan notebook merge dibangun ulang tanpa mengeksekusi sel |
| Pemeriksaan | Checker menangani symlink tanpa mengikuti target, membedakan byte working tree/index, memeriksa output notebook dan lebih banyak bentuk artefak lokal |

Notebook original sebelum dibersihkan dan lima fontlist cache tersedia dalam `.repo-checks/cleanup_26okt/`. Folder itu hanya lokal, diabaikan Git. Daftar file dan ukuran tersimpan di [manifest cleanup](cleanup_26okt.json).

Kode eksperimen bernomor, builder per versi, protokol, laporan agregat dan snapshot historis tetap dipertahankan karena dirujuk oleh jurnal atau merekam resep berbeda. Dataset, environment, model lokal, OOF/prediksi, hasil eksperimen dan ZIP yang sudah ada tidak dihapus.

## Validasi

Pemeriksaan publikasi dan delapan unit test checker dijalankan tanpa dependensi ML. Source sel notebook yang dibersihkan dibandingkan dengan backup; source final merge yang diembed dicocokkan dengan file repository. Sintaks runner shell, whitespace diff, tautan dokumentasi yang ditata, dan snapshot historis ikut diperiksa. Payload commit juga diuji di repository sementara tanpa dataset atau target symlink lokal.

Batas pemeriksaan: tidak mengulang training penuh, tidak membuktikan skor model baru, dan tidak memperkecil history yang sudah ada di remote. Mode replay memerlukan artefak lokal. Langkah commit ada di [panduan GitHub](../github.md).
