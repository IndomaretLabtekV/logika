# Snapshot implementasi historis

Folder ini mempertahankan kode dan notebook yang sebelumnya tersimpan di
`results/`. Seluruh file disalin byte-identik sebelum `results/` dikeluarkan
dari Git. Implementasi aktif tetap berada di `eda/`, `scripts/`, dan `notebook/`.

Struktur relatif sumber dipertahankan di bawah `research/snapshots/`. Snapshot
merupakan bukti resep lama, bukan entrypoint baru: beberapa modul memakai path
relatif terhadap lokasi asalnya. Untuk reproduksi versi lama, gunakan layout
asal pada proyek terisolasi dan sediakan data/artefak yang diperlukan.

Lokasi asal, SHA256, dan arsip konfigurasi tercatat pada
[manifest cleanup](../../docs/integration/github_archive_manifest.json).
Salinan asli serta semua OOF/model tetap tersedia lokal di `results/`.
