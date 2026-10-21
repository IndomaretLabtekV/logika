# Repository untuk GitHub

Repository ini berisi satu basis kode gabungan, konfigurasi, notebook, laporan
ringkas, dan riwayat riset. Riset tetap dipause; cleanup tidak menjalankan model,
mengirim submission, membuat commit, melakukan push, atau menulis ulang history.

## Yang dipublikasikan

- Kode aktif `eda/`, `scripts/`, builder dan notebook dalam `notebook/`.
- Dokumentasi, handoff, indeks eksperimen, dan laporan agregat.
- Resep historis dalam `research/snapshots/`, ringkasan dalam
  `artifacts/history_utama/`, dan sumber teman dalam `artifacts/`.
- LICENSE/NOTICE upstream dan snapshot environment.

## Yang hanya tersimpan lokal

`data/`, `data_pq/`, `results/`, environment `.venv*`, bobot model, OOF,
prediksi test, CSV submission, runtime log, cache, dan contoh diagnostik yang
mengandung ID baris. Mengeluarkan file dari Git tidak menghapusnya dari disk.
Kode historis dan ringkasan penting diarsipkan dahulu; lihat
[manifest](integration/github_archive_manifest.json).

Clone baru tidak membawa hasil model atau dataset. Mode notebook `replay`
membutuhkan prediksi lokal dan tidak bisa dipakai langsung dari clone kosong.
Mode `fresh` menyediakan langkah fitur/training; butuh data dan runtime yang sesuai.
Verifikasi integrasi `eda/159_integration_verify.py` juga memerlukan artefak
lokal. Untuk mengecek clone tanpa data, gunakan pemeriksaan ringan berikut.

## Sebelum commit dan push

```bash
python3 tools/check_repository.py
git diff --check
git status --short
git diff --cached --stat
```

Penghapusan dari index sudah disiapkan menggunakan `git rm --cached`; file lokal
masih ada. Setelah meninjau perubahan, user dapat menjalankan:

```bash
git add -A
python3 tools/check_repository.py --staged
git diff --cached --check
git commit -m "Consolidate research repository and exclude local artifacts"
git push
```

Perintah commit/push di atas adalah untuk user; assistant tidak menjalankannya.
Pemeriksaan melakukan syntax parse dan scan pola kredensial, bukan membuktikan
ulang skor model atau menjamin seluruh bentuk secret yang mungkin ada terdeteksi.

## History Git lama

Checkout yang bersih tidak otomatis membuat history menjadi kecil. Audit sebelum
cleanup menemukan sekitar 559 MiB objek Git, termasuk `Archive.zip` historis
sekitar 85 MB. Tidak ditemukan blob di atas batas 100 MiB pada audit tersebut.
History dipertahankan: cleanup ini tidak melakukan filter/rebase/force-push.
Karena itu push atau clone pertama masih dapat mengunduh history yang besar.
Jika nanti ingin mengecilkan history, kerjakan sebagai tindakan terpisah setelah
backup dan menentukan dampaknya terhadap remote/collaborator.

Backup perubahan dan index sebelum cleanup berada di direktori tetangga di luar
repository, bukan bagian payload GitHub. Rinciannya ada pada
[ringkasan cleanup](integration/github_untracking_summary.json).
