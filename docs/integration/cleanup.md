# Cleanup21 Oktober2026

## Yang dipindahkan

Seluruh508file snapshot teman dipetakan pada [pemetaan_file.csv](pemetaan_file.csv),
dipindahkan,checksum byte diperiksa sebelum adaptasi path,dan dicek lagi oleh
`eda/159_integration_verify.py`. Tidak ada file unik sumber teman yang dibuang.
9builder sudah menjalani build-only di mirror sementara; notebook asli utuh.

## Yang dihapus dan alasannya

| Path lama/objek | Tindakan | Bukti/alasan |
|---|---|---|
| `results/teman/submission.csv` (salinan reguler) | Ganti dengan symlink relatif `submission_v4.csv` | KeduaCSV identik SHA256`890e05a3f0b09500142395d5498e939928328b2770c509ca035f760a8dea8344`; menghemat1.020.035byte tanpa mengubah prediksi atau akses nama lama |
| `Logika-main/` dan subdirektori kosong | Hapus direktori kosong setelah mapping/hash/syntax/CSV/build checks lolos | Semua508file sudah terpetakan; diperiksa tidak ada file/symlink tersisa; menggunakan `rmdir`,bukan penghapusan rekursif |
| Mirror `logika-build-check-*` sementara | Dibersihkan otomatis sesudah build checks | Hanya salinan verifikasi; tidak memuat hasil penelitian baru |

Symlink diperiksa ulang melalui SHA256 dan validasiCSV159. Semua CSV teman tetap
checkpoint historis,tidak otomatis dipromosikan atau disubmit.

## Duplikat yang sengaja dipertahankan

- Snapshot source/OOF/test utama dalam `results/vN`,`results/final*`: bukti resep dan
  reproduksibilitas run historis. Identik byte bukan alasan membuang provenance.
- Metadata teman identik pada `locked_recipe`/confirmation/uncertainty/attempt
  berbeda: makna peran eksperimen dan link historis tetap bernilai.
- Kode builder dengan fungsi ekstraksi berbeda tetap dipertahankan; tidak
  menimpa kode dengan nama sama atau menyamakan nomor versi lintas sumber.

Tidak menghapus dataset,environment,model lokal,hasil unik,OOF/test penting,
checkpoint VIP yang masih dapat dilanjutkan,diagnostik kegagalan atau lisensi.
Script156SSL dipertahankan; setelah pembaruan user tentang izin panitia,
jalur ini boleh diuji secara nested. Tidak ada git push atau submission akibat cleanup.


## Cleanup untuk GitHub setelah riset dipause

User meminta repository dirapikan agar bisa dipush sendiri. Pekerjaan ini tidak
melanjutkan eksperimen dan tidak melakukan commit/push atau rewrite history.

- Hapus4 direktori cache Python dari kode aktif dan salinan lokal,
  total1.079.571 byte; semuanya dapat dibuat ulang. Rincian
  [github_cleanup_deleted.json](github_cleanup_deleted.json).
- Keluarkan1.444 file artefak dari index Git,583.223.208 byte. **File sumber
  lokal tetap ada.** Prediksi/OOF/model/dataset/checkpoint/environment tidak
  dihapus. Rincian [github_untracking_summary.json](github_untracking_summary.json).
- Sebelum untracking, salin444 file kode/notebook/konfigurasi/ringkasan penting,
  total1.986.570 byte, ke `research/snapshots/` dan `artifacts/history_utama/`.
  Setiap salinan diperiksa SHA256; [manifest](github_archive_manifest.json).
- Pertahankan riwayat metode null, kode aktif kedua sumber, laporan, dan lisensi.
  Contoh diagnostik perbaris dan telemetry tetap lokal, tidak dipublikasikan.
- Perbarui README, status, templatehandoff, rekomendasi, dan panduan GitHub agar
  tidak menyatakan runlama masih berjalan atau meresume riset yang dipause.
- Backup index, perubahan awal dan dokumentasi tersimpan di direktori tetangga
  di luar root repository. Dataset/environment tidak dicopy atau dihapus.

History Git tetap menyimpan blob lama; ukuran checkout baru bukan ukuranhistory.
Tidak ada blob > 100 MiB pada pemeriksaan awal; Archive.zip historis sekitar 85 MB
masih ada dihistory. Lihat [docs/github.md](../github.md). Tidak ada rewrite/force push.
