# Rancangan dan rencana kerja

Pembaruan user setelah audit162: pseudo-labeling dan teknik semi-supervised data
panitia kembali diizinkan; bagian laranganSSL dari lampiran awal di bawah adalah
riwayat. Prioritaskan smoke156 lalu screening nested setelah jobVIP selesai.

Instruksi kerja: lampiran user21Okt, pengerjaan otonom; tidak push, tidak pseudo-label
test, tidak fitting distribusi test, tidak submit tanpa breakthrough. Instruksi ini
menggantikan izin SSL/probe historis. Tujuan penuh tetap konsolidasi dan penelitian
berulang menuju breakthrough tervalidasi, bukan sekadar migrasi file.

## Urutan dan bukti selesai

- [x] Baca konteks utama, scripts/laporan/metadata/notebook teman, status Git dan
  proses. Simpan inventaris dan checksum sebelum menyentuh sumber.
- [x] Bandingkan resep, cutoff, split, context, seed, dependensi, artefak/null.
- [x] Migrasikan semua508 file melalui pemetaan deterministik, cegah collision,
  verifikasi checksum byte sebelum adaptasi. Jangan hapus hasil unik.
- [x] Satukan path data/notebook; catat perubahan path terpisah dari perilaku.
  Simpan metadata historis apa adanya; kode notebook diadaptasi, outputs historis
  tidak dijadikan bukti eksekusi sesudah migrasi.
- [x] Uji syntax/import/entrypoints, CSV30k/ID/finite/range dan
  integritas hasil/caches utama. Periksa perubahan awal tetap ada.
- [x] README/handoff/index eksperimen/null dan gitignore tunggal. Hapus folder
  snapshot hanya setelah setiap file terpetakan dan terverifikasi.
- [x] Resume VIP152 dari checkpoint16 dengan config/source/data/packages identik
  setelah memastikan proses lama memang tidak hidup. Catat PID/cgroup/log.
- [x] Selesaikan VIPv48 fold0 dan evaluator155: Δouter−0,006096,P>0=0,147;
  kandidat null, tidak diteruskan ke5×3seed.
- [x] SmokeSSL156:10fit,ID/label/pseudo/cache sehat,replay byte-identik.
- [ ] Selesaikan screeningSSLv49 yang sedang berjalan;outer hanya setelah
  threshold/cap terpilih dari inner, lalu keputusan lanjut sesuai gerbang.
- [ ] Uji perbedaan kategori/input lalu fitur teman di inner dan outer screening;
  kandidat menjanjikan→5×3seed. Catat baseline, kriteria sebelum run, hasil/null.
- [ ] Audit breakthrough, forecast/submission hanya jika seluruh gerbang lolos.
- [ ] Bila resep submission berubah, perbarui builder dan notebook, run lokal penuh.

## Pengujian terarah

Migration verifier memeriksa seluruh manifest, original→destination, SHA256,
source AST (perubahan hanya literal path), seluruh notice/license, tidak ada
ketergantungan aktif folder lama. Snapshot resep utama/OOF/test tidak berubah.
Notebook builder diuji build-only, parse/compile cells, tanpa menjalankan training
mahal/menulis hasil historis. Environment historis teman tetap terpisah karena
numpy1.24/pandas1.5/torch1.13 bertentangan dengan environment Kumo saat ini.

Protokol local gate menggunakan evaluator155 dan validation_v2; evaluator bukan
pengganti audit prosedur pembentukan prediksi. Bootstrap resample baris berpasangan,
seed ensemble tidak dihitung sebagai observasi independen.
