# Status ide per21Oktober2026

**Pembaruan user terbaru setelah audit162:** pseudo-labeling dan teknik
semi-supervised data panitia kembali boleh. Jalur156 diaktifkan untuk smoke dan
screening nested setelah jobVIP selesai. Pernyataan laranganSSL pada paragraf
di bawah ditulis lebih awal dan sudah diganti; batas data eksternal/future
leakage/LB probing serta gerbang breakthrough tidak berubah.

Audit input153 dan precision157 sudah selesai. Engine cache154 dan resume152
lolos smoke; VIPv48 fold0 selesai dengan Δouter−0,006096 dan tidak diteruskan.
SSL156 penuh selesai:12policy kalah inner, fallbackbaseline. Input173 native dan
poolteman, selector177, blend176/179/180 sudah diuji dan belum lolos gerbang.
Log-uang181 naik search tetapi gagal confirmation. Full232 terhalang allocator,
SDK sudah menjalankan estimator satu per satu. Retrieval183 sudah diuji:
bestblend+0,002458<0,003. Jangan mengulang daftar ini sebagai ide belum dicoba.
Same-class labeled-training mixup185 sedang berjalan;berbeda dari pseudo-testSSL.
Belum ada kandidat yang lolos5fold×3seed. Balanced/full56k/e16/denoised/latent
sudah null di protokol teman; lihat [indeks](../../docs/integration/indeks_eksperimen.md).

Instruksi otonom21Okt sudah mengizinkan pekerjaan ini. Setelah pembaruan user,
ideSSLtest di bawah kembali boleh diuji secara nested dengan data panitia.
LB probing tetap dilarang. Semua hasil harus memenuhi gerbang validation_v2
yang diperketat dalam lampiran21Okt.

---

## Daftar historis19Okt (status belum-dicoba saat ditulis)

Semua di sini **belum dijalankan** (atau dibatalkan sebelum ada hasil). Jangan menjalankan tanpa persetujuan user.
Urutan = prioritas menurut bukti yang ada.

| # | Ide | Kenapa layak | File |
|---|---|---|---|
| 1 | **Audit dan perbaikan input Kumo-Tabular** | tim 0,57065 memakai ensemble Kumo 3 seed; Kumo-only kita di public 0,56056 (−0,010). Input kita sederhana dan mungkin salah (kategori dipaksa numerik, hanya 100 fitur top-KS univariat) | `kumo_setup_audit.md` |
| 2 | **VIP-COP** (optimasi context/fitur black-box untuk TFM, arXiv 2605.12904) | metode publik yang secara eksplisit mengoptimalkan fitur/baris context TFM; forward Kumo cukup cepat (3–4 s) | `vipcop_context_optimization.md` |
| 3 | Ensemble Kumo multi-seed dengan input terbaik | meniru tim teman; di OOF kita 3 seed tidak naik, tapi di public belum pernah diuji | `kumo_setup_audit.md` |
| 4 | **Semi-supervised untuk Kumo**: augmentasi context dengan baris test berlabel semu (pseudo-label dari model train fold), dipilih oleh VIP-COP | diizinkan panitia 20 Okt; paper VIP-COP menguji setting augmentasi sampel dan melaporkan gain; self-training pohon v19 dulu ≈ 0, tapi ICL dengan context lebih besar belum pernah diuji | `ide_lain.md` |
| 5 | Ide lain berekspektasi rendah | lihat daftar | `ide_lain.md` |
