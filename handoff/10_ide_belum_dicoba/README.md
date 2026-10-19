# Ide yang belum dicoba (per 19 Okt 2026 malam)

Semua di sini **belum dijalankan** (atau dibatalkan sebelum ada hasil). Jangan menjalankan tanpa persetujuan user.
Urutan = prioritas menurut bukti yang ada.

| # | Ide | Kenapa layak | File |
|---|---|---|---|
| 1 | **Audit dan perbaikan input Kumo-Tabular** | tim 0,57065 memakai ensemble Kumo 3 seed; Kumo-only kita di public 0,56056 (−0,010). Input kita sederhana dan mungkin salah (kategori dipaksa numerik, hanya 100 fitur top-KS univariat) | `kumo_setup_audit.md` |
| 2 | **VIP-COP** (optimasi context/fitur black-box untuk TFM, arXiv 2605.12904) | metode publik yang secara eksplisit mengoptimalkan fitur/baris context TFM; forward Kumo cukup cepat (3–4 s) | `vipcop_context_optimization.md` |
| 3 | Ensemble Kumo multi-seed dengan input terbaik | meniru tim teman; di OOF kita 3 seed tidak naik, tapi di public belum pernah diuji | `kumo_setup_audit.md` |
| 4 | Ide lain berekspektasi rendah | lihat daftar | `ide_lain.md` |
