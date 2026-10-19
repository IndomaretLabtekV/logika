# Langkah berikutnya (per 19 Okt 2026 malam; deadline notebook 30 Okt 23:59 WIB)

Status rekomendasi lama (16 Okt dan 18 Okt) ada di `status_rekomendasi_lama.md`. Ide yang belum dicoba beserta rencana
implementasinya ada di `../10_ide_belum_dicoba/`.

## Aturan dulu (dari teguran user 18–19 Okt)
- **Pertanyaan user = jawab saja.** Jangan menjalankan script/run/submit sebelum user minta.
- **Semua run di lokal**; tidak ada push kernel Kaggle (kuota habis) kecuali user minta saat itu.
- **Jendela submit**: sampai 21 Okt 06:00 WIB hanya breakthrough (lolos validation_v2 dengan gain jelas > 0,002 vs v31,
  idealnya dengan rata-rata seed). Setelah 06:00 tanpa breakthrough, slot boleh untuk analisis (rencana di bawah).
- Jangan menyebut "final".

## Inti situasi
- Public 0,56405 (#33/118). Top-1 0,57788; 9 tim > 0,570. Plafon pendekatan kita ~0,564–0,565.
- Pohon jenuh; CV ICL tidak transfer. Satu-satunya petunjuk kuat yang tersisa datang dari luar: **tim 0,57065 memakai
  ensemble Kumo 3 seed**, sedangkan Kumo-only kita di public 0,56056 → selisih +0,010 justru di keluarga yang menurut
  kita "di-overrate CV". Hipotesis: setup Kumo kita (input fitur, preprocessing, context) yang salah/lemah untuk public,
  bukan Kumo-nya.

## Prioritas yang direkomendasikan (minta persetujuan user dulu)
| # | Langkah | Biaya | Ekspektasi |
|---|---|---|---|
| 1 | **Audit input Kumo**: apa yang dilihat Kumo di `valid2.kumo_fit` (100 numerik top-KS univariat + 4 kategori sebagai kode float, tanpa nama/semantik kategori, NaN thin-file di 44 kolom) vs alternatif (semua 232 fitur dalam 2–3 grup, kategori sebagai stype categorical, fitur gain-ranked, fitur yang dipakai pohon) — screening fold 0 dengan pembanding Kumo ctx penuh yang sama (cache `eda/out/kumo_finetune_f0_base.parquet`) | ~20 menit per varian | ini yang paling mungkin menjelaskan 0,560 vs 0,571 |
| 2 | **VIP-COP tingkat fitur** untuk Kumo (paper arXiv 2605.12904; script `eda/152_vipcop_kumo.py` sudah ada, belum dijalankan sampai selesai) | ~1–1,5 jam fold 0 | moderat |
| 3 | **Ensemble Kumo 3 seed** dengan input terbaik dari 1–2; nilai keluarga Kumo vs pohon; probe public setelah 21 Okt 06:00 (atau lebih awal bila CV + logika kuat — tanya user) | ~3–9 jam lokal | tergantung 1–2 |
| 4 | Banner tim (nomor peserta DSCxxxx, nama anggota; nama tim Kaggle sekarang "lagi cape lomba asli") → rename notebook `ParticipantNumber_TeamName_Notebook` → run lokal penuh (2,8 jam) → cek Spearman vs submission terpilih | ~3 jam | wajib sebelum 30 Okt |
| 5 | Bila resep berubah (mis. Kumo baru): update `notebook/build_final.py`, run lokal penuh, sisakan 2 hari cadangan | ~3 jam | wajib |

## Rencana slot analisis setelah 21 Okt 06:00 (hanya bila TIDAK ada breakthrough; ditetapkan 19 Okt sebelum melihat hasil)
Window 20 Okt (berlaku s.d. 21 Okt 07:00) punya 3 slot pada 06:00–07:00; window 21 Okt mulai 07:00 (3 slot). Satu per
satu, perkiraan ± sd di `04_riwayat_versi/jurnal_18okt_sore.md`; file di `results/v42/`:
1. `lgbxgb_only` (0,5631 ± 0,0019) dan 2. `cat_only` (0,5598 ± 0,0024): apakah CatBoost (pohon simetris lebih halus)
   juga di-overrate CV seperti ICL.
3. `tabm_only` (0,5609 ± 0,0031): apakah keluarga NN ikut tertinggal di public.
4. `trees_kumo_50` (CV 0,52583; realistis ~0,563): pohon + Kumo tanpa Causilo.
5. `trees_deltas` (pohon + delta snapshot; CV ≈ pohon): apakah fitur temporal mengubah public.
6. Bila langkah prioritas 1–3 sudah menghasilkan Kumo baru: probe Kumo-baru-only (paling informatif).
`kumo70k3_only` **tidak dipakai** (Spearman 0,99923 vs v31c → tidak informatif).

## Pilihan submission akhir di Kaggle (bahas hanya kalau user minta)
- Pohon saja 0,56337 ≈ v31 0,56405 ≈ v32 0,56376: hedge yang wajar = v31/v32 (pohon + Kumo) dan satu kandidat
  pohon-berat; jangan memilih dari selisih public < 0,002.
- Notebook harus menghasilkan file yang disubmit: resep notebook sekarang = v34 (pohon + Causilo + Kumo-70k). Kalau
  submission yang dipilih berubah, builder harus diubah dan dijalankan ulang.

## Kalender usulan
| Tanggal | Kegiatan |
|---|---|
| 20 Okt | (setelah izin user) audit input Kumo + VIP-COP fold 0 |
| 21 Okt 06:00–07:00 | slot analisis window 20 Okt (bila tanpa breakthrough) |
| 21–24 Okt | Kumo 5 fold × 3 seed dengan input terbaik; probe; maks 3 submit/hari |
| 24–26 Okt | putuskan resep notebook, update builder, isi banner |
| 27–28 Okt | run lokal penuh notebook, cek output |
| 29–30 Okt | cadangan; kumpulkan sebelum 30 Okt 23:59 WIB |
