# Rekomendasi saat riset dipause

User menghentikan riset21Oktober karena waktu dan hasil belum memenuhi target.
Tidak ada eksperimen atau submission lanjutan tanpa instruksi baru. Pekerjaan
cleanup hanya menyiapkan repository untuk GitHub; user melakukan push sendiri.

Saat user meminta melanjutkan, baca status terbaru dan hasil null terlebih dahulu.
Public terbaikv31=0,56405; v59 terakhir0,56296. Validasi v56baru11/15 dan terputus
hang; jangan blindretry GPU. Data serta checkpoint tetap tersimpan lokal.
Dokumentasi entrypoint dan ketersediaan artefak: [README](../../README.md).

## Rekomendasi historis sebelum pause

Bagian berikut dipertahankan sebagai riwayat ide, bukan jadwal atau izin aktif.

# Langkah berikutnya aktif21Okt

**Steering terbaru user setelah audit162:** aktifkan kembali pseudo-labeling dan
semi-supervised data panitia. Prioritas setelah VIP adalah smoke156, kemudian
screening SSL penuh yang dipilih dari inner. Reference train-only menjadi kontrol
untuk penggunaan covariate tak berlabel, bukan alasan menghentikan jalur SSL.
Gerbang validation_v2 dan submission tetap berlaku. Daftar di bawah merekam urutan
sebelum pembaruan; larangan SSL pada butir4 sudah diganti izin terbaru ini.

1. VIPv48 fold0 selesai dan null: Δouter−0,006096 vs baseline matched,
   P>0=0,147. Jangan lanjut15fit untuk kandidat ini atau memilih tau lain dari
   outer. Screening SSL156 selesai di
   `results/v49/ssl_asymmetric_safe_f0`:12policy kalah di inner-search, pilihan
   noSSL, Δouter0. Jangan lanjut15fit untuk kebijakan ini. Runlama sempat hang/force-shutdown;
   amendemenf142e08 memakai chunk128/allocator0,7 dengan context/e4 tetap.
   Smokeparity dan preflight48k lulus;telemetriCSV dicatat selama model hidup.
   Source/config/data harus tetap sesuai manifest untuk replay.
2. Uji perbedaan konkret teman: kategori native dengan fitur/context/e4/cache
   matched,kemudian pool fitur teman yang direbuild dan reference fit nested.
   Null teman balanced/full56k/e16/latent menghindarkan pengulangan tanpa alasan baru.
   Cache `pp_*` lama mengikutsertakan histori customer test; penggunaannya
   kini diizinkan sebagai transduksi yang dinyatakan. Reference fit-only tetap
   kontrol ilmiah, bukan larangan otomatis SSL. Protokolinput baru621e359
   membatasi atribusi gain antarkeluarga fitur;daftar90historis bukan selector
   bersarang. Runner166 menyimpan dua kategori histori tambahan tanpa otomatis
   memasukkannya ke model. Lihat docs/integration/protokol_fitur_teman_validation_v2.md.
   Build166 inner100k selesai (48kfit/52kquery,1.000numerik+6kategori).
   Prepare173 awal menemukan shortlist200 hanya menghasilkan71 numerik teman;
   amendemen **ca261e5** sebelum inferensi mengisi100 dengan perluasan shortlist
   KS/Spearman yang sama, berlaku pada kedua keluarga. Delapan checks172 lulus.
   Full173 dan blend176 sekarang selesai null: nativeutama+0,000974 saja,
   kedua poolteman−0,018lebih,ketigablend kalah. Outer belumdibuka untuk keluarga
   input. Jangan lanjut15fit untuk pilihanfallback ini. Seleksi177
   menguji gainselector400round fit-only padautama/teman/union;protokol a170272,
   4checks178 lulus;semuasingle kalah. Blend179–180 terbaik+0,002623<0,003.
   Log-uang181 search+0,003385 tetapi confirmation−0,002713;tidaklanjutouter.
   All232engineeringOOM padaallocator0,7;SDK sudahestimatorbatch1,takadajalan
   batching sederhana. Langkahaktif183 adalahretrieval8kelompok/context12k,
   baselinefull48k danuniform12kcontrol. Protokolb0dacf1;3checks184lulus,
   selesai:bestprobblend+0,002458<0,003. Mixup185 juga kalah. FullSGD187 belum
   menghasilkanupdatekarenaOOMforward padaepisode1024/128 dan512/64.
   **Sekarang189fixedblend** utama_raw+union_gain50:50,referenceouter56k:
   outerf0+0,005438,matchedP0,906. Prosedurbaru tidakmemakai innergatebuatanagen
   yangterlaluketat;gateuserberlaku padafullvalidation. **191sedang5fold×3seed**
   diresults/v56;lihatstatus/progresssebelumrunbaru,janganjobberatparalel.
3. Seleksi hanya inner; screeningfold0→kandidat menjanjikan5fold×3seed.
   ΔKS>0,003,mayoritas fold,pairedP≥0,9,audit bersih. Laporkan pooled,perfold,seed.
4. Tanpa breakthrough jangan submit. SSL/pseudo-label data panitia kembali boleh
   sesuai pembaruan user; semua teacher/threshold/sampling nested. LB probing tetap
   dilarang, termasuk rencana slot analisis historis19Okt di bawah.
5. Bila resep berubah,update builder submission dan notebook,run penuh lokal.
   Banner/nama peserta masih memerlukan informasiuser menjelang pengumpulan30Okt23:59.
6. Steeringuser21Okt15:35:targetkenaikan/submissionhariini,kuota tersisa **2**.
   Belum ada submission dari sesi ini;prioritaskan kandidat beralasan dan gate
   tetap, tanpa menggunakan slot untuk null/probing.

Konsolidasi dan verifikasi structural sudah bergerak; lihat
[audit](../../docs/integration/perbandingan_repository.md),
[pemetaan](../../docs/integration/pemetaan_file.md),
[jurnal21](../04_riwayat_versi/jurnal_21okt_integrasi.md).

---

## Rekomendasi historis19Okt (bukan izin menjalankan probe/SSL sekarang)

Status rekomendasi lama (16 Okt dan 18 Okt) ada di `status_rekomendasi_lama.md`. Ide yang belum dicoba beserta rencana
implementasinya ada di `../10_ide_belum_dicoba/`.

## Aturan dulu (dari teguran user 18–19 Okt)
- **Pertanyaan user = jawab saja.** Jangan menjalankan script/run/submit sebelum user minta.
- **Semua run di lokal**; tidak ada push kernel Kaggle (kuota habis) kecuali user minta saat itu.
- **Jendela submit**: sampai 21 Okt 06:00 WIB hanya breakthrough (lolos validation_v2 dengan gain jelas > 0,002 vs v31,
  idealnya dengan rata-rata seed). Setelah 06:00 tanpa breakthrough, slot boleh untuk analisis (rencana di bawah).
- Jangan menyebut "final".
- **Semi-supervised diizinkan panitia (20 Okt)**, hanya data panitia; pipeline SSL wajib di dalam tiap outer fold
  (lihat `01_kompetisi_dan_aturan/aturan_kerja_user.md`).

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
