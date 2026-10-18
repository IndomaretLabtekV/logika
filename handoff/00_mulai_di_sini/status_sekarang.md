# Status per 18 Oktober 2026 (siang)

Ringkasan satu halaman. Detail tiap angka ada di file yang dirujuk; log mentah sesi 17–18 Okt ada di
`04_riwayat_versi/jurnal_17_18okt.md`.

## Posisi
| Item | Nilai |
|---|---|
| Public terbaik | **0,56405** (v31: 6 GBDT 1/3 + Causilo 1/3 + Kumo-Tabular large context penuh 1/3) |
| Peringkat | #24 (18 Okt). Top-1 0,57768, #2 0,57530, #3 0,57290; tujuh tim > 0,570; cutoff top-8 ~0,570 |
| Kurang ke target | ~0,006 ke top-8, ~0,0136 ke top-1 |
| Klaster public resep sekarang | v31 0,56405 · v32 0,56376 · v35 0,56298 · v34 0,56271 → rata-rata 0,5634, sd 0,0006 |
| Validasi | validation_v2 (`eda/valid2.py`), bersih dan nested. v31 = +0,00221 vs resep v8 bersih (P>0 0,93, 4/5 fold) |
| Notebook final | `notebook/build_final.py` → `notebook/final.ipynb`. Sudah dijalankan penuh di Kaggle (`riantamaputra/logika-final`, 4,3 jam) dan **mereproduksi v34** (Spearman 1,000000). Banner tim masih placeholder |
| Submit | 3/hari, reset 07:00 WIB. Kuota window 18 Okt (07:00 18 Okt – 07:00 19 Okt) **sudah habis** |
| Kuota GPU Kaggle | 30 jam/minggu; dipakai cukup banyak 17–18 Okt (v22–v33 + final). Cek sisa di UI sebelum push besar |
| Deadline notebook | **30 Okt 2026 23:59 WIB** (telat −1 poin/menit) |
| `/goal` | sudah di-clear user (18 Okt siang) |

## Apa yang sudah dikerjakan 17–18 Okt (urut)
1. **validation_v2** dibangun dan dipakai untuk semua kandidat (early stopping di inner 1/7, refit budget tetap,
   bobot blend cross-fit antar fold, paired bootstrap + menang fold). Lihat `07_validasi_dan_leaderboard/validation_v2.md`.
2. **v22** = rerun bersih resep v8 (6 GBDT × 3 seed, TabM 3 seed, Causilo). Di protokol bersih: Causilo terbaik
   (0,52583), TabM terlemah (0,52100) — kebalikan kesan v3–v9. Resep v8 bersih = 0,52426 (pembanding semua kandidat).
3. EDA baru `eda/100`–`111` (13 analisis). **Semua null untuk informasi baru**, tapi memberi peta penting: learning
   curve datar, model linear ≈ TabM ≈ GBDT, dua faktor laten + repayment, tidak ada drift, split acak.
4. **Model zoo TabArena** di-screen di fold 0: TabPFN-3.5, TabICLv2, RealMLP, AutoGluon (LGB/XGB/Cat/NN/FastAI/Linear),
   TabSwift, RF/ET/KNN, iLTM, LimiX-2, Mitra, TabDPT, Nori, TabFM, WOE-LR, LR rank-gauss. Hanya **NVIDIA
   Kumo-Tabular large** yang lolos.
5. **Kumo** dari context 20k (2 draw, 0,52484) ke **context penuh 56k (0,52665, model tunggal terbaik)**. Varians
   draw context ±0,005 dan acak estimator ±0,0017 adalah jebakan utama screening ICL.
6. **Submission 17–18 Okt** (6 buah, semua pra-registrasi): v31 0,56405 (terbaik), v31b ICL tanpa pohon 0,55709,
   v31c Kumo saja 0,56056, v32 pohon+Kumo 0,56376, v34 resep notebook final 0,56271, v35 lima keluarga 0,56298.
   Pelajaran: **blend tanpa GBDT jatuh di public** (−0,004 s.d. −0,007) walau CV setara → GBDT minimal 1/3. Varian
   setara-CV jatuh dalam satu klaster ±0,001 di public (noise statistik maksimum KS).
7. **Notebook final** dirakit (layout `ref/notebook/SKILL.md`), Kumo diganti "fit sekali dengan seluruh 70k train
   sebagai context" supaya muat < 12 jam di T4. Run Kaggle penuh lulus dan identik dengan v34.
8. Null lain: Causilo per segmen (v29), Kumo seed generator kedua (v33), TabFM (v27), bobot per segmen, scan harga
   relatif per subkelompok, neighbor gap, Kumo e8/f150/medium.
9. `eda/kumo_finetune.py` (fine-tune Kumo) ditulis; screening dihentikan karena KS baseline-nya 0,298 (harusnya
   ~0,526) → **bug belum ketemu**, lihat `09_langkah_berikutnya/rekomendasi.md` langkah 1.

## Kenapa stuck
- **Plafon informasi fitur.** Learning curve 80→100% datar (`eda/102`), 10-fold = 5-fold, model linear di fitur
  rank-gauss = TabM ≈ GBDT (`lr_q` 0,52119). Kapasitas model tidak menambah; data tambahan tidak menambah.
- **Tidak ada informasi baru yang tersisa di tabel** dari semua sudut yang sudah dicoba: residual 228 fitur vs OOF
  bersih (0 di atas null), 1.142 uji event-window (v9), scan harga relatif 30+ kandidat (maks chi2 5,4 < 9,4), set-encoder
  baris mentah (v14), DFS, utang eksternal bureau, shock makro per segmen.
- **Satu-satunya tuas yang terbukti** sejak v5 adalah keluarga model ICL (Kumo context penuh + Causilo) dicampur pohon:
  +0,0022 CV, +0,0026 public. Varian lain dari campuran itu tidak bisa dibedakan dari noise public.
- Gap ke top-1 (0,0136) ≈ 20 sd selisih public antar varian kita. Penjelasan yang konsisten dengan data: leaderboard
  0,563–0,568 padat, tujuh tim > 0,570 bisa campuran seleksi best-of-many + metode/informasi yang tidak kita temukan.
  Tidak bisa dibuktikan tanpa prediksi mereka. **Target > 0,57005 kemungkinan tidak tercapai dengan pendekatan sekarang.**

## Langkah berikutnya (detail: `09_langkah_berikutnya/rekomendasi.md`)
1. Lacak bug `kumo_finetune.py` (KS 0,298) — mulai dengan menjalankan ulang screening Kumo lama di fold 0 dan cek
   apakah masih 0,52–0,53. Kalau sudah tidak reproduce, masalahnya di environment (driver/torch/sdm), bukan script.
2. Kalau fine-tune Kumo valid dan menang di fold 0 (P>0 ≥ 0,9), full fold lalu blend 1:1:1 seperti v31.
3. Putuskan final: v34 (persis yang direproduksi notebook) atau v31 (public terbaik, butuh Kumo 5 fold = ~16 jam di T4,
   tidak muat 1 sesi). Selisih public 0,0013 masih noise. Rekomendasi: **v34** kecuali fine-tune menghasilkan gain nyata.
4. Isi banner tim (nomor peserta `DSCxxxx`, nama anggota) di `notebook/build_final.py`, pin versi paket, rename
   `ParticipantNumber_TeamName_Notebook`, jalankan ulang sekali di Kaggle, kumpulkan sebelum 30 Okt 23:59 WIB.

## File penting
| File | Isi |
|---|---|
| `results/v31/submission.csv` | public terbaik 0,56405 |
| `results/v34/submission.csv` | resep notebook final, public 0,56271 |
| `results/final_kaggle/submission.csv` | output run Kaggle notebook final (= v34) |
| `results/v22/v22/{gbdt,nn,fm}/` | OOF/test bersih resep v8 (dengan application_id, per seed) |
| `results/v28m/` | Kumo context penuh 5 fold (OOF/test) |
| `eda/valid2.py`, `eda/run_valid2.py`, `eda/blend2.py` | protokol, driver, blend cross-fit |
| `notebook/build_final.py` | builder notebook final |
