# Kompetisi

Sumber lengkap: `docs/context.md`. Ringkasan bagian yang relevan:

## Tugas
- Prediksi probabilitas sebuah pinjaman mencapai **DPD ≥ 90 dalam 12 bulan kalender** setelah origination. Ini *serious repayment difficulty*.
- Data lending konsumen Indonesia (sintetis). Mata uang IDR, zona waktu Asia/Jakarta.
- `decision_datetime` adalah **cutoff informasi**. Fitur hanya boleh memakai data yang tersedia sebelum cutoff ini.

## Metrik
- **KS two-sample**: `scipy.stats.ks_2samp(pred[y==1], pred[y==0]).statistic`.
- Hanya urutan (rank) yang berpengaruh, jadi transformasi monoton pada **satu** skor tidak mengubah KS. Lain halnya dengan transformasi per komponen sebelum blending, itu bisa mengubah hasil (lihat `06_model_dan_blending/blending_dan_stacking.md`).
- Helper lokal `ks(y, s)` di `eda/common.py` identik dengan scorer kompetisi.

## Submission
- Kolom `application_id,default_probability`. Ada 30.000 baris, setiap ID test muncul tepat sekali, nilainya finite dan di dalam [0, 1].
- Batas submission **3 per hari** (dikonfirmasi 16 Okt: CLI "2 submissions remaining today" setelah submit pertama hari itu, lalu 2 submit lagi berhasil).

## Penilaian babak penyisihan
| Komponen | Bobot |
|---|---|
| Public LB | 45% |
| Private LB | 45% |
| Notebook: reproducibility dan eksekusi | 10% |

- Bobot 45/45 **bukan** proporsi baris public/private. Proporsi split sebenarnya tidak diketahui.
- Notebook harus reproducible, jalan di environment yang ditentukan, dan *sesuai dengan pendekatan serta submission tim*.
- Format nama: `ParticipantNumber_TeamName_Notebook` (contoh `DSC0001_RandomSeed_Notebook`).
- **Deadline notebook: 30 Oktober 2026, 23:59 WIB.** Telat dipotong 1 poin per menit.
- 8 tim teratas lolos ke final (offline di UI Depok, presentasi dan laporan).

## Larangan penting (rule 3 dan 11)
- Data eksternal dilarang. Hanya data panitia yang boleh dipakai.
- Leakage dari masa depan atau periode target dilarang.
- **Manipulasi atau upaya memanipulasi skor/leaderboard dilarang.** Ide user di awal ("bikin submission yang bikin tahu kesalahan test") adalah LB probing, jadi **ditolak** sepanjang chat dan diganti dengan diagnosa CV.
- Pretrained model (TabPFN dan sejenisnya) dipakai sepanjang lomba. Rules tidak eksplisit soal ini, tapi tidak melarang. Model-model ini tidak membawa data eksternal ke dalam training.

## Fakta dataset singkat
- 70.000 train dan 30.000 test, satu aplikasi per customer. Customer train dan test disjoint.
- Bad rate train 13,58% (9.507 positif). Stabil 12–15% per bulan.
- Train dan test diacak (interleaved) di bulan yang sama (Jan–Des 2025). PSI semua fitur di bawah 0,002, jadi **split-nya random**, dan stratified KFold adalah skema yang tepat.

## Posisi leaderboard (16 Okt)
- Top-1 public **0,57768** (tim "cupuu"). Cutoff top-8 sekitar **0,57005**.
- Tim kita "lagi mager lomba bentar": 0,56145 (v5 dan v8), peringkat sekitar 34.
- Akun Kaggle: `riantamaputra`. Slug kompetisi: `data-science-competition-logika-ui-2026`.

## Posisi leaderboard (18 Okt)
- Kita **#24** dengan 0,56405 (v31). Top-1 0,57768, #2 0,57530, #3 0,57290; tujuh tim di atas 0,570.
- Rentang 0,563–0,568 sangat padat. Cutoff top-8 sekitar 0,570.
- Penilaian (docs/context.md §6): public 45% + private 45% + notebook 10% (reproducibility). Proporsi baris public/private
  tidak diketahui; analisis noise memakai asumsi public = 30% test (9.000 baris).
- Riwayat lengkap submission: `04_riwayat_versi/ringkasan_skor.md`.

## Posisi leaderboard (19 Okt ~20:00 WIB)
- 118 tim. Kita **#33**, tim **"lagi cape lomba asli"** (athillazaidan, nicholaswisee, riantamaputra), 0,56405, 22 submit.
- Top-1 **Datadataan 0,57788** (19 submit), #2 cupuu 0,57768 (28), #3 Magang Bareng Afiq 0,57458, #4 0,57167, #5 0,57120,
  #6 0,57099, #7 0,57067, #8 BCC mau ke kopdes depok 0,57065, #9 HYPOMONĒ 0,57015; **9 tim > 0,570**; batas top-8 ~0,5707.
- 33 tim ≥ 0,564 → klaster 0,564–0,568 sangat padat.
- Nama tim di Kaggle harus sama dengan yang terdaftar di panitia (rule 1); nama notebook `ParticipantNumber_TeamName_Notebook`.
