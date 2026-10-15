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
- Batas submission per hari: dari pola chat (13 Okt "sisa 1 submission lagi" setelah dua submit, 14 Okt "last submission of the day" untuk yang ketiga), **kelihatannya 3 per hari**. Cek ulang di halaman Kaggle.

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
