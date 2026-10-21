# Preflight gradient seluruh parameter Kumo dengan SGD

21 Oktober, setelah mixup185 selesai null, sebelum mencoba full-gradient SGD.
Ini pemeriksaan teknis dua langkah, bukan eksperimen KS dan bukan kandidat submit.

Hipotesis lanjutan yang sudah tercatat di ide_lain.md: row embedding belum pernah
diadaptasi pada kompetisi ini. ICL-only AdamW sudah null; full-parameter AdamW
sebelumnya OOM. SGD tanpa momentum tidak menyimpan dua tensor moment AdamW,
sehingga mungkin memungkinkan pengujian full-parameter adaptation di perangkat
ini. Jangan mengulang ICL-only atau menyebut penurunan training loss sebagai gain.

Konfigurasi engineering yang ditetapkan:

- Seed42/fold0 validation_v2; selector100 numerik+4kode kategori dipasang hanya
  pada48k inner-fit. Vocabulary juga fit-only. Tidak ada outer/queryvalidation
  rows di Tensor GPU atau loss. Episode-query adalah data inner-fit berlabel.
- Satu episode tetap: context1024 stratified dan128 query lain dari inner-fit,
  dipakai dua langkah supaya perubahan loss dapat diperiksa tanpa mengubah episode.
  Num_estimators1 untuk gradient episode. Generator4200 direset pada tiaplangkah.
- Semua parameter classification Kumo large trainable. SGD lr0,01,momentum0,
  weight_decay0,foreachFalse;AMPfloat16/GradScaler initialscale128,clipnorm1.
  Tidak membuat Adam optimizer. Episode1024/128 lebih kecil daripada run AdamW
  historis2048/512; jika muat, jangan mengatribusikan perbedaan hanya ke optimizer.
- Catat loss,finitegrad,parametercount/gradcount/norm menurut rowembedding dan
  ICL,perubahan setidaknya satu parameter rowembedding,optimizerstatecount,
  allocated/reserved/RSS/waktu. Positive probability wajib dicocokkan nama kelas1,
  bukan posisi kolom. Zero_grad(set_to_none=True) sebelum forward.
- Allocator0,7,cgroup16/20/1GiB/CPU400%,OMP4,affinity0–11,NI10. JikaOOM atau
  gradienttidakfinite,catat phase dan hentikan;tidak mengubahSDK/driver/clock.
  Tidak menulis weights fine-tuned ke Hub/cache atau merekam sebagai modelpromosi.

Tidak menggunakan label search/confirmation/outer untuk loss,stopping,LR atau
pilihan lain. Tidak menilai KS/forecast. Sebelum runkualitas penuh harus ada
protokol terpisah yang mengunci LR/steps dan nested confirmation;outer tetap
evaluasi setelah pilihanbeku. Jika preflight lolos,itu hanya berarti training
step muat dan gradient/updates teramati,sama sekali belum bukti generalisasi.
