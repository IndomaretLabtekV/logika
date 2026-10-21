# Jalur dari validasi menuju submission 21 Oktober

Arahan terbaru user: lanjutkan pekerjaan, usahakan kenaikan dan submission hari
ini; tersisa dua slot. Larangan probing dan syarat breakthrough sebelumnya
tetap berlaku. Pseudo-labeling/SSL sudah diizinkan panitia menurut user.

1. Selesaikan satu run 191 yang aktif, 5 fold × seed 42/7/2026. Resep 189
   tetap: dua input Kumo, e4, context outer-train 56k, mean rank 50:50.
   Jangan mengubah resep berdasarkan hasil outer yang bertambah.
2. Audit prediksi, split, cache, hash kode/data/checkpoint, selector serta
   reference setiap fit. Evaluator 155 harus lolos syarat lokal lengkap;
   bootstrap menggunakan baris customer sesudah merata-ratakan seed.
3. Hanya bila layak dipromosikan, prediksi 30k test dengan setiap context
   56k dan selector yang sama dari 15 fit. Reference fitur teman memakai
   cache per-fold yang sama; tidak dipasang ulang ke test. Kedua konstituen
   dirank pada cohort test 30k, mean 50:50, kemudian mean 15 prediksi.
   Matched baseline menggunakan budget context/estimator/fold/seed yang sama.
4. Audit CSV mengikuti sample submission. Tulis perkiraan public beserta SD
   dari bukti historis yang relevan dan ketidakpastiannya. Gain CV tidak
   otomatis menjadi gain public; jangan memaksakan forecast ≥0,570.
5. Jika seluruh syarat terpenuhi, perbarui builder/notebook produksi,
   verifikasi reproduksi lokal, cek kuota terkini, lalu submit satu file
   dengan deskripsi resep yang jelas. Tinjau skor sebelum memakai slot kedua.

Saat validasi aktif, persiapan hanya pemeriksaan ringan dan penulisan kode
terpisah; tidak mengedit sumber yang terkunci atau membuka job GPU lain.
Semua fit berat memakai scope terbatas, affinity 0–11, NI aktual 10,
threads 4, allocator GPU 0,7 dan chunk 128. Hang sebelumnya diikuti force
shutdown manual menurut user; penyebab sistemnya belum terbukti.

Kaggle memakai 9k public /21k private. Rubrik lomba memberi bobot public
45% dan private45%; keduanya berbeda dari pembagian baris Kaggle. Kemenangan
public saja tidak cukup untuk menilai keandalan generalisasi.
