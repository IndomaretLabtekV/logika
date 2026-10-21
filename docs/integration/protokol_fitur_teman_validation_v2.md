# Input dan pool fitur teman dalam validation_v2

Disusun 21 Oktober sebelum build data nyata 166 atau evaluasi model pada fitur
teman. Ini tahap riset berikutnya setelah VIP/SSL, bukan resep yang dipromosikan.
Semua sumber dan hasil historis tetap disimpan. Izin SSL terbaru tidak mewajibkan
semua eksperimen memakai test; eksperimen input ini tidak menambah pseudo-context.

## Hipotesis dan pembanding

Representasi kategori native dan pool agregasi teman mungkin memperbaiki input
Kumo dibanding top100 numerik utama dengan empat kategori sebagai kode float.
Petunjuknya adalah perbedaan implementasi yang ditemukan saat integrasi, bukan
bukti skor public milik tim lain akan tercapai oleh kode ini.

Pembanding utama: Kumo large cached, fitur legacy yang dipilih bersarang,
100 numerik + empat kategori, e4, seluruh real context di partition fitting,
generator100×seed+fold, chunk500. Prediksi baseline dihitung pada baris dan
anggaran yang sama dengan kandidat. Referensi historis v31 dilaporkan terpisah.

Amendemen eksekusi setelah laptop hang21Okt, sebelum run kualitas input:
chunk **128** dan allocatorGPU **0,7** untuk semua policy, mengikuti bukti
preflight170 dan amendemenSSLf142e08. Context/e4/split/gerbang tidak berubah.
Nilai chunk500 di paragraf atas merekam rancangan awal; baseline baru dihitung
dengan konfigurasi yang sama dengan kandidat. Ini bukan penyesuaian dari skor
native/fiturteman, karena skor tersebut belum tersedia.

Calon policy yang ditentukan sebelum run:

| Policy | Pool numerik | Empat kategori aplikasi | Maksud perbandingan |
|---|---|---|---|
| utama_raw | utama, seleksi100 di fitting | kode float | baseline matched |
| utama_native | sama persis dengan utama_raw | native categorical | isolasi encoding |
| teman_raw | pool teman, seleksi100 di fitting | kode float | perubahan keluarga input |
| teman_native | sama persis dengan teman_raw | native categorical | isolasi encoding pada pool teman |

Perbandingan keluarga input menggabungkan agregasi, nama/formula fitur, dan
reference harga/prior yang berbeda. Ini sengaja menguji pipeline alternatif
sebagai satu bundle; gain tidak boleh diatribusikan hanya pada normalisasi.
Policy utama memakai reference legacy transduktif yang telah dinyatakan di162.
Policy teman memakai reference fit-only. Kontrol reference utama yang fit-only
tetap eksperimen terpisah bila diperlukan untuk menjelaskan perbedaan.

## Build dan audit fitur

- Runner 166 membuat output di direktori eksperimen baru; sembilan CSV ditautkan ke
  data utama tanpa menggandakan dataset mentah. Manifest mengikat sumber, paket,
  data, seed/fold dan fitting IDs. Tidak menimpa cache/artifacts historis.
- Inner fitting IDs 48k ditentukan oleh valid2. Semua target nonfit dihapus sebelum
  V3/V4/V5. Formula asli dipanggil tanpa perubahan; reference customer fit-only.
- Simpan seluruh pool dan skema. Kode sumber menambahkan dua kategori histori:
  v2_prev_last_decision dan v2_ecom_last_payment_method. Keduanya dipertahankan
  dalam cache, tetapi **tidak otomatis menjadi input** empat policy di atas.
  Hanya empat kategori aplikasi dipakai agar encoding dapat diuji secara setara.
- Numerik 100 dipilih dari kolom numerik saja; jangan memaksa kategori histori
  menjadi float atau memasukkannya ke v.fm_select yang mengasumsikan empat
  kategori. Seleksi KS/Spearman mengikuti kebijakan baseline, pada fitting rows.
- Amendemen struktural sebelum skor model: prepare-only173 pertama berhenti
  karena shortlist200 selektor asli menghasilkan hanya71 numerik teman (+4
  kategori), dibanding100 numerik utama. Belum ada inferensi/KS keluarga input.
  Aturan bersama kini: mulai dari selektor asli n=100; bila kurang100, gandakan
  shortlist dengan panggilan selektor asli n=200,400,... sampai seluruh pool;
  ambil100 numerik pertama dengan ranking KS dan filter |Spearman|<0,95 yang
  sama. Gagal eksplisit bila pool penuh tidak mencukupi. Terapkan pada kedua
  keluarga dan outer refit. Pool yang sudah menghasilkan100 tetap persis sama.
  Ini menjaga jumlah kolom104 pada perbandingan, bukan memilih shortlist dari
  skor query. Simpan jumlah kolom legacy dan kesamaan baseline dalam input_audit.
- Daftar 90 historis hanya pemeriksaan reproduksi dan metadata. Daftar itu pernah
  dipilih dari label global; jangan menjadikannya seleksi bersarang baru.
- Audit100kID/customer unik, cakupanID terhadap data utama, contoh kedua kelas,
  dtype, missingness, infinity, cutoff dan pemisahan target.165 sudah menguji
  invariansi reference terhadap label/covariate query pada fixture; itu belum
  membuktikan fullbuild benar atau pipeline meningkatkan kualitas.

## Seleksi, evaluasi, dan keputusan

1. Screening seed42 fold0. Inner48k real context, search4k, confirmation4k yang
   saling lepas. Keempat policy e4 dan context identik. Vocab/seleksi/prior
   dipasang hanya pada fitting partition. Pilih satu challenger terbaik di search.
2. Challenger harus mengungguli baseline >0,003 pada search dan confirmation
   terpisah; selain itu baseline dipertahankan. Simpan pilihan sebelum outer.
3. Outer56k: build ulang reference dan seleksi fitur dari outer-train; vocab
   dipasang kembali di outer-train. Evaluasi14k outer hanya setelah policy beku.
   Jangan memilih policy baru dari hasil outer atau angka public.
4. Kandidat menjanjikan diteruskan5fold×seed42/7/2026. Laporkan pooledKS,
   perfoldKS, variasi seed, paired bootstrap baris setelah rata-rata seed.
   Gerbang breakthrough user tetap: gain>0,003, mayoritas fold, P≥0,9, audit
   bersih dan alasan forecast public≥0,570 sebelum boleh submit.

Belum ada quality run dari protokol ini. Implementasi runner model dan bukti
parity harus disiapkan sebelum mengeksekusi tahap seleksi. Satu job berat lokal;
tidak menjalankannya bersamaan dengan VIP, SSL atau build166.
