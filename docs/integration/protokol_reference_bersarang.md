# Protokol berikutnya: normalisasi harga yang bersarang

## Kelanjutan 23 Oktober 2026

Instruksi baru mengharuskan reference populasi dipasang pada fitting partition.
Gerbang submission sekarang pooled KS OOF lengkap 70.000 baris >=0,53000 dengan
audit bersih; syarat breakthrough/public forecast historis bukan gerbang aktif.

`features.v5_features(..., reference_customers=fit_customer_ids)` kini memfilter
populasi reference tanpa menghapus event milik query dari agregasi query sendiri.
Default tanpa argumen hanya untuk reproduksi resep historis, bukan run induktif
baru. Smoke sintetis membuktikan perubahan harga query tidak lagi mengubah fitur
context, sementara fitur query sendiri tetap berubah. Default historis tetap
menghasilkan angka yang sama pada fixture. Integrasi runner dan audit data nyata
belum selesai; perubahan ini belum menghasilkan skor model baru.


Pembaruan user setelah protokol ditulis: SSL/pseudo-label data panitia kembali
diizinkan. Train-only reference tetap menjadi kontrol untuk eksperimen
transduktif. Bukan prasyarat untuk melarang semua penggunaan covariate tak berlabel;
policy yang memakai covariate itu harus dinyatakan, diaudit dan dibandingkan adil.
Modul kontrol ini belum diimplementasikan saat steering tersebut diterima.

Pembaruanimplementasi21Okt: kontrol V3/V4/V5 **teman** sudah dibuat di
`eda/teammate_feature_adapter.py` dan runnerfeature-only166;7checkssintetis165
lolos. Fullbuilddatanyata belum dijalankan. Ini bukan penggantianreference`pp_*`
utama yang dijabarkan di bawah. KebijakanSSL terbaru mengatasi larangan awal
fittingtest;referencefit-only di sini dipakai sebagai kontrolinduktif yang adil.

Dicatat21Okt sebelum melihat skor model untuk perubahan ini. Bukti mekanistis:
audit162 menemukan30,0570%–30,1146% event reference harga legacy berasal dari
customer test. Reference teman memakai semua customer train sebelum outer split.

## Hipotesis dan pembanding

Pemasangan reference hanya dari fitting partition diperlukan untuk mematuhi
larangan fitting distribusi test dan menjaga independensi evaluasi. **Belum ada
alasan menyimpulkan ini akan menaikkan KS** atau menjelaskan gap public0,010.
Hipotesis kualitas terpisah: kategori native/fitur90 teman dapat memperbaiki
geometri input setelah reference dibuat bersih.

Baseline historis Kumo/v31 tetap disimpan byte-identik. Baseline matched untuk
uji kualitas baru harus memakai fitur/preprocessing bersih yang sama; jangan
menjual selisih akibat perubahan fitting population sebagai gain fitur VIP.

## Kontrak implementasi

1. Pisahkan fit reference dari transform application: reference hanya memakai
   IDcustomer dalam fitting partition,tidak ada IDtest/outer-query.
2. Pertahankan window temporal utama: eventreference sebelum hari pertama bulan
   keputusan aplikasi; setiap eventquery juga sebelum cutoff customer sendiri.
   Jangan mengganti window bersamaan dengan filter population.
3. Inner-fit membangun reference untuk context,search,confirmation; outer refit
   hanya pada outer-train sesudah policy dipilih. Tidak memakai target untuk
   estimasi mean/std/median/priorcovariate.
4. Fitur utama nonprice dan bentuk agregasi percustomer tetap. Kode tambahan
   mengembalikan frame ber-ID untuk mengganti `pp_*`,bukan menimpa cache historis.
5. Audit IDcoverage,sampel kedua kelas,dtype,missing,cutoff,duplikasi. Uji dengan
   fixture bahwa perubahan covariate query/test tidak mengubah fitted reference
   atau fitur context; perubahan histori query sendiri tetap boleh mengubah
   fitur query. Uji ekuivalensi formula pada data sintetis tanpa fittingtestreal.
6. Catat SHAcode/fitIDs/raw source/cache,seed,checkpoint dan konfigurasi. Jangan
   mengubah152 yang sedang berjalan sehingga checkpoint tetap dapat di-resume.

## Evaluasi dan keputusan

Screeningfold0: reference bersih terlebih dahulu,kemudian rawvsnativekategori
matched104kolom,ctx/e4/generator/chunk identik; selection hanya inner. Raw
referencelegacy hanya diagnostik historis dan tidak boleh jadi alasan promosi.
Kandidat menjanjikan baru5fold×3seed. Pooled/perfoldKS,variasi seed,bootstrap
baris berpasangan setelah rata-rata seed. Gerbang penuh user tetap berlaku.
Tidak ada run baru dari protokol ini yang sudah dijalankan pada saat pencatatan.
