# Geometri input uang dan batas100 fitur Kumo

21Okt setelah180 null; sebelum inferensi geometri. Audit153/157 dan rencana154
sudah menyebut dua hipotesis ini, tetapi hanya smoke154 dijalankan sebelumnya.
Native/gain100 kini sudah diuji dalam173/177, tidak diulang sebagai fit baru.

Hipotesis: signed-log kolom uang mengubah geometri input ke estimator yang tidak
murni rank-based; atau batas100 univariat membuang informasi yang dibutuhkan
model. **Bukan** klaim kehilangan presisi uang: audit157 menemukanΔKS uang
akibatfloat32 tepat0. Kumo sudah memproses Identity/Power/Robust/RankGaussian
secara internal, sehingga manfaat log eksternal belum tentu ada.

Baseline utama_raw173,query search4k+confirmation4k,realcontext48k,e4,generator4200,
chunk128/allocator0,7. Tambahkan utama_native173 sebagai single pembanding yang
sudah ada, tanpa fitulang. Kandidat baru:

- money_raw:100numerik+4kategori kode yang sama baseline; signed-log hanya
  daftar42kolom uang yang diaudit153, diinterseksi kolom terpilih (12di foldini).
- money_native:kolom dan transformasi sama, empatkategori native. Bandingkan
  juga dengan utama_native agar pengaruh transformasi dalam encoding jelas.
- all_raw:seluruh228numerik+4kategori utama,raw,tidak memilih kolom dari query.

Signed-log=sign(x)*log1p(abs(x)),NaN tetapNaN. Daftar unit moneter bukan hasil
pencarian label baru; daftar/metadata/source/data diikatsha. Vocab tetapfit-only.
Outerrefit ranking100/vocab dilakukan di outertrain; transformasi formula tetap.

Sebelum quality all_raw, jalankan preflight context48k/e4,query128,232fitur,
chunk128/allocator0,7 pada jobterpisah dengan telemetri durabel. Tidak menilaiKS
atau memakai outer. JikaOOM/terputus, catat keterbatasan teknis dan lewati all_raw
pada perangkat ini; jangan menaikkan allocator,mengecilkan realcontext,atau
menyebutnya null kualitas. Dua money policy tetap100+4 dan dapat diteruskan.

Search memilih satu dari baseline/nativecache dan kandidat feasible di atas.
HarusΔKS>0,003 terhadap utama_raw,confirmation hanya baseline+1terpilih dan
harusΔ>0,003. Fallback baseline selainitu;tidak memilih ulang dari confirmation.
Split inner digunakan lagi, bukan holdout baru antarproyek. Bekukan sebelumouter.
Jika menjanjikan, prosedur per fold5×seed42/7/2026 dan seluruh gate user berlaku;
tanpa bukti lokal kuat dan forecastpublic≥0,570 jangan submit.

Implementasi baru181 dan helper cache terpisah menjaga source173/177 yang sudah
dipakai dalam manifest. Jangan mengubah source/SDK/checkpoint historis diam-diam.
