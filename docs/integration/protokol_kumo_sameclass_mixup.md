# Augmentasi context dari training berlabel

Disusun 21 Oktober setelah retrieval183 selesai, sebelum membuat sampel atau
menilai kandidat baru. Baseline adalah utama_raw173, context48k/e4, search4k
dan confirmation4k. Tidak memakai baris test sebagai parent maupun pseudo-label.

Motivasi: [VIP-COpv2§4.2](https://arxiv.org/html/2605.12904v2) menguji augmentasi
training bersama context selection. Teks paper mencampur pasangan berlabel,
sedangkan jurnal20Okt mencatat implementasi publik same-class Beta(0,2;0,2).
Endpoint kode saat ini mengembalikan403, sehingga jangan menyebut eksperimen ini
reproduksi persis paper/kode. Ini adaptasi supervised mixup untuk Kumo tanpa
optimasi VIP atas baris sintetis. Paper tidak membuktikan gain KS kompetisi ini.

Hipotesis: interpolasi dalam kelas dan kelompok kategori yang sama menambah
variasi context sambil mempertahankan prior kelas. Pseudo-testSSL156 berbeda:
teacher-nya menghasilkan label semu dan context-nya didominasi negatif. Semua
sampel baru di sini berasal dari dua baris inner-fit yang labelnya diketahui.
Label hasil interpolasi tetap asumsi augmentasi, bukan ground truth baru.

Konfigurasi sebelum run:

- Semua48k real context tetap. Kolom100numerik+4kategori raw sama baseline,
  selector dipasang pada real inner-fit sebelum augmentasi.
- Cap sintetis2.000 dan8.000. Anchor dipilih tanpa pengembalian, dengan jumlah
  positif dibulatkan menurut prevalensi real context; bukan class balancing.
  RNG=100×seed+fold+701+cap. Tidak memilih cap dari outer atau skor public.
- Donor hanya inner-fit, label sama, keempat kategori aplikasi sama, dan status
  ada riwayat pembayaran sama (`rp_frac_dpd1` tidak missing). Bila sel hanya satu
  anggota, donor=anchor; jumlah clone dilaporkan, bukan disembunyikan.
- Lambda~Beta(0,2;0,2). Numerik dengan kedua nilai tersedia diinterpolasi;
  bila donor missing atau anchor missing, nilai/missingness anchor dipertahankan.
  Kolom biner menurut real-fit {0,1} disalin dari anchor. Kategori disalin dari
  anchor, sehingga tidak muncul kategori baru. Ini menjaga mask, tetapi tidak
  menjamin seluruh rasio/count sintetis sesuai customer nyata; audit sebagai
  regularisasi numerik, bukan aplikasi/pinjaman yang benar-benar diamati.
- Simpan parent IDs, label, lambda, prefix ID sintetis yang unik, kolom biner,
  hash hasil dan konfigurasi. Parent/query/outer/test disjoint. Jangan menghitung
  sampel sintetis sebagai observasi independen saat bootstrap evaluasi.

Kontrol: context48k ditambah8k salinan anchor, search saja dan tidak boleh menjadi
calon promosi. Ini membedakan manfaat interpolasi dari pengulangan/ukuran context.
Calon adalah baseline, mixup2k, mixup8k, serta mean probabilitas baseline+mixup8k
masing-masing0,5. Tidak mencari alpha atau lambda distribution dari query.

Semua fit Kumo large/e4/generator100×seed+fold, chunk128/allocator0,7. Search
memilih satu challenger, wajibΔKS>0,003; confirmation hanya baseline dan satu
terpilih, wajibΔ>0,003. Selain itu fallback. Confirmation bukan holdout baru antar
proyek; outer belum dibuka untuk keluarga input ini. Freeze sebelum outer.
Outer refit selector dan parent sampler pada56k real training; seluruh real
context tetap, cap sintetis yang terpilih tetap. Test bila dipromosikan memakai
ensemble konteks outer-train agar setiap context muat, dengan kebijakan sama
untuk baseline dan kandidat; jangan diam-diam mengganti ke konteks70k+augmentasi
yang belum diperiksa memori. Gate5fold×3seed/bootstrap/audit/forecast tetap.

Satu job berat lokal. Tanpa mengubah SDK/checkpoint, menaikkan allocator, data
eksternal, future leakage, probing atau submission sebelum gerbang user lolos.
