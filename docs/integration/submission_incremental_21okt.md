# Kandidat incremental untuk arahan submission 21 Oktober

Arahan user terbaru: "Later on coba submit aja kalau emang dapet yang bagus,
ga perlu tahan tahan submission", lalu "gua ingin ada submisi yang tersubmit
hari ini". Ini mengizinkan satu kandidat menjanjikan dengan kenaikan kecil;
syarat lama >0,003/P>=0,9 tetap dipakai untuk menyebut **breakthrough**.
Tidak mengarang forecast >=0,570 untuk meloloskan kandidat.

Resep produksi yang diperiksa: enam GBDT historis masing-masing 1/18,
Causilo seed42 1/3, dan Kumo fixed-union tiga seed 1/3. Kumo adalah mean
15 context; tiap context memakai mean rank primary/union 50:50. Tidak ada
grid bobot, penyesuaian per kelompok atau pemilihan dari public. Ini satu
penggantian komponen dari v31, menjaga kontribusi pohon yang terbukti berguna.

Dicatat sebelum evaluasi hybrid lengkap, sesudah melihat 11/15 fit Kumo;
bukan klaim seluruh riset bebas seleksi. Hasil155 asli dan gate breakthrough
tidak diubah. Semua15fit dan audit provenance tetap wajib. Evaluasi203
membandingkan hybrid dengan resep v31 yang tepat dan hybrid kontrol yang
memakai matched primary Kumo tiga seed. Urutan ID/label diperiksa, tree family
tetap mean enam percentile rank, tanpa rank kedua yang mengubah resep v31.

Kriteria incremental: delta positif dan mayoritas fold seed42 terhadap kedua
pembanding hybrid, bootstrap baris-customer P(delta>0)>=0,60 terhadap keduanya.
Angka ini kriteria keputusan praktis untuk arahan baru, **bukan** bukti kuat.
Seed dirata-rata dahulu, bukan diperlakukan sebagai customer independen.
Tidak memilih bobot berdasarkan hasil ini; bila gagal, jangan melabelinya
layak atau memakai slot untuk diagnosis. Test hanya diprediksi setelah audit.

Sebelum submit: simpan forecast berjangkar public v31 historis0,56405,
delta lokal dan noise subset9k; nyatakan kemungkinan transfer buruk.
Notebook harus memuat pipeline dan diperiksa lokal. Satu job berat, pacing
parity terverifikasi,56kcontext/e4/chunk128/allocator0,7 tetap.

Pemulihan21:27: file dokumen ini ditemukan kosong sesudah reboot21:23.
Teks asli yang sudah ditulis sebelum evaluasi hybrid dipulihkan dari riwayat
tool dan disinkronkan ke disk.203 belum pernah dijalankan. CSV193 pada mode
incremental adalah **komponen Kumo**, bukan hybrid yang boleh dikirim;
hybrid harus dibangun dan diaudit terhadap formula di atas secara terpisah.
