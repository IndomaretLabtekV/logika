# CPU union dan pseudo-label lunak setelah gangguan GPU

21Okt21:30, sebelum run204. GPU fixed-union masih11/15fit; pacing dan
inhibitor sesi tidak mencegah reboot berikutnya. Tidak mengubah atau
menganggap lengkap hasil tersebut. Satu jobCPU dibatasi250%/threads4,
affinity0–11/NI10/memory20G. Data panitia saja; semua konteks SSL diizinkan
arahan user terbaru. Tidak menulis atau memakai label test sebenarnya.

Hipotesis fitur: poolunion utama+teman mungkin lebih cocok untuk GBDT
dibanding Kumo104kolom. HipotesisSSL terpisah: confidence-tail soft labels
dengan prior kelas terjaga menghindari ketimpangan pseudo lama. Audit198
menemukan6,11%pseudo positif vs13,58%real; itu belum bukti sebab penurunan.

Satu konfigurasiLightGBM v.LGB, deterministicCPU4threads, early stopping
AUC di inner48k/8k, maksimal3000round/200patience; refitceil(best*1,15).
Main baseline memakai seluruh fiturutama; union menambah1000numerikteman,
empatkategori sekali. Referencefitoutertrain dari166 yang sudah diaudit;
tidak memakai shortlist90historis/seleksi dari outer. Vocabulary kategori
dipasang pada innerfit atauoutertrain sesuai model. Tidak ada pencarian
hyperparameter, feature subset, bobot atauthreshold di outer/public.

SSL: teacherunion48k memprediksi30ktest; pilih maksimal2000baris dari
ekor probabilitas, jumlah high-tailround(2000*priorreal) dan low-tail sisanya,
tanpa overlap. Target student adalah probabilitas teacher, objective
cross_entropy, bukan label keras. Bobot dua kelompok diatur agar rerata
target lunak berbobot sama dengan prior real; total bobot pseudo500.
Jika prior real tidak diapit mean dua ekor atau class tidak lengkap,
SSL tidak layak. Semua realrows tetap. Student memakai roundrefitteacher
yang sama. Jika KS innerstudent lebih besar dari teacher, aktifkanSSL;
jika tidak, fallbackunionteacher. Pilihan dibekukan sebelum outerprediksi.
Refitteacher56k dan bentuk pseudo ulang dari teacher56k, refitstudent
dengan policy terpilih tanpa menggunakan label outer. Model uniontanpaSSL
tetap disimpan untuk memisahkan manfaatfitur dariSSL.

Screeningseed42fold0; teruskan5fold×seed42/7/2026 jika innerunion atau
SSL mengungguli mainbaseline. Jangan mengubah policy setelah melihatouter.
Prediksi outer dan30ktest dibuat bersamaan, model/ID/label/hash disimpan.
Resep submissionhipotesis tetap:0,75v31+0,25rankunionpolicy. Pembanding
matched:0,75v31+0,25rankmainLightGBM. V31 memakai6tree1/18,Causilo1/3,
Kumo56k1/3 yang artefak testnya telah tersedia; tidak mengganti Kumo70k.
Bobot tetap ini tidak dicari dariouter. EvaluasiKS pooled,5fold,3seed,
pairedcustomerbootstrap setelahmeanseed. Bukti breakthrough tetap>.003,
mayoritasfold/P>=.9; keputusan incremental sesuai arahanbaru memerlukan
delta>0/mayoritasfold/P>=.60 terhadapv31danmatched sebelum submission.
Forecast/CSV/notebook/reproduksi lokal wajib. Bila hasilnya tidak mendukung,
jangan menyebutnya kandidatbagus hanya agar ada file yang dikirim.
