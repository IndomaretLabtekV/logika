# Hasil audit v6 dan EDA lanjutan

Audit 14 Oktober 2026. Public v6 **0.56123** berasal dari laporan pengguna. Public v5 dan konfirmasi nama file yang disubmit belum tersedia; besar penurunan public v5→v6 belum bisa dihitung. Target mengalahkan 0.57059 adalah target eksperimen, bukan hasil yang sudah terbukti.

## Keputusan

**V6 belum terbukti lebih buruk secara generalisasi, dan belum terbukti lebih baik secara stabil. Jangan rollback seluruh preprocessing atau mengganti semua model berdasarkan satu public score.** Penyebab yang paling didukung bukti adalah margin OOF sangat kecil, pemilihan bobot blend yang sensitif, dan variasi sampel leaderboard. Itu penjelasan yang masuk akal, bukan identifikasi sebab penurunan test secara pasti.

Tidak ditemukan kerusakan CSV submission: rekonstruksi dari prediksi test tersimpan dan bobot notebook cocok berdasarkan application_id. Tidak ditemukan duplikasi identitas pada audit yang dijalankan. Beberapa perbaikan semantik fitur tersedia, tetapi belum ada bukti bahwa tambahan tersebut mengalahkan v6.

## 1. Model mana yang sebenarnya turun?

| Komponen | OOF KS v5 | OOF KS v6 |
|---|---:|---:|
| LGB | 0.524051 | 0.524558 |
| XGB | 0.523530 | 0.522370 |
| CatBoost | 0.523441 | 0.521066 |
| TabM PLE | 0.525735 | 0.524804 |
| TabM periodic | — | 0.525598 |
| TabPFN, representasi berubah | 0.491191 | 0.518564 |
| Selected blend | **0.527424** | **0.528334** |

Blend AUC naik 0.846151→0.846832. Jadi tidak benar semua komponen memburuk. TabPFN naik besar setelah beralih dari DFS ke fitur manual; perbandingan ini **tidak mengisolasi arsitektur**. Notebook v6 menjalankan TabPFN pada feature matrix, bukan eksperimen native TabPFN-Rel end-to-end.

Delta blend +0.000910; 400 bootstrap berpasangan memberi interval persentil 95% **[-0.001596, +0.003049]**. Ini resampling prediksi yang sudah dipilih, bukan interval yang mencakup retraining dan seluruh proses seleksi.

Simulasi subset berukuran total 30.000 dari OOF, dengan share public asumsi 30%/50%, menghasilkan v6 lebih rendah daripada v5 pada **37.25%/33.00%** percobaan. Persentil 95 selisih absolut public-private v6 **0.02933/0.02813**. Ini ilustrasi sampling di distribusi OOF, bukan peluang resmi atau jaminan gap private. Label test dan split leaderboard sebenarnya tidak diketahui.

Segmen tanpa histori: **0.500427→0.499416**; dengan histori: **0.535460→0.537107**. Variasi bulanan v6 sekitar 0.500–0.572. Grafik tersebut adalah slicing random OOF, **bukan** validasi training masa lalu→masa depan.

## 2. Masalah pada keputusan ensemble dan eksekusi notebook

- Bobot global TabPFN v6 **20%**, tetapi optimasi pada dua separuh OOF memilih sekitar **3.33% dan 6.67%**. Bobot model lain juga berubah. Ini bukti sensitivitas pemilihan, bukan bukti TabPFN merusak prediksi.
- Dari log notebook, greedy blend yang dievaluasi pada separuh seberang sekitar **0.5263**, sedangkan campuran tetap 80% trees + 20% TabM sekitar **0.5260**. Margin sangat kecil untuk membenarkan kompleksitas pemilihan bobot.
- Menghapus satu komponen tanpa optimasi ulang tidak meningkatkan full-OOF blend: menghapus TabPFN menghasilkan 0.526917. Jangan menyimpulkan bobot 0% pasti lebih baik dari public yang turun.
- Model trees sangat berkorelasi; dua TabM punya korelasi ranking **0.988818**. Banyak model tidak otomatis berarti banyak informasi baru.
- EXAONE **berhasil menjalankan probe** 3.000 context/3.000 query, sekitar 42 detik dan query KS 0.5242. Model kemudian dilewati karena estimasi context 4.731 berada di bawah batas minimum buatan 10.000. Ini bukan kegagalan OOM pada probe. Query KS kecil itu tidak sebanding langsung dengan OOF 70.000.
- Estimasi waktu quadratic dari satu probe mencampur overhead dan biaya inference. Untuk Kaggle, ukur biaya sesudah warm-up pada beberapa ukuran context sebelum memutuskan skip; 10.000 bukan batas kualitas yang telah dibuktikan.
- Seleksi fitur FM menggunakan korelasi; pasangan yang menghasilkan NaN perlu ditangani eksplisit agar fitur tidak terbuang hanya karena overlap/missingness. Jangan mengganti semua NaN menjadi korelasi nol tanpa memeriksa coverage.
- Beberapa perubahan dilakukan bersama: reference harga, preprocessing neural, representasi TabPFN, periodic TabM, dan bobot. Tidak ada eksperimen terkontrol yang memungkinkan menyatakan salah satu perubahan menyebabkan penurunan public.

## 3. Preprocessing harga: sudah diuji, tidak rusak

V6 memakai referensi harga kumulatif dari bulan sebelumnya, tetapi mencakup customer train+test. Ini referensi unsupervised transduktif, **bukan kebocoran target**, namun tidak sepenuhnya fold-local.

Saya bangun ulang referensi dengan hanya customer training pada satu fold diagnostik. Pada 14.001 baris validasi:

- KS fitur harga v6 **0.382581**, fold-local **0.382663**.
- Korelasi ranking **0.99999838**; median perubahan absolut **0.001109**, p99 **0.002608**.
- Perturbasi customer validasi tidak mengubah reference; mean/std satu reference diperiksa ulang secara independen dan cocok.

Grafik sebelum/sesudah hampir tepat diagonal; distribusi kedua kelas tetap terjaga. Tidak ada bukti transformasi menghancurkan sinyal harga. Ini **KS satu fitur**, bukan hasil training ulang. Refit reference dalam setiap fold tetap lebih bersih untuk evaluasi, tetapi tidak layak dijanjikan sebagai peningkatan besar.

## 4. Koreksi exposure cicilan dan provider biro

Dari 1.361.341 cicilan, yang sudah punya kesempatan mencapai keterlambatan 30/60/90 hari masing-masing **1.280.311 / 1.198.742 / 1.117.163**. Tidak ditemukan max DPD melampaui kesempatan pada ketiga threshold tersebut. Snapshot DPD tidak dibackdate ke tanggal jatuh tempo.

Saya buat rasio late hanya di antara cicilan yang cukup matang, beserta jumlah exposure dan late. Customer tanpa exposure tetap null. Pada training, 46.312 customer punya histori repayment; hanya 44.459 punya exposure 90 hari. Ini contoh penting mengapa tidak teramati tidak sama dengan nol.

Ranking raw vs corrected rate 90 hari masih **0.999719**. Nilai rasio berubah, tetapi urutan hampir sama: grafik menunjukkan massa sangat besar di nol. KS observed-only raw90 **0.330637** dan mature90 **0.348749**, tetapi subset observasinya berbeda; **jangan klaim delta itu sebagai gain berpasangan**.

Fitur `bh_score_delta` v6 mengurangi mean skor current lintas-provider dengan satu skor histori terakhir. Saya buat alternatif delta current–historical di provider yang sama, lalu agregasi. Korelasi dengan fitur lama **0.657579**: definisinya memang berubah cukup banyak. Akan tetapi KS univariat matched delta hanya **0.036830**.

Total 15 kandidat diuji secara eksploratif pada dua separuh data, dikondisikan pada bin ranking v6, histori, dan bulan, dengan BH correction. **0/15 lolos q<0.05 di kedua separuh.** Matched-provider delta tampak menarik pada discovery (p=0.01345), tetapi confirmation p=0.44176. Keputusan: simpan untuk ablation kecil, jangan langsung menambah semua fitur ke pipeline utama.

Tes ini bukan pembuktian tidak ada sinyal: binning, sparsity, power, dan pemakaian OOF yang sudah melalui seleksi membatasi kesimpulan. Semua label training sudah pernah dipakai dalam pengembangan; separuh confirmation ini bukan holdout baru yang benar-benar untouched.

## 5. Struktur sintetis dan duplikasi

- **0** duplicate installment ID, account customer-month, item ID, profil penuh tanpa ID, dan feature-row penuh tanpa ID. **0** overlap customer train/test.
- Identitas `closing = opening + inflows - outflows` tepat pada **1.084.418** baris account. Tidak ada alasan memperbaiki saldo berdasarkan dugaan kesalahan aritmetika.
- Minimum balance nol pada **8.14%** baris dan tidak ada nilai negatif. Ini konsisten dengan batas bawah data; bukan bukti kesalahan yang harus diimputasi atau dibuang.
- **100%** item_total dan order_total berada pada grid 1.000. Income hampir tidak terkuantisasi seperti itu (sekitar 0.10% kelipatan 1.000). Jangan menerapkan rounding seragam pada seluruh kolom finansial atau menambahkan jitter untuk memecahkan ties.
- Ada **2.388 baris dengan konten item berulang**, tetapi item ID berbeda. Jangan hapus sebagai duplikat tanpa bukti bahwa transaksi memang tercatat dua kali.
- Penjumlahan order_total di setiap baris item menggelembungkan total **1.739704×**. Agregasi yang benar menghitung total sekali per order; semua order konsisten owner/total dan jumlah item_total sama dengan order_total.

Temuan order ini juga ada pada EDA sebelumnya; bukan bug baru yang otomatis menjelaskan penurunan v6. Artefak baru menunjukkan before/after dan sampel kedua kelas. Sifat sintetis tidak cukup untuk menyimpulkan rumus pembentuk target atau batas maksimum skor. Tidak dilakukan probing leaderboard atau rekonstruksi label test.

## 6. Eksperimen Kaggle berikutnya yang paling layak

1. **Pertahankan v6 sebagai baseline**, simpan application_id + fold + seed pada semua OOF/test, serta checkpoint/model settings dan waktu inference. Pastikan yang dibandingkan adalah file submission utama yang sama.
2. Dengan fold dan seed identik, isolasi satu perubahan per eksperimen: preprocessing v5/v6 dengan estimator tetap; TabM PLE/periodic dengan input sama; kemudian hanya reference fold-local. Jangan sekaligus mengganti fitur dan ensemble.
3. Bandingkan blend sederhana dengan greedy memakai bobot yang dipilih hanya pada inner/discovery folds. Lihat paired KS, kestabilan bobot, thin/history dan bulan. Selisih full-OOF yang sudah dituning bukan validasi independen. Untuk nested validation ketat, training estimator juga harus mengikuti outer split.
4. Ablation tambahan cukup dua kelompok: exposure DPD dan matched-provider delta. Sertakan denominator/coverage; jangan drop fitur lama sebelum perbandingan. Tidak ada dukungan saat ini untuk feature explosion.
5. Prioritas foundation model: pertahankan **TabPFN pada fitur manual** karena peningkatannya terukur; lalu **EXAONE bounded pilot** karena probe benar-benar berhasil. Native **TabPFN-Rel** adalah eksperimen representasi terpisah, bukan nama baru untuk model di fitur manual.

[TabPFN-Rel resmi](https://github.com/PriorLabs/tabpfn-rel) menggabungkan fitur relasional dan TabPFN; cek environment Python 3.11/3.12 serta pin versi/checkpoint. [TabPFN resmi](https://github.com/PriorLabs/TabPFN) mendokumentasikan keluarga 3.5; nama paket saja tidak membuktikan checkpoint yang dijalankan, jadi catat artefaknya.

[EXAONE resmi](https://github.com/LGAI-Research/EXAONE-Tabular) menyediakan pengaturan memory; hindari ketergantungan pada private attention attribute kalau konfigurasi publik mencukupi. [Causilo resmi](https://github.com/nums-ai/causilo) mensyaratkan Torch 2.13+, sedangkan log v6 memakai 2.11: perlu environment terpisah/kompatibel, bukan upgrade serampangan dalam run utama.

[Mitra-v2 model card](https://huggingface.co/autogluon/mitra-classifier-2) memuat recipe yang melibatkan fine-tuning/bagging, sehingga bukan drop-in murah tanpa budget tambahan. [RelArena](https://github.com/PriorLabs/relarena) membandingkan sistem relasional termasuk RT-PluRel; ranking benchmark tersebut bukan bukti KS LOGIKA atau kelayakan runtime T4x2. Causilo, Mitra-v2, RT-PluRel tetap kandidat sekunder sampai baseline, validasi, dan budget jelas.

## Batas hasil

Keempat grafik baru diperiksa secara visual; angka dan sampel preprocessing tersimpan. Tidak ada model, inference, notebook, submission, atau pencarian label test yang dijalankan di lokal. Tidak ada notebook baru atau perubahan notebook v6.

Belum ada val KS model yang memakai kandidat baru, tidak ada label private, dan tidak ada kepastian melewati 0.57059. EDA yang baik membantu memilih eksperimen dan menghindari kesalahan; preprocessing yang benar tidak menjamin model apa pun akan menang. Arah paling defensible saat ini adalah **stabilkan validasi dan pemilihan ensemble**, bukan menganggap data perlu dibersihkan lebih agresif.
