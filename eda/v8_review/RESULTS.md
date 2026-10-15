# V8: kenapa belum breakthrough dan apa yang perlu diubah

Public **0.56145** dan top leaderboard **0.57493** berasal dari laporan pengguna. Selisih **0.01348** belum menjelaskan metode pemenang atau skor private. Audit dilakukan atas notebook lokal, exported run, seluruh OOF/test yang tersedia, cache fitur historis, dan tabel mentah yang relevan. Tidak ada model atau notebook dijalankan lokal.

## Keputusan

**V8 belum memberi bukti peningkatan yang kuat; penambahan historical training rows tidak bekerja sebagai pengganti data baru dari distribusi target yang sama.** Ada tiga hambatan yang konkret: historical domain/state tidak setara selected loans, beberapa model baru belum menghasilkan OOF lengkap, dan sebagian screening EDA sebelumnya terlalu kasar untuk fitur dengan banyak nol.

Jangan menambah model/fitur sekaligus dalam run panjang berikutnya. Prioritas: selesaikan eksperimen RealMLP dan EXAONE secara terpisah, jadikan training selected-only sebagai kontrol, serta perlakukan augmentation historis sebagai eksperimen domain transfer yang belum lolos. Tidak ada bukti dataset sudah mencapai ceiling, dan tidak ada transformasi baru di audit ini yang boleh diklaim pasti mengalahkan top leaderboard.

## 1. CSV yang sebenarnya dinilai

`notebook/v8.ipynb` dan `results/v8/v8_run.ipynb` **bukan source snapshot yang sama**. Notebook lokal sekarang mematikan TabICL dan menambahkan safety-net; exported run mengaktifkan TabICL, dengan eksekusi tercatat berakhir pada cell TabICL. RealMLP/EXAONE tidak memiliki OOF/test artifact. Tidak ada traceback yang cukup untuk memastikan apakah kernel mati karena GPU VRAM, host RAM, atau sebab lain.

CSV tidak cocok dengan formula safety-net 80% trees + 20% TabM. Saya menyelesaikan persamaan linear dari 16 kolom test ranks dan CSV aktual. Sistem full-rank, semua bobot nonnegatif, dan maksimum selisih rekonstruksi hanya **4.44e-16**. Ini bukan pencarian bobot memakai target; hanya memulihkan kombinasi yang sudah ada.

Bobot aktual terbesar:

| Komponen | Bobot |
|---|---:|
| TabM augmented | 20.67% |
| TabM base | 14.50% |
| Causilo | 13.67% |
| LGB augmented | 10.83% |
| XGB augmented | 9.33% |
| Semua augmented models, total | **54.83%** |

OOF dengan bobot aktual **KS 0.527025**, **AUC 0.847166**. Pembanding sebelumnya v7 KS **0.526992**, v6 **0.528334**, v5 **0.527424**. V8−v7 hanya **+0.000033**. Public yang tetap sekitar 0.561 tidak bertentangan dengan hasil lokal ini. AUC meningkat, tetapi objective kompetisinya KS.

Asal sesi yang menulis CSV final belum dibuktikan oleh exported run yang tersedia. Rekonstruksi ini memastikan isi dan komposisinya, bukan cara bobot dipilih. Simpan weights.json, OOF ber-ID, status setiap model, versi source dan file hash pada setiap run berikutnya.

## 2. Augmentation menambah volume, bukan otomatis informasi yang sesuai

| Model | Base KS | Augmented KS | Delta |
|---|---:|---:|---:|
| LGB | 0.523929 | 0.523953 | +0.000024 |
| LGB shallow | 0.524492 | 0.521853 | −0.002639 |
| XGB | 0.522370 | 0.523894 | +0.001524 |
| XGB shallow | 0.524057 | 0.523511 | −0.000547 |
| CatBoost | 0.521251 | 0.520932 | −0.000319 |
| CatBoost shallow | 0.520766 | 0.519798 | −0.000968 |
| TabM | 0.525669 | 0.525767 | +0.000099 |

Semua interval bootstrap berpasangan 95% dari 200 resampling per pasangan melintasi nol. Ini fixed-prediction uncertainty, belum termasuk retraining/selection. Tidak ada dasar menganggap augmentation sudah lolos hanya karena sebagian rata-rata fold naik.

Equal tree blend **0.523140→0.523128**. Thin-file **0.497793→0.493738**, history **0.532472→0.534792**. Jadi ada tradeoff antarsegmen yang masuk akal untuk data tambahan yang hanya berasal dari customer dengan historical loans.

Menghapus semua augmented members dari bobot actual lalu menormalkan sisanya menghasilkan KS **0.526671**, sedikit di bawah actual. Ini menunjukkan augmentation masih dapat memberi diversitas kecil. Bukan bukti bobot 0% pasti menang, dan bukan alasan mempertahankan bobot 54.83% tanpa evaluasi selection yang lebih ketat. Tidak ada submission alternatif dibuat dari ablation ini.

## 3. EDA utama: historical snapshots tidak identik dengan observasi masa lalu

Dataset menyediakan snapshot pembayaran pada **selected cutoff**. `last_payment_datetime` adalah alokasi terakhir sebelum cutoff, bukan seluruh ledger. Kode snapshot historis menetapkan `amount_paid=0` jika alokasi terakhir terjadi setelah T.

Secara informasi, langkah tersebut tidak selalu dapat dibenarkan. Contoh yang diuji di skrip:

- Ledger A: membayar 100 pada hari 40.
- Ledger B: membayar 40 pada hari 10, lalu 60 pada hari 40.
- Snapshot hari 50 sama: amount_paid=100, last_payment=hari 40. Tetapi pada T=hari 20 amount_paid adalah 0 vs 40.

Fixture ini memperlihatkan masalah identifiability, **bukan klaim ledger sintetis sebenarnya pasti mengikuti B**. Dari data aktual:

- **141.710** pasangan installment–historical snapshot dengan due_date sebelum T.
- **13.513** pasangan mempunyai alokasi terakhir setelah T.
- **8.026 dari 81.702 historical rows** terpengaruh setidaknya satu pasangan demikian.
- Ambiguitas terdapat pada 5.669/69.958 proxy-negative rows dan 2.357/11.744 proxy-positive rows; proporsinya berbeda antarkelas.

Current profiles juga tidak memiliki riwayat perubahan income/employment/dependents/housing. Mengurangi umur dan masa kerja tidak membuktikan current income valid pada T. Pada **10.913 historical rows**, current occupation tenure lebih pendek daripada jarak ke historical decision: pekerjaan sekarang bahkan tidak mencakup seluruh lookback yang dipakai. Completed age pada tanggal lama juga tidak bisa diperoleh secara tepat tanpa tanggal lahir; memakai floor elapsed years dapat meleset.

Identity test di selected cutoff dan perturbasi future rows hanya menguji perilaku kode terhadap kasus tertentu. Keduanya tidak membuktikan snapshot historis sama dengan keadaan nyata saat T. Memakai profil kini untuk auxiliary prediction historis mungkin masih berguna sebagai task terpisah, tetapi jangan menyebutnya as-of reconstruction yang exact. Temuan ini **tidak membuktikan kebocoran target selected loan**; masalahnya adalah semantik fitur dan perpindahan domain pada auxiliary training.

Keputusan: jangan memperbaiki jumlah pembayaran historis dengan tebakan tunggal lalu menganggap benar. Jika jalur historis diteruskan, gunakan hanya dated evidence yang dapat dipertanggungjawabkan, tandai/null-kan state yang tidak diketahui, atau pisahkan auxiliary task dari main selected-loan prediction. Belum dibuat historical feature matrix “bersih” palsu karena data tidak cukup untuk mengetahui semua state tersebut.

## 4. Domain shift dan ketergantungan customer terukur

Historical rows berjumlah **81.702**, tetapi hanya berasal dari **50.510 customer**; hingga tujuh pinjaman per customer. Tidak ada duplicate historical application_id. Repeated customers adalah observasi terkait, bukan duplikasi yang boleh dihapus sembarangan.

| Karakteristik | Selected | Historical |
|---|---:|---:|
| Tenor 12 bulan | Semua | 27.850/81.702 |
| Tenor lain | Tidak ada | 25.542 tenor 6m; 28.310 tenor 18m |
| Coverage pp_mean | 100% | **31.15%** |
| Coverage ec_accel_30d | 100% | **31.16%** |
| Coverage repayment fraction | 66.16% | **32.79%** |
| Loan purpose diketahui | Semua | Tidak ada |

Kurva bad rate dihitung dengan **batas bin yang sama**, bukan decile masing-masing populasi. Pada e-commerce acceleration, selected bad rate meningkat tajam dari sekitar 0.05 ke 0.28, sedangkan historical curve jauh lebih datar/nonmonotonik. Profile income/age historis juga memakai asumsi waktu yang berbeda. Ini alasan konkret mengapa menambah 81k baris tidak setara menggandakan training dari target distribution. Ini asosiasi, bukan bukti kausal bahwa satu keluarga fitur menyebabkan penurunan public.

Kode `train_rows` memasukkan **seluruh** historical rows pada setiap fold, termasuk history milik validation/test customers. Pada diagnostic fold 81015:

- 45.652 rows milik training customers.
- 11.537 rows milik validation customers.
- 24.513 rows milik test customers.

Label tersebut bukan target selected loan dan berasal dari data yang diberikan; jangan otomatis menyebutnya test-label leakage. Namun protokolnya transduktif dan berbeda dari training yang benar-benar mengecualikan semua baris customer validation. Simpan keduanya sebagai protokol berbeda; manifest yang dibuat memungkinkan ablation strict customer exclusion.

Bobot 0.5 per historical row memberi total bobot histori sampai 3.5 pada customer dengan tujuh pinjaman. Jika ingin melanjutkan transfer, uji bobot total per customer yang dinormalisasi dan subset tenor 12m sebagai eksperimen terpisah. Jangan bersamaan mengubah sampling, fitur dan arsitektur lalu mengatribusikan gain ke satu perubahan.

## 5. Preprocessing alternatif yang sudah dibuat dan diperiksa

Saya mengagregasi histori yang horizon 12 bulannya sudah berakhir menjadi **sembilan fitur pada selected cutoff**: jumlah mature loans, jumlah/fraction proxy-bad, outcome terakhir, usia outcome, subset tenor 12m, dan coverage flag.

Ini menghindari rekonstruksi profil/repayment pada tanggal lama dan tidak menambah training rows. Tetap memakai proxy `max_dpd>=90` dan `due+90<=origination+12mo` yang sudah ada; bukan rekonstruksi exact ledger/end-of-day loan trajectory. Missing fraction tetap null, count kosong nol dengan coverage terpisah.

Coverage mature history pada train **35.407/70.000**. Fraction historical bad punya observed-only KS **0.343925**, tetapi itu bukan model KS pada seluruh train. Korelasinya dengan repayment fraction **0.5743**. Angka tersebut menunjukkan sinyal univariat, belum menunjukkan informasi tambahan atas blend v8.

Sampel dari kedua kelas diperiksa. Customer tanpa mature history mempunyai null pada fraction/last outcome, sehingga tidak tersamar sebagai customer berhistori baik. Artefak `03_selected_history_features.parquet` berisi 100.000 aplikasi dengan ID + sembilan predictors, tanpa label kompetisi. Tidak dipromosikan sebagai fitur yang sudah terbukti meningkatkan model.

## 6. Koreksi penting pada screening EDA saya sebelumnya

Saya menemukan kelemahan nyata pada penggunaan bin kuantil dalam review sebelumnya. Untuk data 95 nol dan lima angka satu, quantile cutpoint menjadi nol; `searchsorted(..., side='right')` memasukkan **nol dan satu ke bin yang sama**. Test kemudian tidak lagi menguji perbedaan nol-versus-positif. Ini khususnya berbahaya untuk jumlah default, binary flags dan rasio dengan massa nol besar.

Saya memperbaiki binning di helper review v8: nilai diskret dengan sedikit level dipertahankan; nol struktural dipisahkan untuk fitur lainnya; missing terpisah. Fixture assert memastikan 0/1/missing tidak digabung. Helper lama tidak ditimpa agar angka laporan terdahulu tetap dapat dilacak; gunakan screening terkoreksi untuk keputusan baru.

Kemudian **30 kandidat** dari review v6/v7 dan fitur baru v8 diuji ulang terhadap score v8 yang sama. **22 kandidat** memperoleh lebih banyak bin. Setelah BH correction pada seluruh 30 kandidat, **0 lolos q<0.05 pada kedua separuh**.

Contoh matched-provider bureau delta memiliki p sekitar 0.0144/0.0119, tetapi q sekitar 0.216/0.358; menarik sebagai hipotesis kecil, bukan discovery terkonfirmasi. Historical bad fraction p confirmation sekitar 0.106 setelah nilai diskretnya dijaga. Tidak ada pembenaran untuk feature explosion atau klaim breakthrough dari tes ini.

Koreksi ini penting walaupun hasil akhir tetap negatif. Kesimpulan lama “tidak ada sinyal” pada fitur yang bin-nya kolaps tidak cukup didukung. Bahkan tes baru tetap eksploratif: label berulang kali dipakai, OOF/ensemble sudah dipilih, conditioning coarse dan sel kecil membatasi power. Tidak boleh menyimpulkan semua model/representasi sudah mencapai ceiling.

## 7. Model berikutnya: selesaikan eksperimen yang belum selesai

**Belum tersedia OOF/test EXAONE, RealMLP, TabICL atau periodic TabM di results/v8.** Karena itu v8 tidak membuktikan rekomendasi model tersebut sudah dicoba secara lengkap dan gagal menaikkan KS.

1. **RealMLP selected-only, satu eksperimen mandiri.** Sudah ada implementasi notebook; jangan menaruhnya sesudah model yang dapat mematikan kernel. Pakai preprocessing median/missing flags/vocabulary training-only dan fold yang sama. Simpan hasil tiap fold; tidak perlu mengulang 14 model base/aug sebelumnya.
2. **EXAONE 12k context, mandiri.** V7 probe 12k sudah berhasil. V8 sudah memperbaiki pengenalan error `cannot fit`, tetapi tidak ada bukti stage ini mencapai complete OOF. Ukur query workload nyata, memori dan waktu di satu GPU sebelum lima fold. [Dokumentasi EXAONE](https://github.com/LGAI-Research/EXAONE-Tabular) menyediakan memory-budget planning; context dan query merupakan dua batas berbeda.
3. **TabICLv2 baru setelah pilot kecil.** `whole_query=True` melewati FM_CHUNK sehingga query sekitar 44k dikirim sekaligus; mengecilkan context tidak mengecilkan query. Dua worker juga berbagi host memory. Kernel death tanpa trace belum membuktikan OOM, tetapi workload tersebut perlu dipisahkan dan diukur. Gunakan proses terisolasi, context/query kecil, satu GPU, catat RSS/VRAM; verifikasi chunking tidak mengubah semantik prediksi sebelum scale-up. [Repo resmi](https://github.com/soda-inria/tabicl) mendokumentasikan offloading dan inference configuration; offload CPU bukan otomatis aman jika RAM host habis.
4. **Causilo adalah kontrol FM yang sudah berhasil.** V8 typed-category Causilo KS **0.524066**, dibanding v7 **0.522647**. Ini perkembangan positif, meski tidak cukup mengatribusikan semua gain hanya ke category handling. Pertahankan environment sukses dan evaluasi kontribusi pada blend tanpa menambah search besar.
5. **TabPFN-Rel tetap eksperimen schema, bukan tombol breakthrough.** V7 native DFS masih kalah dari handcrafted TabPFN. [TabPFN-Rel resmi](https://github.com/PriorLabs/tabpfn-rel) adalah relational harness DFS + TabPFN 3.5 dengan task/temporal schema dan recipe bertanggal. Perbaiki lineage, categorical metadata, dan snapshot semantics dahulu; jangan menaikkan DFS depth atau jumlah estimator sekadar karena Elo benchmark.
6. **Mitra-v2/RT-PluRel cadangan.** [Mitra-v2 model card](https://huggingface.co/autogluon/mitra-classifier-2) mempunyai recipe fine-tuning/bagging; [RelArena](https://github.com/PriorLabs/relarena) membandingkan sistem relasional. Belum ada bukti dataset ini membutuhkan biaya tersebut. Jadwalkan hanya setelah dua kandidat yang belum selesai memberi hasil fair, bukan ditumpuk dalam satu notebook yang sama.

Koreksi orchestration ini lebih mendesak daripada mencari nama model baru. Satu run yang menyelesaikan satu pertanyaan lebih informatif daripada run panjang yang berhenti sebelum pertanyaan utama diuji.

## 8. Keputusan eksperimen dan public/private

Kontrol utama: selected-only features v6/v8, fold manifest tetap, model/seed tetap. Bandingkan kandidat satu per satu. Untuk historical transfer, pisahkan (a) own-history features pada selected rows, (b) clean dated-evidence auxiliary task, dan (c) same-customer/transductive vs strict customer exclusion. Jangan menjadikan augmentation penuh sebagai default sementara domain shift dan rekonstruksi belum selesai.

Pilih kebijakan ensemble pada bagian development/inner split; nilai pada outer fold. Simpan per-model per-fold predictions dan learning curves, bukan hanya rata-rata akhir. Pooled OOF KS tetap headline; laporkan subgroup thin/history dan waktu, paired delta, serta ketidakpastian selection. Bootstrap yang ada tidak menjamin private KS dan tidak mengubah label lama menjadi untouched holdout.

Top public 0.57493 mungkin mencerminkan metode lebih baik, variasi subset, atau keduanya. Tanpa prediksi/labelnya tidak dapat ditentukan. Jangan menerjemahkan gap 0.01348 menjadi “butuh fitur X” atau memakai submission untuk menebak label test. Audit ini tidak melakukan probing leaderboard atau membuat submission baru.

## Output dan batas

Lima skrip analisis beserta tabel, lima gambar, contoh preprocessing kedua kelas, manifest kualitas/split, dan sembilan candidate features tersedia di folder ini. Semua grafik diperiksa visual; missing bin pada domain plot dipisahkan agar tidak tampak sebagai titik lanjutan numerik.

Belum ada training/inference baru, val KS setelah fitur baru, atau evaluasi private. Tidak ada dasar menjanjikan breakthrough dari EDA saja. Temuan audit memberi alasan konkret untuk mengganti desain eksperimen, bukan sekadar mengulang saran “tambah model”.
