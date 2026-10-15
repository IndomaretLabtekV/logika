# Analisis v1–v7: penyebab stagnasi dan eksperimen berikutnya

Tanggal audit: 15 Oktober 2026. Seluruh analisis baru dijalankan CPU. Tidak ada model atau notebook yang dieksekusi. Public v6 0.56123 berasal dari laporan pengguna; public v5/v7 belum diberikan dalam sesi ini. Angka 0.57059 adalah target leaderboard yang sebelumnya disebut pengguna, bukan posisi leaderboard terkini yang diverifikasi.

## Keputusan utama

**Kita belum membuktikan dataset sudah mencapai ceiling. Kita sudah melihat diminishing returns pada representasi dan konfigurasi yang hampir sama.** Prioritas berikutnya adalah memperbaiki eksperimen EXAONE, mempertahankan tipe kategori pada Causilo/DFS, mencoba RealMLP sebagai pembanding supervised, dan menguji objective/checkpoint selection yang sesuai KS secara terkontrol. TabICLv2 layak satu eksperimen terpisah. Menambah puluhan agregasi atau seed tanpa hipotesis baru bukan prioritas.

EDA tetap penting, tetapi preprocessing yang benar tidak menjamin model apa pun akan menang. Percobaan baru di sini justru menunjukkan transformasi yang masuk akal secara teori bisa menurunkan KS fitur. Tidak ada janji melewati 0.57059.

## 1. Apa yang benar-benar berubah sepanjang tujuh versi?

| Versi | Selected OOF KS | AUC | Interpretasi |
|---|---:|---:|---|
| v1 | 0.505143 | 0.833748 | Baseline; equal blend memasukkan dua FM yang lebih lemah |
| v2 | 0.513474 | 0.836831 | Fitur perilaku baru membantu; FM justru turun |
| v3 | 0.514227 | 0.838370 | TabM membantu; gain tambahan kecil |
| v4 | 0.511407 | 0.837616 | Nilai dari log, fold/seed dan selection protocol berbeda; bukan ablation murni v3 |
| v5 | 0.527424 | 0.846151 | Lompatan terbesar sejak v2, terutama representasi harga/perilaku |
| v6 | 0.528334 | 0.846832 | Gain kecil, kombinasi beberapa perubahan |
| v7 | 0.526992 | 0.846937 | AUC tertinggi, tetapi KS turun; bukan kontradiksi |

Prediksi v7 direkonstruksi dari parquet yang memiliki application_id; target cocok dengan train, dan min-max blend test cocok dengan submission.csv. Jadi tidak ada indikasi salah mapping CSV pada artefak yang diperiksa. Untuk v2–v6, NPY lama tidak membawa ID; kesesuaian KS dengan log mendukung urutan, bukan bukti provenance independen. v1 memakai manifest ID yang tersedia. v4 tidak punya OOF tersimpan di results/v4 sehingga tidak masuk bootstrap berpasangan.

400 bootstrap berpasangan pada prediksi tetap:

- v7−v5: **−0.000432**, interval persentil 95% **[−0.002312, +0.002703]**.
- v7−v6: **−0.001342**, interval **[−0.003150, +0.001253]**.
- Korelasi ranking v6/v7 **0.999337**, v5/v7 **0.997785**.

Interval ini tidak mencakup ketidakpastian retraining atau pencarian fitur/blend yang sudah memakai label berkali-kali. Tidak membuktikan equivalence, tetapi belum mendukung kesimpulan v7 memburuk secara kuat.

Thin-file v5/v6/v7: **0.500427 / 0.499416 / 0.499890**. History: **0.535460 / 0.537107 / 0.536502**. Stagnasi terutama terlihat sejak v5; bukan karena tujuh versi tidak menghasilkan kemajuan sama sekali.

## 2. EXAONE: masalahnya konkret, bukan model pasti jelek

| Versi | Yang terjadi |
|---|---|
| v5 | `No available kernel` pada jalur attention |
| v6 | Probe context/query 3k berhasil; run dilewati karena context hasil budgeting 4.731 < batas minimum 10k |
| v7 | Probe 4k dan 12k berhasil; full context 56k ditolak perencana VRAM |

Log v7: probe 4k sekitar **65 detik**, KS **0.5437**; probe 12k sekitar **346 detik**, KS **0.5120**. Query kedua probe berbeda (`perm[n:2*n]`), sehingga **tidak boleh menyimpulkan context lebih banyak menurunkan kualitas**. Pembanding LGB OOF pada query yang sama 0.5467/0.5189 juga menggunakan jumlah data training berbeda, sehingga bukan equal-data benchmark.

Budget waktu v7 memperkirakan context 56k muat dalam sisa waktu, tetapi error menyatakan minimum reserved memory **25,891,438,592 byte**, target **13,034,943,283 byte**. Dua T4 menjalankan fold terpisah; VRAM keduanya tidak otomatis menjadi satu pool untuk satu model.

Root cause kode: `run_fm` hanya mengecilkan context bila teks error mengandung `out of memory`. Pesan sebenarnya **`detected VRAM cannot fit the minimum model forward`** tidak cocok, sehingga exception naik dan seluruh EXAONE dilewati. Ini batas resource yang dikenali salah, bukan OOF gagal.

Perbaikan untuk eksperimen Kaggle berikutnya, tanpa mengubah notebook sekarang:

1. Gunakan satu query evaluasi tetap yang tidak pernah masuk context; nested context 4k→8k→12k dari pool training yang sama, dengan label query hanya untuk evaluasi.
2. Mulai 12k karena probe sudah membuktikan konfigurasi itu berjalan. Probe query workload yang representatif juga diperlukan; context 12k/query 12k belum menjamin semua skenario 44k query.
3. Pisahkan memory cap dari time cap. Kenali preflight memory error secara spesifik serta CUDA OOM, kurangi context dan ulangi; exception lain tetap harus dilaporkan, jangan disembunyikan.
4. Jika satu fold harus mengecil ke 8k, gunakan context 8k konsisten untuk perbandingan lima fold pertama; simpan status, jumlah context aktual, dan prediksi per fold.
5. Jangan kembali ke 56k sebelum ada bukti pengukuran memory. Perluasan context bukan tujuan jika kontribusi OOF/blend pada budget sama tidak bertambah.

[Dokumentasi resmi EXAONE](https://github.com/LGAI-Research/EXAONE-Tabular) menyediakan `max_vram_bytes` dan memory planning. Pin versi/checkpoint serta catat override attention untuk T4; lebih baik konfigurasi publik jika tersedia daripada bergantung pada atribut privat. Belum ada hasil OOF EXAONE lengkap untuk menyatakan ia lebih baik/buruk.

## 3. Dua masalah representasi FM yang masih terbuka

**Causilo:** v7 benar-benar berjalan dan memperoleh **KS 0.522647**, jauh lebih baik daripada v1/v2 sekitar 0.4953/0.4848. Jadi asumsi lama bahwa dependency deklaratif pasti mencegah Causilo berjalan tidak berlaku untuk run ini.

Namun `fm_matrix` mengubah kategori menjadi `.cat.codes`, lalu seluruh matrix menjadi float32. Factory Causilo mengabaikan `cat_idx`. [README resmi Causilo](https://github.com/nums-ai/causilo/blob/main/README.md) meminta pandas categorical dtype untuk kode kategori numerik; tipe tersebut hilang di v7. Perbaikan yang layak diuji: kirim DataFrame dengan kategori yang terjaga dan vocabulary training-only. Tidak ada bukti besar gain-nya; hanya empat kategori, mungkin efeknya kecil.

**Native DFS/TabPFN-Rel:** v7 sudah menjalankan featurizer native, depth 3, 301 kolom, 291 nonconstant; hasil **0.491727**, sedangkan handcrafted TabPFN **0.518884**. Tetapi pipeline mengubah categorical DFS menjadi kode float, mengganti nama semua fitur menjadi rel_###, lalu menyatakan categorical columns kosong. Ini belum perbandingan representasi yang sepenuhnya adil.

Pertahankan mapping nama/lineage dan categorical metadata, jangan menebak kategori dari nilai angka saja. Periksa alignment output featurizer terhadap customer/application IDs; kode saat ini reset index lalu memasangkan apps IDs secara posisi tanpa pemeriksaan mapping eksplisit. Tidak ditemukan bukti mismatch aktual, tetapi provenance itu belum tersimpan untuk diverifikasi. Snapshot repayment tidak boleh dianggap riwayat status pada due_date untuk query historis yang lebih awal; pada satu query selected cutoff per customer gunakan semantik snapshot yang didokumentasikan.

Tidak perlu depth DFS lebih tinggi dulu. Jalankan satu perbandingan representasi dengan kategori dan alignment benar, jumlah context sama, serta feature budget yang tercatat. Top-100 univariate selection dapat membuang fitur interaksi lemah secara marginal; bandingkan handcrafted all-features vs top-100 dengan budget terukur sebelum memperluas HPO.

## 4. AUC bukan proxy yang selalu searah dengan KS

Markdown v7 menyatakan AUC monoton terkait KS. Itu tidak benar untuk ranking berbeda. Dari prediksi aktual ada **lima pasangan model dengan arah perbedaan AUC dan KS berlawanan**.

Contoh TabM PLE vs periodic v7: PLE AUC **0.846106**, KS **0.525311**; periodic AUC **0.846349**, KS **0.525144**. Blend v7 juga menaikkan AUC dari v6 sambil menurunkan KS.

Ini bukan alasan langsung early-stop pada maksimum KS setiap epoch: KS lebih berisik dan menambah selection bias. Eksperimen yang tepat adalah merekam kurva AUC, logloss, KS, epoch/iteration, kemudian membandingkan kebijakan checkpoint yang ditentukan lebih dulu pada inner validation; outer fold untuk evaluasi. Log sekarang belum menyimpan kurva yang cukup untuk mendiagnosis underfit/overfit secara pasti.

Aggregation tetap 80% enam trees + 20% TabM, tanpa tuning bobot lokal:

| Skala aggregation | OOF KS |
|---|---:|
| Probability mean | 0.525038 |
| Rank mean | 0.524621 |
| Logit mean | 0.524547 |
| Selected v7, bobot berbeda | 0.526992 |

Rank averaging tidak otomatis terbaik, meskipun monotonic transform **satu score** mempertahankan KS. Mengubah masing-masing komponen sebelum blending bisa mengubah ranking gabungan. Selisih kecil di atas belum cukup untuk mengganti submission. Jangan memperlakukan skor rank/min-max sebagai probabilitas default terkalibrasi.

Threshold dari separuh lain menghasilkan sensitivitas sekitar **0.761/0.816**, spesifisitas **0.756/0.714** pada dua separuh. Error contoh disimpan. Banyak false positive di decile 7–8 wajar untuk threshold Youden/KS pada prevalensi rendah; itu bukan bukti label sintetis salah. Jangan hapus/relabel customer hanya karena model yakin tetapi salah.

## 5. Fold KS lebih tinggi: bukan gain gratis dari normalisasi

Saya merekonstruksi seed-42 StratifiedKFold dengan NumPy dan memastikan lima KS TabPFN cocok dengan log sampai presisi empat desimal. Kemudian membandingkan gap mean-fold vs pooled dengan 80 partisi acak pada prediksi yang sama.

| Model | Pooled KS | Mean fold KS | Gap partisi acak rata-rata | KS setelah rank per fold |
|---|---:|---:|---:|---:|
| TabPFN | 0.518884 | 0.523135 | 0.002887 | 0.517978 |
| TabPFN-Rel | 0.491727 | 0.495962 | 0.003786 | 0.491567 |
| Causilo | 0.522647 | 0.525298 | 0.002538 | 0.522552 |
| Blend | 0.526992 | 0.530199 | 0.002470 | 0.525381 |

Memaksimalkan KS secara terpisah pada subset memang cenderung memberi rata-rata lebih tinggi. Ada variasi skala antar-fold, tetapi normalisasi rank per-fold **menurunkan** semua pooled KS pada pemeriksaan ini. Jangan menjual mean fold 0.5302 sebagai pooled OOF 0.5302 atau mencoba memperbaiki gap hanya dengan kalibrasi.

Rank-per-fold di sini hanya diagnosis. Prediksi test per-fold sudah dirata-ratakan dan tidak disimpan; transformasi pasangan test tidak dapat direkonstruksi secara benar. Untuk blend multi-seed, grup seed 42 juga bukan pure fold bagi semua komponen.

## 6. EDA baru: reliabilitas harga dan preprocessing yang ternyata tidak membantu

Harga adalah sinyal penting sejak v5. Hipotesis baru: mean premium dari sedikit order memiliki ketidakpastian lebih besar, sehingga reliability-aware aggregation mungkin lebih baik.

Order dibagi deterministik ke dua kelompok menggunakan hash order_id. ID hanya untuk assignment diagnostik, tidak menjadi predictor. Referensi kategori dihitung pada training customers fold diagnostik dan hanya bulan sebelum query.

| Jumlah order | Customer train | Korelasi dua mean | Median selisih absolut |
|---|---:|---:|---:|
| 1–15 | 4.521 | 0.499 | 0.328 |
| 16–30 | 36.815 | 0.610 | 0.245 |
| 31–60 | 27.600 | 0.673 | 0.186 |
| >60 | 1.064 | 0.743 | 0.143 |

Grafik benar-benar menunjukkan estimator lebih goyah untuk histori pendek. Perbedaan ini mencakup heterogenitas pembelian, bukan semata measurement noise, dan bukan identifikasi latent risk generator.

Saya coba order-equal mean, median, dan fixed shrinkage `mean*n/(n+10)`, dengan jumlah order/std/standard-error proxy. Pada fold diagnostik yang sama:

- pp_mean lama: **feature KS 0.373289**.
- order-equal mean: **0.375045**.
- median: **0.346032**.
- fixed shrinkage: **0.371817**.

Grafik sebelum/sesudah memperlihatkan shrinkage menarik distribusi ke nol. Median dan shrinkage tidak memperbaiki separation ini. Enam kandidat tidak ada yang lolos BH q<0.05 pada kedua separuh setelah conditioning pada ranking v7, bulan, dan histori. **Tidak dipromosikan sebagai fitur wajib.** Jika diuji model nanti, mulai dari order-equal mean + count/dispersion sebagai ablation kecil, jangan mengganti pp_mean dengan shrinkage yang sudah kalah pada diagnosis.

Screening tersebut eksploratif: OOF v7 sudah dipilih menggunakan label, binning punya keterbatasan power, dan confirmation split bukan label baru yang belum pernah dilihat dalam pengembangan. Tidak adanya hasil signifikan bukan bukti sequence model atau representasi lain mustahil membantu.

## 7. Input RealMLP dan duplikasi

[PyTabKit resmi](https://github.com/dholzmueller/pytabkit) menyediakan RealMLP_TD_Classifier dan preprocessing native, tetapi dokumentasinya meminta missing numerik ditangani terlebih dulu. Saya membuat input demonstrasi fold-local:

- 228 numerik: median training-only, observed values tetap persis sama.
- 228 missing flags per fitur: menjaga arti ketiadaan histori dan mendukung missing baru. Banyak flag konstan/redundan; ini format konservatif, model boleh membuang konstanta di training fold.
- Empat kategori: vocabulary training-only, unknown eksplisit; belum dipaksa menjadi ordinal float.
- Total 460 predictors, 100.000 aplikasi; target dipisah dari input.

Semua nilai numerik output finite; tidak ada observed value berubah. Tidak ada missing baru pada kolom fit-complete di validation/test aktual. Grafik menunjukkan spike median akibat imputasi; missing flags diperlukan supaya spike itu tidak disalahartikan sebagai nilai finansial sungguhan. Tidak menumpuk quantile transform TabM di atas native RealMLP preprocessing tanpa ablation.

Audit ulang menemukan **0 duplicate feature rows tanpa ID** dan **0 overlap customer train/test**. Audit detail sebelumnya pada `eda/v6_review` menemukan nol duplicate primary key item/installment/account-month; repeated item content tidak cukup untuk menghapus transaksi berbeda. Hasil lama tentang order_total berulang per item tetap berlaku: aggregate satu kali per order. Tidak dilakukan deduplikasi berdasarkan kemiripan risiko atau menghapus error model.

## 8. Kesimpulan EDA lama yang perlu diluruskan

- `eda/69_ceiling_and_leaderboard.py` mensimulasikan label dari probabilitas yang dikalibrasi dari prediksi v6. Kesamaan simulated KS dengan observed KS adalah self-consistency di dunia asumsi tersebut; **bukan estimasi Bayes ceiling dataset**. Mengubah simulated truth dengan temperature juga bukan gain yang bisa diperoleh dengan mengubah temperature prediksi pada label tetap.
- Selisih public–OOF satu model bukan konstanta untuk mengonversi skor top leaderboard menjadi target OOF universal. Split, retraining, dan selection noise berbeda. Nilai 0.541 “top-1 equivalent” pada analisis lama tidak defensible sebagai target tepat.
- `eda/72_sequence_bigrams.py` hanya menguji representasi tertentu dengan scoring terbatas. Hasil negatif tidak membuktikan urutan tidak membawa informasi tambahan. Script itu juga mengestimasi class-conditional likelihood dari label; walaupun disebut counting, secara statistik itu model supervised sederhana. **Tidak dijalankan ulang** karena instruksi sekarang melarang model lokal.
- Sejumlah analisis lama memakai `calibrate(y, score)` pada label yang sama, lalu residual testing. Itu tidak menjadi validasi independen hanya karena residual diberi cross-fitting sesudahnya. Threshold dari 30 random features juga tidak otomatis mengontrol banyak hipotesis.
- Train/test pada periode yang sama mendukung random grouped-by-customer CV, tetapi tidak membuktikan tidak ada hidden distribution shift. Time slicing berguna sebagai stress check; itu berbeda dari out-of-time validation.
- Simulasi public/private bukan jaminan gap dan bukan “noise floor” pasti. Bandingkan **paired prediction differences**, bukan hanya selisih skor dengan SD antar-fold yang berisi kesulitan fold berbeda.

File lama tidak diubah agar provenance tetap ada. Koreksi interpretasi ada di laporan ini.

## 9. Prioritas model dan rencana 16 hari

Urutan ini prioritas eksperimen berdasarkan bottleneck saat ini, bukan ranking SOTA universal.

| Prioritas | Eksperimen | Alasan dan batas |
|---|---|---|
| 1 | EXAONE context tetap 8–12k + memory fallback | Menghasilkan OOF pertama yang lengkap; memperbaiki blocker nyata |
| 1 | RealMLP-TD pada fitur manual | Supervised pembanding yang belum ada di v1–v7; input CPU sudah disiapkan |
| 2 | Causilo typed categories | Model sudah cukup kuat; ubah satu keputusan representasi |
| 2 | Checkpoint AUC vs kebijakan KS yang ditetapkan sebelumnya | Actual AUC/KS disagreement terukur; butuh inner selection dan kurva training |
| 3 | TabICLv2 pada fitur manual | Keluarga FM alternatif; satu pilot runtime dan lima fold jika masuk budget |
| 3 | Native DFS dengan schema/lineage benar | Memperbaiki kualitas perbandingan sebelum depth/feature count ditambah |
| Cadangan | Sequence/relational transformer, Mitra-v2 | Hipotesis belum tertutup, tetapi lebih mahal/kompleks; jangan ambil slot utama sebelum baseline fair |

[TabICLv2 resmi](https://github.com/soda-inria/tabicl) menyediakan checkpoint classifier v2, kontrol mixed precision/offloading, dan inference configuration. [Paper primer](https://arxiv.org/abs/2602.11139) mengevaluasi arsitektur dan prior sintetis yang diperbarui. Ini alasan mencoba satu pilot, bukan bukti ia mengalahkan TabM pada LOGIKA. T4 runtime/VRAM perlu diukur di Kaggle; jangan menyamakan benchmark GPU lain dengan laptop/T4.

RealMLP mulai dari TD, bukan HPO puluhan konfigurasi. Gunakan outer folds sama dengan baseline, training-only imputation/vocabulary, dan inner validation untuk early stopping. Catat data yang benar-benar dipakai untuk fit agar perbandingan adil. Causilo yang sudah berjalan tidak perlu dipaksa upgrade dependency hanya karena deklarasi versi; pin environment sukses sebelum mengubah apa pun.

Rencana kerja konkret, 15–30 Oktober (inklusif 16 hari; deadline dokumen 30 Oktober 23:59 WIB):

1. **15–16 Oktober:** bekukan baseline v6/v7 dan fold manifest; pin environment; simpan per-fold/per-seed predictions, checkpoint metadata, kurva AUC/KS/logloss. Perbaiki EXAONE dan schema kategori.
2. **17–20 Oktober:** EXAONE bounded-context, RealMLP-TD, Causilo typed-categories. Satu perubahan per run; evaluasi semua pada folds yang sama. Jangan mengulang ensemble search tiap selesai satu fold.
3. **21–24 Oktober:** TabICLv2 dan satu controlled checkpoint-selection ablation. Jika kandidat baru memberi residual berbeda serta gain konsisten, lanjut; jika tidak, jangan menambah sweep hanya mengejar public.
4. **25–27 Oktober:** ulangi kandidat terpilih dengan seed kedua; uji satu group feature kecil atau schema DFS bila masih ada budget. Pilih blend sederhana dan satu prosedur bagging yang dibekukan, bukan belasan blend baru.
5. **28–30 Oktober:** verifikasi rerun end-to-end di Kaggle, audit ID/cutoff/reproducibility, pilih final dari bukti OOF dan stabilitas subgroup. Sisakan waktu kegagalan teknis dan pengumpulan sebelum deadline.

Promosi kandidat membutuhkan gain paired yang konsisten, risiko subgroup yang dapat diterima, dan waktu/memori yang masuk budget. Interval fixed-OOF tetap hanya alat diagnosis. Untuk estimasi yang lebih jujur, nested validation harus mencakup seleksi model/fit estimator; membagi OOF lama dua bagian tidak otomatis menghasilkan holdout baru. Private score tidak dapat dijamin dari train/public saja.

## Output dan batas pemeriksaan

Lima analisis `.py` telah dijalankan; masing-masing menghasilkan angka dan gambar. Semua lima PNG diperiksa visual, termasuk hasil preprocessing kedua kelas. Hash input disimpan; tidak ada notebook yang dibuat/diubah, model/inference dijalankan lokal, data dikirim ke API, atau submission untuk menebak label test.

Belum ada KS model baru dari RealMLP, EXAONE lengkap, typed-category Causilo, atau kandidat preprocessing. Skor private dan public v5/v7 belum tersedia. Batas inilah yang membedakan keputusan eksperimen dari klaim peningkatan yang sudah terbukti.
