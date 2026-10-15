# Audit v5 dan keputusan EDA — 14 Oktober 2026

**Keputusan:** v5 adalah kemajuan nyata dibanding v3 pada prediksi tersimpan. Jalur yang paling masuk akal berikutnya adalah memperbaiki validasi/preprocessing dan menyamakan input eksperimen model. EDA ini belum menemukan fitur tambahan yang cukup kuat untuk menjanjikan kenaikan lagi, apalagi menjanjikan public KS melewati **0.57059**. Angka top-1 tersebut berasal dari informasi pengguna; public score v5 belum diberikan.

Yang dilakukan: audit statis notebook, membaca prediksi Kaggle, menjalankan enam analisis CPU, memeriksa grafik, memperbaiki preprocessing, dan menguji hasilnya. Tidak ada model/notebook yang dijalankan. Sumber bukti lokal utama: [00_scores.csv](out/00_scores.csv), [log per analisis](out/), dan manifest input/output.

## 1. Seberapa bagus v5?

| Hasil tersimpan | OOF KS |
|---|---:|
| v3 selected blend | 0.514227 |
| v5 LightGBM | 0.524051 |
| v5 equal-weight six GBDT | 0.524642 |
| v5 TabM | 0.525735 |
| v5 DFS + TabPFN | 0.491191 |
| v5 selected blend | **0.527424** |

Delta blend v5–v3 **+0.013196**. Bootstrap berpasangan 400 kali menghasilkan interval persentil 95% **[+0.007817, +0.018032]**. Ini conditional pada prediksi/fitur/bobot yang telah dipilih, bukan interval yang mencakup training ulang dan semua proses pemilihan.

Bukti yang lebih terkontrol tersedia dalam output ablation v5: konfigurasi shallow LightGBM dan seed yang sama, fitur lama **0.5070 → 0.5221**, lima dari lima fold membaik; rata-rata kenaikan fold **+0.0139**. Ablation ini dibaca dari notebook, tidak dijalankan ulang.

Thin-file membaik **0.476767 → 0.500427**; dengan histori **0.527930 → 0.535460**. V5 membaik pada 11/12 irisan bulan, tetapi Januari turun. KS bulanan v5 berkisar **0.499439–0.566514**. Ini irisan random OOF, **bukan** validasi forward-time.

Bobot blend yang direkonstruksi tepat adalah `[4,0,0,1,9,3,0,13]/30` untuk `[lgb,lgb_s,xgb,xgb_s,cat,cat_s,tabpfn_rel,tabm]`. Rekonstruksi test cocok dengan submission menurut application_id. Bobot hasil dua half-split sangat berbeda: porsi TabM sekitar 23% versus 60%. Interval bootstrap keunggulan selected blend dibanding fixed 20% TabM + 80% trees mencakup nol: **[-0.000632, +0.004205]**. Jangan habiskan iterasi hanya mengutak-atik bobot greedy.

[Grafik validasi dan korelasi prediksi](out/00_validation.png) memperlihatkan bahwa prediksi GBDT/TabM sangat berkorelasi. Menambah seed pada semua model serupa kemungkinan memberi diversifikasi lebih kecil dibanding memperbaiki representasi input—ini hipotesis prioritas komputasi, bukan hasil ablation baru.

## 2. Kesalahan dan keterbatasan v5 yang harus dibereskan

| Lokasi kode v5 | Temuan | Perbaikan |
|---|---|---|
| `v5_features`, normalisasi `lu` per kategori | Mean/std memakai item seluruh customer train+test dan seluruh tanggal. Filter cutoff per customer tidak mencegah transaksi customer lain yang terjadi lebih belakangan memengaruhi referensi aplikasi awal. | Fit referensi hanya pada training fold dan gunakan observasi sebelum waktu query. Implementasi dan uji ada di script 01. |
| `nn_prep` / pembentukan `XN_tr` sebelum fold | ECDF, pemilihan kolom, missing-pattern compression memakai seluruh train, termasuk validation fold. Ini pengaruh distribusi validation tanpa label, bukan target leakage langsung. | Fit di dalam fold. Ukur dampak terpisah; jangan menganggap seluruh gain v5 berasal dari leakage. |
| `stratified_subsample` lalu slicing probe EXAONE | Positif digabung di depan negatif. Sampel 6.000 menjadi context 815 positif/3.000 dan query **0 positif/3.000**. Probe KS tidak dapat menilai diskriminasi. | Dua subset stratified yang disjoint dan diacak; implementasi menghasilkan 407 positif di masing-masing subset 3.000. |
| `select_fm_features('hand', np.arange(len(y)))` pada probe | Seleksi menggunakan label seluruh train, termasuk query probe. | Pilih fitur hanya dari context/training, lalu bekukan untuk query. |
| `FM_N_FEATURES=100` ditambah 4 kategori | EXAONE menerima 104 kolom; log gagal pada `[FS] attention selection`. | Coba maksimal **100 total kolom** terlebih dahulu. Ini menghindari seleksi internal tambahan; belum membuktikan semua kernel inference T4 kompatibel. |
| Manipulasi `module._sdpa_backend` | Private attribute tidak menjamin seluruh jalur, termasuk feature selection, mengikuti backend tersebut. | Pin versi dan uji jalur resmi pada GPU Kaggle. Jangan menganggap patch tersebut sudah menyelesaikan error. |
| Nama result `tabpfn_rel` | Yang dijalankan adalah TabPFN pada DFS buatan sendiri; bukan evaluasi native package TabPFN-Rel. Input DFS juga tidak berisi fitur harga relatif dan burst terbaru yang membantu v5. | Sebut hasil sesuai implementasinya; uji handcrafted / gabungan / native harness secara terpisah. |
| Pelaporan blend CV | Greedy bobot dipilih pada seluruh label OOF; gate half-split dan seleksi fitur juga memakai data ini. | Pisahkan skor pemilihan dari estimasi evaluasi. Simpan OOF per seed/fold beserta IDs dan bobot; pilih blend pada lapisan validasi yang dipisahkan. |
| Dependensi / checkpoint | Sebagian besar pip package dan Git EXAONE tidak dipin. Print “TabPFN-3.5” bukan manifest checkpoint. Ada warning konflik Polars dengan cudf-polars; yang diaudit tidak memakai cudf-polars. | Simpan package versions, commit, checkpoint hash, feature list dan fold IDs. Jangan mengganti environment baseline diam-diam. |

Dokumentasi EXAONE menyatakan classifier memakai maksimal 100 kolom, dan input lebih lebar menjalankan feature selection berbasis attention. Float64 pada array input bukan bukti compute CUDA berjalan float64. Karena log berhenti sebelum inference, kualitas EXAONE **belum diukur**. [Sumber resmi EXAONE](https://github.com/LGAI-Research/EXAONE-Tabular#feature-selection-wide-tables).

Cache saat audit berisi **232 predictor**, sesuai output notebook, tanpa `pp_median`. Tidak ditemukan mismatch cache tersebut saat ini. File NPY tidak menyimpan IDs: alignment didukung oleh kesesuaian KS tercatat dan rekonstruksi CSV, dengan keterbatasan provenance yang disebut di README.

## 3. Harga relatif: koreksi leakage tanpa membuang sinyal

Script 01 membangun referensi mean/std log unit price dari training customers setiap fold. Untuk aplikasi pada bulan M, referensi hanya memasukkan transaksi sebelum awal bulan M. Item customer query sendiri tetap dibatasi cutoff keputusan. Referensi ini konservatif satu bulan; bukan imputasi label atau target encoding.

| Fold diagnostik | KS pp_mean global lama | KS mean harga aman |
|---|---:|---:|
| 0 | 0.382975 | 0.382663 |
| 1 | 0.359176 | 0.358994 |
| 2 | 0.363327 | 0.364294 |
| 3 | 0.371935 | 0.373452 |
| 4 | 0.376144 | 0.375789 |

Ini **KS fitur tunggal**, bukan KS model setelah perbaikan. Korelasi peringkat lama/baru **0.99917–0.99924**; perubahan absolut median premium sekitar **0.017–0.019**. Plot kelas tetap memisahkan target 0/1 dan scatter sebelum/sesudah hampir diagonal. Koreksi ini mempertahankan sinyal yang terlihat; tidak menjamin model score identik.

Uji yang benar-benar dijalankan:

- Mean/std satu referensi dihitung ulang secara independen dari baris yang memenuhi customer training dan batas tanggal.
- Menambah item validation tidak mengubah referensi training.
- Menambah 10 transaksi ekstrem dari customer training yang lebih belakangan, tetapi masih sebelum cutoff customer tersebut, mengubah referensi global lama. Referensi aplikasi awal yang aman tidak berubah: maksimum delta z **0.0**.
- Seluruh query memakai timestamp referensi lebih awal dari query month.
- Contoh hasil transformasi disimpan untuk kedua kelas.

[Grafik harga sebelum/sesudah dan korelasi](out/01_price_before_after.png), [contoh dua kelas](out/01_transformed_samples.csv), [uji future lintas customer](out/01_cross_customer_future_injection.json), [referensi per fold](out/01_preprocessing_manifest.json).

## 4. Duplikasi: jangan merusak total belanja

Ada **4.359.817 item** pada **2.986.855 order**. Primary key item unik. Tidak ada customer overlap train–test, duplikasi profile persis tanpa ID, atau duplikasi penuh feature vector tanpa ID.

**2.388 baris item mempunyai isi berulang** pada kolom order/kategori/quantity/item_total. Ini tidak cukup untuk menyatakan baris salah: item berbeda memiliki ID berbeda dan jumlah seluruh item cocok dengan order_total pada setiap order.

Menjumlah order_total di setiap item menggelembungkan total **1.739704×**. Solusi: aggregate order_total sekali per order, tetapi tetap simpan item berbeda untuk analisis quantity/unit price. Nol duplikasi persis tidak membuktikan tidak ada semua jenis near-duplicate; tidak dilakukan pencarian semua pasangan nearest-neighbor.

[Angka audit duplikasi](out/01_duplicates.json). Uji integritas tabel lengkap dan pembuktian kerusakan akibat dedup konten sudah tersedia pada EDA sebelumnya di `eda/deep/00_integrity.py`.

## 5. Kandidat tambahan: belum ada bukti incremental yang cukup

29 kandidat meliputi premium dengan bobot setara per order/per kategori, dispersi, kategori, entropy, quantity, window yang tidak overlap (0–30, 30–90, 90–182, 182–365, >365 hari), jumlah observasi, dan perubahan harga pada customer yang cukup teramati di kedua window.

62 variabel diuji total: kandidat baru + 27 fitur v5 + enam kontrol negatif. Analisis menggunakan dua half-split, bin fitur ditentukan di discovery, stratifikasi skor v5/history/bulan, koreksi Benjamini–Hochberg, serta sensitivitas jumlah risk bins 20 versus 60. Tidak dibuat model koreksi residual dan tidak mengoptimalkan submission.

Hasil: **0 kandidat baru lolos replication rule**, 0 fitur lama dan 0 kontrol negatif lolos. Beberapa nominal p-value kecil tidak konsisten di kedua subset atau hilang setelah koreksi pengujian. Ini bukan bukti bahwa semua interaksi mustahil berguna; hanya tidak ada alasan kuat untuk menambahkan seluruh keluarga fitur baru sekarang.

Ada bias cohort yang perlu diperhatikan: syarat minimal tiga item di window recent dan older dipenuhi **70.54% nasabah bad**, tetapi hanya **44.65% good**. Perubahan harga rata-rata pada cohort matched adalah **+0.0803** untuk bad versus **+0.0210** untuk good. Menafsirkan perbedaan tersebut sebagai pola universal akan mengabaikan siapa yang teramati. Setelah conditioning terhadap v5, kandidat belum menunjukkan residual yang konsisten.

Korelasi premium mean dengan equal-order **0.9839** dan equal-category **0.9165** juga menunjukkan redundansi. Grafik residual memperlihatkan tren marginal yang kuat pada recent activity, tetapi tren marginal tidak sama dengan tambahan informasi di atas model.

[Screen lengkap](out/02_residual_screen.csv), [keputusan](out/02_decision.json), [grafik](out/02_residual_candidates.png), [cohort matched](out/02_matched_temporal_cohort.csv).

Keterbatasan: skor v5 dan pemilihan fitur sebelumnya sudah dipengaruhi label pada dataset ini. “Confirmation” di sini adalah replikasi eksploratif, bukan fresh untouched holdout. Tidak melanjutkan pencarian bin/fitur berulang pada confirmation hingga p-value terlihat bagus.

## 6. Preprocessing numerik sudah diuji, efek pada model belum

Perbandingan ECDF global v5 dengan ECDF fold-local mempertahankan 266 dimensi pada diagnostic fold0. Perubahan nilai numerik validation median **0.00125**, p99 **0.00971**. Jadi pengaruh validation terhadap ECDF terukur, tetapi kecil; tidak ada dasar menyalahkannya sebagai sumber seluruh gain v5.

Transformasi interpolasi dan mask per kolom memakai ulang implementasi EDA yang telah diuji. Pada lima fitur pemeriksaan utama, KS univariat sebelum/sesudah tidak berubah dan korelasi peringkat mendekati 1. Plot income menjadi lebih teratur tanpa membalik urutan; price separation tetap terlihat. Nilai DPD tetap memiliki massa diskret; missing=0 pada transformasi dibedakan dengan mask, bukan dianggap DPD nol.

Ekspor akhir **menggabungkan** kedua perbaikan: seluruh keluarga pp lama diganti fitur harga per fold, kemudian numeric preprocessing fit hanya pada training fold0. Hasilnya **240 fitur numerik + 151 missing masks = 391 dimensi**, ditambah empat kategori. Asserts finiteness, mask consistency, batch/order invariance, dan ID alignment lolos. Schema baru ini kandidat ablation, bukan bukti bahwa dimensi lebih banyak lebih baik.

[Grafik pemeriksaan transformasi numerik](out/03_numeric_before_after.png), [KS fitur pemeriksaan](out/03_transform_metrics.csv), [manifest gabungan](out/03_integrated_manifest.json), [array NPZ](out/03_fold0_numeric_diagnostic.npz). Grafik numerik mengisolasi perubahan ECDF pada input v5; grafik script 01 memeriksa perubahan referensi harga. Raw histograms di script 03 hanya dipotong rentangnya untuk tampilan, bukan clipping pada data ekspor.

## 7. Input foundation model perlu dibandingkan secara adil

DFS-only tidak memiliki fitur premium per kategori ataupun burst ec2_ewm. Seleksi 100 numerik + empat kategori di diagnostic fold memberi coverage numerik thin-file **36.56%**, dibanding **56.12%** pada gabungan handcrafted + DFS. Gabungan tersebut mempunyai 16 fitur dengan nama price dan tiga burst features; jumlah bukan feature importance.

Ini menjelaskan ketidakseimbangan eksperimen, bukan membuktikan penyebab seluruh kegagalan TabPFN. V5 membandingkan model supervised dengan fitur 232 kolom yang berbeda dari 104 kolom jalur FM. Sebelum menyimpulkan foundation model jelek, bandingkan model pada feature view dan fold yang sama.

Ekspor raw gabungan mempertahankan NaN dan kategori. Seleksi hanya memakai training diagnostic fold0. Budget EXAONE disediakan sebagai daftar **96 numerik + 4 kategori**, sedangkan jalur lainnya boleh memakai seluruh view. Wajib fit ulang seleksi dan referensi di actual folds.

[Grafik coverage/family](out/04_feature_views.png), [manifest kolom](out/04_feature_view_manifest.json), [raw input](out/04_joint_raw_fold0.parquet).

## 8. Validasi dan risiko public–private

Train dan test berada pada periode Januari–Desember 2025 dan keduanya mempunyai 30% thin-file. Maksimum KS train–test numerik **0.010405**, maksimum PSI **0.001159**, tidak ada drift fitur yang signifikan setelah FDR 5%. Raw prediction distribution juga mirip; diperiksa sebelum rank transform yang dapat menyembunyikan perubahan distribusi.

Audit lima tabel history menemukan nol baris melewati availability cutoff dan nol baris tepat di cutoff. Perbedaan operator < versus <= belum mengubah baris dataset ini. Bureau menggunakan available_datetime, monthly account baru tersedia setelah bulan lengkap, last_payment_datetime repayment harus sebelum cutoff. DPD repayment adalah snapshot keputusan; jangan membackdate nilainya menjadi observasi yang sudah diketahui di due_date.

Simulasi random 30.000 OOF sebagai pseudo-test (400 kali):

| Skenario public share | Median absolute gap | p95 absolute gap |
|---|---:|---:|
| 30% | 0.00938 | 0.02789 |
| 50% | 0.01011 | 0.02626 |

Proporsi 30/50% adalah skenario, bukan klaim split aktual. Bobot public/private 45%/45% pada rules tidak menyatakan proporsi row split. Simulasi ini conditional pada OOF dan asumsi random sample; tidak mengukur distribution shift yang belum terlihat atau selection bias.

Random stratified CV menjadi pilihan utama yang masuk akal berdasarkan struktur ini, bukan sesuatu yang dapat dibuktikan “pasti benar” dari PSI saja. Periksa subgroup dan irisan waktu; evaluasi time-blocked tambahan memerlukan model baru di Kaggle. Public KS top-1 tidak boleh langsung dibandingkan dengan OOF lokal atau dipakai untuk merekonstruksi label test.

[Drift/timing](out/05_drift_timing.png), [audit waktu](out/05_information_timing.csv), [simulasi gap](out/00_gap_scenarios.csv).

## 9. Prioritas model dan eksperimen berikutnya

Riset dokumentasi resmi diperiksa 14 Oktober 2026. Peringkat eksperimen di bawah adalah penilaian untuk dataset ini, bukan ranking benchmark atau hasil training baru.

| Kandidat | Keputusan praktis |
|---|---|
| TabM + CatBoost/LightGBM | Tetap baseline utama berdasarkan hasil v5. Catat KS dan early-stop reason/epoch per fold. Uji perbaikan harga dan ECDF satu per satu sebelum mengganti banyak komponen. |
| TabPFN-3.5 | Prioritas FM pertama untuk feature view yang setara. Pin checkpoint/version; default paket dapat berubah. Dokumentasi sekarang memakai 3.5 sebagai default. [Repo resmi](https://github.com/PriorLabs/TabPFN). |
| Native TabPFN-Rel | Jalur berbeda yang layak dibandingkan setelah kontrol handcrafted. Harness mengikuti relasi dan membangun DFS; Python 3.11/3.12 dibutuhkan. Gunakan environment kompatibel di Kaggle dan task dengan label kompetisi/cutoff sendiri; jangan mengunduh dataset contoh sebagai data training tambahan. [Repo resmi](https://github.com/PriorLabs/tabpfn-rel). |
| Causilo | Challenger berikutnya, mempertahankan categorical dtype dan missing values. Dokumentasi saat ini meminta PyTorch 2.13+, sementara log v5 menunjukkan 2.11.0+cu128; environment perlu diselesaikan sebelum benchmark. Jangan mengekstrapolasi benchmark H100 ke T4. [Repo resmi](https://github.com/nums-ai/causilo). |
| EXAONE | Setelah probe split dan budget total kolom diperbaiki. Gunakan knob publik compute dtype dan max_vram_bytes; kompatibilitas attention T4 harus diukur di Kaggle. Error sekarang tidak memberi skor kualitas model. [Dokumentasi resmi](https://github.com/LGAI-Research/EXAONE-Tabular). |
| Mitra-v2 | Eksperimen setelah baseline; model card menyebut 75.7M parameter dan recipe 50-step fine-tuning dengan 8-fold bagging. Biaya recipe tersebut perlu dibandingkan pada budget GPU yang sama. [Model card resmi](https://huggingface.co/autogluon/mitra-classifier-2). |
| RT-PluRel | Prioritas lebih rendah sampai jalur relasional yang lebih sederhana lolos. RelArena mengklasifikasikannya sebagai system, bukan satu classifier drop-in. Perlu adapter dan audit timestamp snapshot. [RelArena resmi](https://github.com/PriorLabs/relarena). |

Urutan ablation yang disarankan di Kaggle:

1. Reproduksi v5 dengan fold IDs, package/checkpoint pins dan input hash tersimpan.
2. Pertahankan fitur/model/seed, ubah hanya referensi harga; lalu uji hanya ECDF fold-local; kemudian gabungkan. Seluruh fitting transformasi harus berada dalam fold.
3. Bandingkan TabM dan GBDT pada raw/encoded view yang sesuai. Jangan menganggap normalisasi wajib untuk trees atau menyeragamkan seluruh preprocessing model.
4. Bandingkan TabPFN pada handcrafted versus gabungan versus DFS-only dengan split yang sama; native TabPFN-Rel dilaporkan terpisah.
5. Causilo/EXAONE hanya diteruskan jika smoke test GPU dan OOF lengkap berhasil; OOF parsial tidak boleh masuk blend.
6. Evaluasi kandidat berdasarkan paired fold KS, pooled OOF KS, thin/history dan month stability, bukan satu seed atau public leaderboard. Blend sederhana tetap menjadi pembanding; gunakan evaluasi bobot yang dipisahkan.
7. Simpan OOF dan test bersama application_id, seed, fold, raw probabilities, feature names dan runtime. Jangan memakai leaderboard probing untuk menebak kesalahan per test row.

Tidak ada skor baru yang diklaim dari langkah-langkah itu. Mengulang pembagian label yang sudah sering dieksplorasi tidak menghasilkan holdout yang benar-benar baru. Private leaderboard masih satu-satunya evaluasi yang labelnya belum terlihat di sini.

**Batas kesimpulan:** EDA membantu menghindari fitur rusak dan perbandingan tidak adil. EDA/preprocessing yang benar tidak menjamin semua model berhasil. Yang berhasil dibuktikan sekarang adalah gain v5 pada hasil tersimpan, bug probe/representasi, koreksi preprocessing yang teruji, dan belum adanya kandidat residual baru yang cukup meyakinkan. Pilihan paling rasional adalah menguji perbaikan pipeline dan feature view dahulu, sambil mempertahankan v5 sebagai baseline kuat—bukan menjanjikan 0.57059+.
