# Analisis LOGIKA UI 2026

Analisis untuk kompetisi dengan slug **`data-science-competition-logika-ui-2026`**. Data diunduh menggunakan kredensial Kaggle lokal. Kompetisi ini memprediksi probabilitas mencapai DPD 90 dalam 12 bulan kalender.

## Hasil yang dapat dibuka

- [Notebook pipeline: EDA sampai submission](../../notebook/README_teman.md), dengan kode langsung di sel dan penjelasan per versi
- [Notebook reverse engineering RNG dan jitter](../../notebook/07_reverse_engineering_rng.ipynb), dengan probe seed, inferensi noise, dan generator pendekatan
- [Checkpoint Kumo Large: public **0,56262**](../../results/teman/submission_kumo.csv), dikonfirmasi melalui API Kaggle; [notebook 08](../../notebook/08_kumo_tabular.ipynb)
- [Eksperimen lanjutan Kumo](../../notebook/09_kumo_improvement.ipynb): fitur laten, TabICLv2, sampling kelas, ensemble lebih besar, dan konteks penuh; [laporan](../../artifacts/optimization/kumo_improvement/laporan_improvement.md)
- [Checkpoint Kumo–spline: public **0,56393**](../../results/teman/submission_kumo_spline_exploratory.csv), menurut laporan pengguna 18 Oktober 2026; [bukti pencatatan file](../../artifacts/optimization/kumo_hybrid/public_result_056393.json). Pooled KS lima fold 0,52851 vs 0,52722; rata-rata KS turun dan hanya 1/5 fold membaik; [notebook 10](../../notebook/10_kumo_spline.ipynb)
- [Riset target KS OOF 0,54](../../notebook/12_goal054_research.ipynb): lima fold identik, kandidat interim KS **0,52930** vs **0,52851** (5/5 fold membaik), kode model dan audit; [progres dan batas interpretasi](../../artifacts/optimization/goal054/README.md). Target belum tercapai.
- [Kandidat versi 5](../../results/teman/submission_v5.csv), versi 5; KS OOF 0,52446 dibanding versi 4 0,52293, public **0,55680** menurut laporan pengguna
- [Submission default](../../results/teman/submission_v4.csv), versi 4; `submission.csv` berisi prediksi yang sama. Versi 5 belum memenuhi syarat promosi minimal enam fold membaik
- [Checkpoint public 0,55845](../../results/teman/submission_v3.csv), terkonfirmasi melalui API Kaggle
- [Laporan validasi versi 4](../../artifacts/optimization/v4/laporan_submission_v4.md)
- [Laporan bergrafik](../../artifacts/laporan_analisis.html), satu berkas HTML dengan grafik tertanam
- [Laporan Markdown](../../artifacts/laporan_analisis.md)
- [Notebook analisis](../../notebook/analisis_logika.ipynb)
- [Tabel hasil lengkap](../../artifacts/tables/)
- [Audit kualitas dan waktu](../../artifacts/quality_audit.json)
- [Ringkasan JSON](../../artifacts/summary.json)

Ada 70.000 aplikasi latih, 30.000 aplikasi uji, dan 9.507 target positif pada train (13,58%). Pada baseline XGBoost berparameter tetap, ROC-AUC holdout stratified meningkat dari 0,6411 untuk aplikasi saja menjadi 0,8302 dengan seluruh riwayat; ROC-AUC temporal 0,8258. Metrik resmi kompetisi adalah **Kolmogorov-Smirnov (KS)** sesuai Evaluation yang diberikan pengguna. KS baseline holdout 0,50671, sedangkan skor publik yang diperoleh pengguna adalah 0,52623. Tidak ada submission yang dikirim otomatis oleh skrip.

Eksperimen tambahan: [Kumo Large](../../artifacts/optimization/kumo/laporan_kumo.md), KS OOF tiga fold **0,52807**, dibanding V5 **0,52529** pada fold identik. [CSV Kumo](../../results/teman/submission_kumo.csv) dan [notebook 08](../../notebook/08_kumo_tabular.ipynb) tersedia; public **0,56262**, dikonfirmasi melalui API Kaggle pada 18 Oktober 2026.

## Reproduksi

Lingkungan yang dipakai tercatat pada `artifacts/summary.json`; gunakan Python 3.10 atau 3.11 dengan versi paket yang tercantum.

```bash
python -m pip install -r environments/teman/requirements.txt
python scripts/download_data.py
python scripts/analyze.py
```

Skrip unduh mencari berkas `*kaggle*.json` di direktori proyek, kemudian `~/.kaggle/kaggle.json` atau `~/.config/kaggle/kaggle.json`. Jalur lain dapat diberikan secara eksplisit:

```bash
python scripts/download_data.py --credentials '/path/to/kaggle.json'
```

Kredensial juga dapat disediakan melalui environment Kaggle. Isi kredensial tidak dicetak atau dimasukkan ke laporan/notebook. Data mentah sekitar 949 MB; berkas yang sudah lengkap dilewati saat unduh ulang. Manifest unduhan disimpan di `data/download_manifest.json`.

```bash
# Gunakan agregasi yang sudah tersedia, jalankan ulang EDA dan baseline:
python scripts/analyze.py --reuse-features

# Perbarui laporan dari fitur dan benchmark yang sudah tersimpan:
python scripts/analyze.py --report-only

# Analisis tanpa melatih baseline:
python scripts/analyze.py --reuse-features --skip-model

# Dependensi tambahan untuk menjalankan notebook:
python -m pip install -r environments/teman/requirements-notebook.txt
```

`--report-only` menggunakan fitur dan benchmark yang sudah ada; gunakan setelah keduanya dibangun dari versi data yang sama. `--skip-model` menghasilkan laporan tanpa bagian baseline. Notebook EDA lama di root adalah pembaca hasil. [Notebook pipeline baru](../../notebook/README_teman.md) memuat EDA, kode agregasi, training, validasi, prediksi test, dan ekspor CSV untuk seluruh versi.

## Aturan agregasi

Cutoff analisis adalah waktu keputusan aplikasi terpilih. Observasi biro harus sudah dilaporkan dan tersedia sebelum cutoff; rekening hanya memakai bulan yang selesai; riwayat aplikasi, cicilan, dan transaksi memakai informasi sebelum cutoff. Fitur snapshot pembayaran mengikuti definisi kamus data. Seluruh waktu observasi diperiksa terhadap cutoff.

Transaksi e-commerce tersimpan per barang. `order_total` dideduplicasi per pesanan dan hanya completed dihitung sebagai belanja. Saldo antarprovider biro tidak dijumlahkan karena cakupan utangnya dapat tumpang tindih. ID dan tanggal mentah dikeluarkan dari model. Nilai kosong struktural dipertahankan dan indikator ketersediaan riwayat dibentuk.

Contoh submission memiliki urutan berbeda dari test. Fungsi `align_submission` pada notebook EDA lama menunjukkan join melalui `application_id` dengan pemeriksaan ID unik dan probabilitas. Notebook pipeline baru menjalankan pemeriksaan yang sama dan menghasilkan `submission.csv` pada folder output masing-masing versi. Tidak ada pengiriman otomatis ke Kaggle.

## Membuat submission

```bash
python scripts/create_submission.py
```

Skrip melatih baseline semua riwayat pada 70.000 label, lalu membuat `submission.csv` di direktori proyek dengan 30.000 probabilitas dalam urutan contoh submission. Model disimpan di `artifacts/submission_model.joblib`; parameter, ringkasan probabilitas, checksum, dan hasil pemeriksaan tersimpan di `artifacts/submission_metadata.json`. Skrip tidak mengirim berkas ke Kaggle.

Data mentah, fitur hasil agregasi, prediksi validasi per baris, dan model tersimpan dikecualikan dari Git. Laporan, grafik, tabel ringkasan, dan metadata eksperimen disertakan agar hasil dapat ditinjau langsung di GitHub. Berkas kredensial dikecualikan secara khusus dalam `.gitignore`.

## Optimasi untuk KS

```bash
python -m pip install -r environments/teman/requirements-optimization.txt
python scripts/extend_features.py
python scripts/optimize_ks.py --features base
python scripts/optimize_ks.py --features extended
python scripts/optimize_ks.py --features compact
python scripts/finalize_ks_submission.py
python scripts/crossvalidate_ks.py
```

Fitur extended menambahkan riwayat/tren biro, beban pinjaman aktif, keterlambatan terbaru, tren rekening, dan belanja pada beberapa jendela waktu. Compact memakai fitur awal ditambah fitur ber-gain tinggi yang dipelajari pada bagian development. Semua waktu observasi tetap sebelum keputusan; audit tersedia di `artifacts/extended_feature_audit.json`.

Tuning memakai 42.000 baris train dan 10.500 baris validasi. Holdout awal 17.500 baris dipertahankan untuk konfirmasi parameter dan bobot yang sudah dikunci. Tahap selanjutnya menjalankan validasi silang tiga fold pada kandidat berparameter tetap dan memilih bobot gabungan berdasarkan rata-rata KS fold. Karena pemilihan bobot menggunakan OOF, skor CV merupakan estimasi validasi, bukan skor test akhir yang bebas optimisme seleksi.

Hasil lengkap tersedia pada `artifacts/optimization/`. `submission_v2.csv` berasal dari kandidat tuning yang dilatih pada seluruh label; `submission_ks.csv` berasal dari rata-rata model tiga fold. `submission.csv` diperbarui ke versi CV jika KS rata-rata fold dan KS OOF melampaui baseline pada split yang sama. Baseline asli dipertahankan sebagai `artifacts/optimization/submission_baseline_0_52623.csv`.

## Submission versi 3

Skor public `submission_ks.csv` terkonfirmasi **0,54710** melalui API Kaggle. Checkpoint persisnya disimpan di `artifacts/optimization/submission_public_0_54710.csv`.

Versi 3 menambahkan harga per unit dan komposisi belanja per kategori, variasi belanja bulanan, shrinkage keterlambatan historis, serta cakupan cicilan oleh arus kas. Dua model baru memakai pembobotan kelas dan sepuluh fold dengan parameter yang sudah dikunci. Prediksi digabungkan dengan versi 0,54710, masing-masing sepertiga.

```bash
python scripts/build_features_v3.py
python scripts/refine_ks.py
python scripts/refine_ks.py --balanced
python scripts/select_refinement.py
python scripts/train_tenfold_submission.py
```

`submission_v3.csv` adalah checkpoint public versi 3. Pada tahap tersebut, `submission.csv` diperbarui ke berkas yang sama setelah seluruh pemeriksaan berhasil. Perbandingan yang memakai tiga fold sama meningkatkan rata-rata KS dari 0,51492 ke 0,52285. KS OOF keseluruhan tahap akhir 0,51901, dibanding 0,51147 untuk versi sebelumnya; model baru memakai 90% baris per fold, dibanding 67% sebelumnya. Kedua angka rata-rata fold 3 dan 10 tidak dibandingkan langsung. Skor public versi 3 adalah **0,55845**, terkonfirmasi melalui API Kaggle, meningkat 0,01135 dari versi 0,54710.

Definisi, bobot yang dikunci, seluruh metrik dan model akhir ada di `artifacts/optimization/tenfold/`. Laporan lengkap: `artifacts/optimization/tenfold/laporan_submission_v3.md`. Hasil validasi dapat mengandung optimisme seleksi karena fitur, jumlah pohon, dan bobot dipilih dari label training yang sama; tidak ada label test yang digunakan.


## Submission versi 4

`submission_v4.csv` menggabungkan versi 3, CatBoost dengan fitur baru, dan regresi logistik dengan spline, masing-masing sepertiga. `submission.csv` memuat prediksi versi 4; checkpoint versi 3 tetap tersedia terpisah. Skor public versi 4 belum tersedia dan tidak ada upload otomatis ke Kaggle.

Fitur baru merangkum harga yang dinormalisasi menurut kategori, frekuensi keterlambatan pada cicilan yang cukup umur untuk mencapai ambang DPD, serta skor biro dengan bobot setara antarprovider. Seluruh waktu observasi diperiksa sebelum keputusan. Normalisasi dan prior historis tidak memakai label target; imputasi dan transformasi spline dipelajari pada bagian training setiap fold.

Seleksi pada tiga fold meningkatkan KS OOF dari 0,51970 menjadi 0,52411, dengan peningkatan pada ketiga fold. Pada tahap akhir sepuluh fold, KS OOF meningkat dari **0.51901** menjadi **0.52293**, dan KS membaik pada **9 dari 10 fold**. Fitur, jumlah pohon, dan bobot dipilih dari label training yang sama; hasil lokal mengandung optimisme seleksi dan tidak menjamin skor leaderboard.

```bash
# Setelah fitur dan prediksi versi 3 tersedia:
python scripts/build_features_v4.py
python scripts/evaluate_v4.py
python scripts/evaluate_spline_v4.py
python scripts/select_v4.py
python scripts/train_v4_candidate.py --candidate spline
python scripts/train_v4_candidate.py --candidate v4_cat5_balanced_500
python scripts/export_v4_submission.py
```

[Laporan versi 4](../../artifacts/optimization/v4/laporan_submission_v4.md) memuat perbandingan pada fold yang sama, grafik KS/ROC, bobot gabungan, dan checksum. [Metadata submission](../../artifacts/optimization/v4/submission_v4_metadata.json), [audit fitur](../../artifacts/optimization/v4/feature_audit.json), dan [perbandingan kelompok customer](../../artifacts/optimization/v4/cohort_metrics.json) juga disertakan. Model akhir dan prediksi validasi per baris disimpan lokal dan dikecualikan dari Git.

## Kandidat submission versi 5

Versi 5 memakai **75% versi 4 + 25% TabM**, dilatih dari label kompetisi tanpa model pretrained atau data eksternal. Empat fitur tambahan merangkum harga per unit dari semua basket, termasuk cancelled sebagai pilihan produk; cancelled tetap tidak dihitung sebagai belanja. Early stopping TabM berada di dalam training setiap outer fold, lalu model dilatih ulang pada seluruh training fold dengan jumlah epoch yang dipilih.

Pada sepuluh fold yang sama, KS OOF meningkat dari **0,522934 menjadi 0,524464**, rata-rata KS fold dari **0,529401 menjadi 0,530121**, dan ROC-AUC dari 0,844598 menjadi 0,845405. Lima dari sepuluh fold membaik. Kandidat ini belum memenuhi syarat promosi enam fold, sehingga **`submission_v5.csv` disediakan terpisah dan `submission.csv` tetap versi 4**. Bootstrap berpasangan masih mencakup nol; peningkatan kecil ini belum membuktikan keuntungan yang stabil atau skor public lebih tinggi. Public versi 5 **0,55680**, dilaporkan pengguna pada 2026-10-15, turun **0,00165** dibanding versi 3 (0,55845). Kenaikan validasi lokal belum diikuti kenaikan public. Versi 3 tetap checkpoint public terbaik yang tercatat; skor public versi 4 belum tersedia. Tidak ada submit otomatis.

[Notebook 06](../../notebook/06_submission_v5.ipynb) memuat seluruh pipeline hingga CSV. [Laporan versi 5](../../artifacts/optimization/v5/laporan_submission_v5.md) menjelaskan pemilihan eksploratif, hasil eksperimen yang tidak dipromosikan, dan keterbatasan validasi. Model tersimpan berhasil mereproduksi OOF dan prediksi test secara persis.

```bash
# Python 3.10; setelah fitur dan model versi 4 tersedia:
python -m pip install -r environments/teman/requirements-v5.txt -r environments/teman/requirements-notebook.txt
python scripts/build_features_v5.py
python scripts/evaluate_v5.py --group tabm
python scripts/train_v5_candidate.py --candidate v5_tabm_weight1 --new-weight .25
python scripts/export_v5_submission.py

# Buat ulang notebook 06 dan jalankan seluruh sel:
python scripts/build_review_notebook_v5.py --execute
```

Untuk mengulang screening lain, gunakan `evaluate_v5.py --group` dengan `spline_plus`, `ebm`, `xgb`, `specialist`, `basket`, `shallow`, `neuralcore`, atau `smoothing`. `select_v5.py` mereview bobot kasar 25%/50% dengan aturan seleksi ketat; tidak ada kandidat yang lolos aturan itu pada eksperimen ini. Kandidat TabM diteruskan sebagai konfirmasi eksploratif karena KS gabungan dan rata-ratanya meningkat sedikit pada tiga fold. Fitur, model, serta bobot menggunakan label train yang sama sehingga hasil mengandung optimisme seleksi.

## Reverse engineering generator dan jitter

[Notebook 07](../../notebook/07_reverse_engineering_rng.ipynb) dan [laporan rinci](../../artifacts/reverse_engineering/laporan_reverse_engineering.md) memisahkan aturan observasi yang terkonfirmasi dari estimasi noise dan hipotesis generator. **Seed RNG asli belum ditemukan.** Probe mencakup stream usia, template hash ID, RNG per customer, ID sebagai raw output, dan uji cloning state MT19937 dengan positive control.

Cicilan terpilih tepat sama dengan `ceil` formula anuitas. Semua unit price merupakan kelipatan Rp1.000 sebelum dikalikan quantity. Delay keputusan cocok dengan tiga rentang uniform 0,2–3, 6–30, dan 36–120 jam, dengan proporsi sekitar 45%/35%/20% dan dua celah tanpa observasi. SD noise pada log harga diperkirakan 0,654; pada skor biro sekitar 23 poin. Angka noise dan efek laten adalah estimasi, bukan nilai RNG asli yang diketahui.

Sebanyak 21 fitur laten/ketidakpastian diuji dengan CatBoost dan spline pada tiga fold yang sama. Belum ada peningkatan KS yang konsisten, sehingga submission tidak berubah. Prototype di `synthetic_generator_prototype.py` memakai seed demo yang dipilih sendiri dan hanya meniru sebagian pola data; tidak menghasilkan label test asli.

```bash
python scripts/reverse_rng_probe.py --seed-limit 10000
python scripts/reverse_synthetic_patterns.py
python scripts/validate_reverse_features.py
python scripts/report_reverse_engineering.py
python scripts/build_reverse_notebook.py --execute
```

Audit mentah dan simulasi notebook dapat berjalan tanpa model sebelumnya. Bagian validasi memerlukan fitur dan OOF versi 5. Hasil audit ada di `artifacts/reverse_engineering/`; fitur per customer dan model tetap disimpan lokal serta dikecualikan dari Git.

## Eksperimen Kumo Tabular

[Submission Kumo murni](../../results/teman/submission_kumo.csv) memakai Kumo Large, 90 fitur, delapan
anggota ensemble, dan seluruh training setiap fold (46.666–46.667 konteks). Pada
tiga fold yang identik dengan seleksi V5, KS OOF naik dari **0,52529** ke **0,52807**;
rata-rata KS fold naik dari **0,52752** ke **0,52907**, dengan **3/3 fold membaik**.
Ini berbeda dari validasi akhir V5 sepuluh fold. Skor public Kumo murni adalah **0,56262**,
dikonfirmasi melalui API Kaggle pada 18 Oktober 2026: naik **0,00417** terhadap V3
(0,55845) dan **0,00582** terhadap V5 (0,55680). Skor public campuran belum diketahui.
[Campuran 75% V5 + 25% Kumo](../../results/teman/submission_kumo_blend.csv) tersedia terpisah.
[Laporan lengkap](../../artifacts/optimization/kumo/laporan_kumo.md) mencatat seluruh konfigurasi.

Kumo dijalankan lokal memakai bobot pretrained NVIDIA `nvidia/Kumo-Tabular`, revisi
`v1.0.1`, melalui `structured-data-models==0.2.1`. Lingkungan Python 3.11 terpisah
menghindari perubahan pada dependensi historis V5. `fit` membangun preprocessing dan
cache konteks berlabel; eksperimen ini tidak melakukan fine-tuning bobot.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r environments/teman/requirements-kumo.txt
# Setelah fitur V5 tersedia (OOF V5 opsional untuk perbandingan):
.venv/bin/python scripts/kumo_tabular.py --size large --features v5 --context 50000 --estimators 8 --batch-size 8192 --test
.venv/bin/python scripts/report_kumo.py --export large_v5_ctx50000_e8 --blend-weight 0.25
.venv/bin/python -m ipykernel install --user --name logika-kumo --display-name "LOGIKA Kumo (Python 3.11)"
```

[Notebook 08](../../notebook/08_kumo_tabular.ipynb) menampilkan EDA, seluruh kode agregasi
fitur, konteks Kumo, validasi, dan ekspor. Tiga fold identik dengan seleksi V5
(`StratifiedKFold`, seed 8247). Baris validation tidak masuk konteks prediksinya.
Preprocessing dipelajari dari konteks, dan proporsi kelas konteks mengikuti data asli.
Hasil tersedia di `artifacts/optimization/kumo/`; submission kandidat terpisah
`submission_kumo.csv` tidak mengganti `submission.csv`.

Sumber: [model NVIDIA](https://huggingface.co/nvidia/Kumo-Tabular) dan
[kode SDM](https://github.com/NVIDIA/structured-data-models).

## Eksperimen lanjutan Kumo

[Notebook 09](../../notebook/09_kumo_improvement.ipynb) memperlihatkan delapan kandidat
pada tiga fold awal, lalu dua uji pada lima fold yang sama dengan baseline.
Campuran 75% Kumo baseline + 25% konteks seimbang menaikkan rata-rata KS lima fold
sebesar 0,00029, tetapi pooled KS turun 0,00020. Konteks penuh 56.000 baris
menaikkan rata-rata KS sebesar 0,00033, tetapi pooled KS turun 0,00017.
Keduanya membaik pada 3/5 fold dan belum memenuhi syarat kedua ringkasan KS naik.
Checkpoint public **0,56262** tetap tersedia; eksperimen ini tidak mengekspor CSV baru.

Baseline pooled KS lima fold adalah **0,52722**. Angka tersebut hanya dibandingkan
dengan kandidat pada lima fold identik. Uji konteks penuh dilakukan setelah
campuran awal gagal dan memakai lima split yang sama, sehingga hasil tetap
mengandung optimisme pemilihan model. [Laporan lengkap](../../artifacts/optimization/kumo_improvement/laporan_improvement.md)
dan metadata menyimpan percobaan yang tidak lolos.

```bash
# Dalam lingkungan Python 3.11 Kumo; notebook membangun fitur jika belum ada:
python scripts/build_kumo_improvement_notebook.py --execute
# Jalur CLI setelah fitur V5 dan reverse tersedia:
python scripts/run_kumo_improvement.py
python scripts/verify_kumo_improvement.py
```

## Gabungan Kumo–spline dan pilot fine-tuning

[Notebook 10](../../notebook/10_kumo_spline.ipynb) melatih spline dengan fitur laten dan
menggabungkannya dengan Kumo, masing-masing 25% dan 75%. Tiga fold awal menaikkan
pooled KS dari 0,52807 menjadi 0,52914 serta rata-rata KS dari 0,52907 menjadi
0,53037. Pada lima fold, pooled KS naik dari **0,52722 menjadi 0,52851**, tetapi
rata-rata KS turun dari **0,53198 menjadi 0,53095**, dengan hanya **1/5 fold membaik**.

[submission_kumo_spline_exploratory.csv](../../results/teman/submission_kumo_spline_exploratory.csv)
awalnya disediakan sebagai kandidat **eksploratif** karena belum memenuhi aturan
promosi lokal. Pengguna kemudian melaporkan public **0,56393** pada 18 Oktober 2026,
naik **0,00131** dari Kumo murni. CSV dengan checksum yang sama dipertahankan sebagai
checkpoint public terbaru. CSV berisi 30.000 probabilitas. [Laporan](../../artifacts/optimization/kumo_hybrid/laporan_hybrid.md)
mencatat semua metrik dan batas interpretasi. Campuran TabM langsung juga diuji
dari OOF historis dan belum lolos screening.

[Notebook 11](../../notebook/11_kumo_finetune_pilot.ipynb) mendokumentasikan satu pilot
fine-tuning dua layer ICL terakhir dan kepala klasifikasi. Validasi internal memilih
checkpoint langkah 0, sehingga pilot memakai kembali bobot pretrained dan tidak
membuat submission baru. Hasil ini hanya mencakup satu outer fold.

Keterangan "public belum diketahui" pada metadata ekspor, notebook, dan laporan
awal mencatat keadaan
saat eksperimen dibuat; [pencatatan hasil baru](../../artifacts/optimization/kumo_hybrid/public_result_056393.json)
memisahkan umpan balik Kaggle dari skor validasi.

Riset berikutnya menargetkan **pooled KS OOF ≥ 0,54** pada lima fold seed 31027.
[Rencana riset](../../artifacts/optimization/goal054/research_plan.json) memuat baseline,
protokol, dan keluarga eksperimen awal. Target belum tercapai.

```bash
# Dalam lingkungan Python 3.11 Kumo, setelah fitur tersedia:
python scripts/hybrid_kumo.py --export-exploratory
python scripts/report_hybrid_kumo.py
python scripts/build_hybrid_kumo_notebook.py --execute
python scripts/build_finetune_pilot_notebook.py --execute
```
