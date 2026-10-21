# Panduan review notebook LOGIKA

Mulai dari **01** untuk mempelajari alur paling sederhana. Baca **02–06** untuk melihat perubahan fitur, validasi, dan model. **08** memuat Kumo Tabular; **09** mereview eksperimen lanjutannya. **10** memuat checkpoint public 0,56393; **12** memuat riset terbaru dan kandidat koreksi dengan KS OOF 0,52930. **07** membahas reverse engineering data sintetis. Submission default tetap **05**. Kode EDA, agregasi fitur, preprocessing, training, prediksi test, dan ekspor CSV tersedia di sel notebook.

| Notebook | Pipeline yang direview | Validasi / model final | Public historis |
|---|---|---|---:|
| [01 — Baseline XGBoost](01_baseline_xgboost.ipynb) | Profil dan agregasi seluruh riwayat | Holdout 25%; model final dilatih pada seluruh train | 0,52623 |
| [02 — Tuning KS versi 2](02_tuning_ks_v2.ipynb) | Fitur extended dan perbandingan kandidat pada data tuning | Core 42.000 / tuning 10.500 / confirmation 17.500; final LightGBM 486 pohon | Belum tercatat |
| [03 — Ensemble KS](03_ensemble_ks.ipynb) | Pemilihan bobot LightGBM dan CatBoost | Tiga fold; 25% LightGBM + 75% CatBoost | 0,54710 |
| [04 — Submission versi 3](04_submission_v3.ipynb) | Harga per kategori, shrinkage, arus kas, dan class weighting | Dua model baru sepuluh fold, digabungkan dengan versi KS | 0,55845 |
| [05 — Submission versi 4](05_submission_v4.ipynb) | Harga dinormalisasi, cicilan matang, dan model spline | Versi 3 + CatBoost baru + spline, masing-masing sepertiga | Belum tercatat |
| [06 — Kandidat versi 5](06_submission_v5.ipynb) | Harga semua basket dan TabM, early stopping di dalam training fold | 75% versi 4 + 25% TabM; KS OOF 0,52446 | 0,55680 (laporan pengguna) |

Skor public di atas berasal dari file historis. Angka validasi di notebook adalah KS lokal, bukan skor leaderboard. `submission_v2.csv` adalah alternatif yang pernah dibuat; skor 0,54710 berasal dari `submission_ks.csv`.

## Cara menjalankan

1. Instal paket dari direktori proyek: `python -m pip install -r environments/teman/requirements-optimization.txt -r environments/teman/requirements-notebook.txt`.
2. Unduh data sekali: `python scripts/download_data.py`. Kredensial Kaggle tetap di file lokal yang dikecualikan dari Git; jangan menyalin key ke sel notebook.
3. Buka notebook melalui Jupyter atau VS Code, pilih kernel Python yang memiliki paket tersebut, lalu **Restart & Run All**.
4. Ambil `submission.csv` pada folder output yang ditampilkan sel terakhir.

Untuk **06**, gunakan Python 3.10 dan tambahkan `environments/teman/requirements-v5.txt`. Lingkungan referensi memakai PyTorch 1.13.1+cu116 dan GPU A100. CPU dapat dipakai, dengan durasi dan kemungkinan pembulatan berbeda. Skor validasi versi 5 naik dibanding versi 4, tetapi hanya lima fold membaik dari syarat enam; file kandidat terpisah belum mengganti `submission.csv` di root.

| Notebook | Lokasi output dari direktori proyek |
|---|---|
| 01 | `artifacts/notebook_runs/01_baseline/submission.csv` |
| 02 | `artifacts/notebook_runs/02_v2/submission.csv` |
| 03 | `artifacts/notebook_runs/03_ks/submission.csv` |
| 04 | `artifacts/notebook_runs/04_v3/submission.csv` |
| 05 | `artifacts/notebook_runs/05_v4/submission.csv` |
| 06 | `artifacts/notebook_runs/06_v5/submission.csv` |

Model historis, data mentah, fitur parquet, dan prediksi OOF per baris tidak disertakan di GitHub. Pada clone baru, sel agregasi membangun fitur yang belum ada, dan `.fit()` melatih model yang belum ada. Versi 3 dan 4 memerlukan lebih banyak pelatihan karena juga membangun komponen ensemble sebelumnya.

## Cara membaca kode

- Bagian **EDA** menunjukkan ukuran data, missing value, ketidakseimbangan target, pola bulanan, dan perbedaan kelompok pekerjaan.
- Bagian **fitur** dipecah per sumber data. Fokus pada cutoff, `groupby`, denominator rasio, dan join. Semua kode agregasi terlihat di sel, termasuk ketika cache membuatnya dilewati.
- Bagian **preprocessing** menjelaskan bagaimana kategori, missing numerik, dan class weighting ditangani.
- Bagian **validasi** membedakan tuning, confirmation, OOF, dan model final. Kandidat serta grid bobot historis juga ditampilkan.
- Bagian **prediksi test dan ekspor** menghitung ulang probabilitas dari model, menggabungkan anggota ensemble, memeriksa ID/probabilitas, lalu menulis CSV.

Gunakan panel **Table of Contents / Outline** untuk melompat antarbagian. Bagian agregasi cukup panjang karena semua kode tersedia langsung untuk direview; jika sudah memahami fitur dasar pada 01, Anda dapat fokus pada tambahan fitur dan model di versi selanjutnya.

## Mengendalikan cache dan eksperimen

| Flag di sel persiapan | Jika `True` |
|---|---|
| `FORCE_REBUILD_FEATURES` | Membaca ulang riwayat dan membangun kembali fitur |
| `FORCE_RETRAIN_MODELS` | Melatih ulang model, termasuk komponen ensemble sebelumnya; OOF dihitung dari model tersebut |
| `FORCE_RECOMPUTE_VALIDATION` | Menghitung ulang prediksi validasi dari model fold yang sesuai |
| `RUN_OPTIONAL_SEARCH` | Mengulang pencarian kandidat/jumlah pohon yang lebih mahal |

Default semua flag adalah `False`. Jika cache tersedia, notebook menggunakan model/OOF historis dan **tetap menghitung ulang prediksi test dari model**. Jika Anda mengubah parameter, daftar fitur, preprocessing, atau data mentah, aktifkan `FORCE_RETRAIN_MODELS`; jika mengubah agregasi, aktifkan pula `FORCE_REBUILD_FEATURES`. Beberapa ketidaksesuaian parameter cache diperiksa dan menghentikan eksekusi dengan pesan untuk melatih ulang.

Parameter dan bobot final historis ditulis eksplisit. Menjalankan pencarian opsional menampilkan hasil pencarian; untuk memakai pilihan baru, ubah konfigurasi final pada sel model. Ini membuat perbedaan antara reproduksi dan eksperimen mudah diperiksa.

Model validasi tidak berlatih pada baris validasinya. Namun fitur compact, jumlah pohon, dan bobot pernah dipilih dari label training yang sama, sehingga hasil CV dapat mengandung optimisme seleksi. Cache OOF tidak berasal dari model yang dilatih pada seluruh label.

## Pemeriksaan yang telah dilakukan

Kelima notebook dijalankan sampai ekspor dengan model historis yang tersedia. Semua CSV memiliki 30.000 ID sesuai sample, probabilitas valid, serta prediksi dan SHA-256 yang sama dengan checkpoint masing-masing. Sel fitur juga dijalankan ulang dari data mentah di direktori terpisah: fitur dasar, extended, versi 3, versi 4, dan tabel biro terakhir identik dengan tabel eksperimen asli. Rincian verifikasi tersedia pada [verification.json](verification.json).

Notebook EDA lama di root tetap merupakan pembaca laporan. Enam notebook dalam folder ini adalah pipeline yang dapat membuat submission. Notebook 06 juga dijalankan sampai selesai: CSV identik dengan `submission_v5.csv`, dan prediksi OOF/test direproduksi persis dari sepuluh model TabM tersimpan. Detail pemeriksaan versi 5 tersedia di `artifacts/optimization/v5/`.

Untuk meregenerasi notebook dari kode sumber: `python scripts/build_review_notebooks.py`. Untuk sekaligus mengeksekusinya: `python scripts/build_review_notebooks.py --execute`.

Untuk notebook 06: `python scripts/build_review_notebook_v5.py --execute`. Generator 06 memakai sel pipeline notebook 05 sebagai komponen versi 4, lalu menambahkan kode fitur dan TabM secara langsung. Jalankan generator 01–05 terlebih dahulu jika ingin meregenerasi seluruh notebook dari awal.

## Notebook audit generator

[07 — Reverse engineering RNG dan jitter](07_reverse_engineering_rng.ipynb) mereview struktur sintetis, rounding, probe seed dan ID, inferensi noise harga/biro/income, prior keterlambatan, serta generator pendekatan. Seed asli belum ditemukan. Kode seluruh langkah tersedia langsung di sel; simulasi memakai seed demo sendiri.

Bagian audit dapat dijalankan dari data mentah dengan paket optimization dan notebook. Validasi CatBoost/spline memerlukan fitur serta OOF versi 5; bila prasyarat belum tersedia, tabel validasi historis tetap ditampilkan. Notebook ini menulis output audit di `artifacts/notebook_runs/07_reverse/` dan tidak menghasilkan atau mengganti submission.

Untuk membangun dan mengeksekusi notebook 07: `python scripts/build_reverse_notebook.py --execute`. Flag mengatur pengulangan probe RNG, audit mentah, validasi model dan retraining. [Laporan reverse engineering](../artifacts/reverse_engineering/laporan_reverse_engineering.md) menjelaskan batas identifikasi dan hasil pengujian fitur.

Notebook 07 telah dijalankan seluruhnya: 21 fitur dari data mentah identik dengan eksperimen, seluruh OOF CatBoost/spline serta metriknya direproduksi persis, dan checksum kedua submission sebelumnya tidak berubah. Rincian ada pada [verifikasi notebook 07](../artifacts/reverse_engineering/notebook_verification.json).

## 08 — Kumo Tabular

[08_kumo_tabular.ipynb](08_kumo_tabular.ipynb) memakai lingkungan Python 3.11 terpisah
(`environments/teman/requirements-kumo.txt`) dan GPU CUDA. Kode EDA dan agregasi fitur terlihat di sel,
dilanjutkan inferensi Kumo pretrained dan perbandingan dengan tiga fold seleksi V5.
`fit` membentuk cache konteks berlabel; bobot model tidak dilatih ulang. Data tetap lokal.

OOF V5 `selection_oof_tabm.parquet` dipakai jika tersedia untuk perbandingan pada baris
identik. Tanpa OOF V5, Kumo tetap berjalan hingga ekspor CSV. Aktifkan
`FORCE_RERUN_KUMO` untuk menghitung ulang inferensi; cache diperiksa terhadap resep dan
fingerprint data. Output terpisah adalah `submission_kumo.csv` di root proyek.

Regenerasi: `python scripts/build_kumo_notebook.py --variant large_v5_ctx50000_e8`.
Notebook 08 telah dieksekusi sampai ekspor: **48 sel kode, tanpa error**.
Rincian: `artifacts/optimization/kumo/notebook_verification.json`.

Default menjalankan Medium (32.768 konteks, 4 anggota ensemble) dan Large (seluruh
training fold, 8 anggota), memakai cache prediksi yang resep dan datanya cocok.
`RUN_ALL_VARIANTS=False` membatasi ke konfigurasi terpilih. Kumo Large menghasilkan
KS OOF **0,52807**, dibanding V5 **0,52529** pada tiga fold identik, dan membaik pada
ketiga fold. Notebook mengekspor Kumo murni dan campuran 25% Kumo secara terpisah.
Skor tersebut adalah validasi lokal. Public score Kumo murni **0,56262** dikonfirmasi
melalui API Kaggle pada 18 Oktober 2026. Skor campuran belum diketahui; skor historis
hanya digunakan kembali jika checksum CSV cocok dengan catatan hasil.

## 09 — Eksperimen lanjutan Kumo

[09_kumo_improvement.ipynb](09_kumo_improvement.ipynb) memakai lingkungan yang sama
dengan 08. Sel memperlihatkan seluruh kode fitur dan model, termasuk TabICLv2
dengan checkpoint resmi yang dipatok ke commit tertentu. Sampling konteks mengambil
hanya baris outer-training; query dibuat untuk susunan anggota ensemble yang sama.

Default membaca laporan screening yang tersimpan, lalu mereproduksi lima fold
baseline dan kandidat konteks penuh dari cache yang datanya cocok atau menghitungnya
jika belum tersedia. `RUN_SCREEN=True` mengulang seluruh screening tiga fold.
`FORCE_RERUN_IMPROVEMENT=True` menghitung ulang inferensi. Semua prediksi validation
bersifat OOF, tetapi pemilihan berulang memakai label training yang sama.

Campuran 75/25 dan penambahan konteks belum memenuhi kriteria pooled KS dan rata-rata
KS sama-sama naik. Karena itu sel ekspor mencetak alasan tidak membuat kandidat CSV.
Checkpoint `submission_kumo.csv` dengan public **0,56262** dipertahankan.
Rincian ada di [laporan eksperimen](../artifacts/optimization/kumo_improvement/laporan_improvement.md).

Regenerasi dan eksekusi: `python scripts/build_kumo_improvement_notebook.py --execute`.
Audit memeriksa KS secara independen dengan SciPy, pembagian konteks/validation,
checksum checkpoint, dan kesesuaian kandidat CSV jika kelak memenuhi kriteria.

Notebook 09 sudah dieksekusi: **58 sel kode, tanpa error**. Sel ekspor dengan benar
melewati pembuatan CSV karena kandidat tidak lolos. Semua checksum submission lama
dan notebook analisis pengguna tetap sama. [Verifikasi](../artifacts/optimization/kumo_improvement/notebook_verification.json).

## 10 — Kumo dan spline fitur laten

[10_kumo_spline.ipynb](10_kumo_spline.ipynb) memuat pipeline lengkap gabungan
**75% Kumo Large + 25% spline**, memakai lingkungan Python 3.11 Kumo. Kode spline
meliputi transformasi log bertanda, imputasi, spline kubik lima knot, standardisasi,
indikator missing, one-hot kategori, serta regresi logistik C=0,05.

Tiga fold screening memakai seed 8247, dan lima fold berikutnya seed 31027. Data
input baseline Kumo diperiksa melalui fingerprint; kedua model dibandingkan pada
baris yang sama. Model spline disimpan dan prediksi test dihitung ulang untuk
memeriksa cache. Dataset dan lima split pernah dipakai untuk eksperimen lain;
hasil tetap mengandung optimisme seleksi.

Sel ekspor hanya membuat `submission_kumo_v2.csv` jika pooled KS dan rata-rata KS
naik serta minimal 3/5 fold membaik. Salinan bernama `submission.csv` berada pada
`artifacts/notebook_runs/10_kumo_spline/`. Checkpoint public 0,56262 tetap disimpan.
Pada saat notebook dibuat, skor public kandidat belum diketahui. Pengguna kemudian
melaporkan **0,56393** pada 18 Oktober 2026; [pencatatan hasil](../artifacts/optimization/kumo_hybrid/public_result_056393.json)
mengikat skor itu ke checksum CSV yang sama. Output notebook tetap menjadi catatan
keadaan pada saat eksekusi.

Hasil aktual belum lolos aturan promosi: pooled KS naik **0,52722 → 0,52851**,
tetapi rata-rata KS turun **0,53198 → 0,53095**, dan hanya **1/5 fold membaik**.
Dengan `EXPORT_EXPLORATORY=True`, notebook membuat file terpisah
`submission_kumo_spline_exploratory.csv` untuk uji Kaggle. File eksploratif ini tidak
mengganti checkpoint public dan tidak diberi skor public historis.

Regenerasi: `python scripts/build_hybrid_kumo_notebook.py --execute`. Jalur CLI:
`python scripts/hybrid_kumo.py`, setelah fitur V5 dan reverse tersedia.

Notebook 10 sudah dieksekusi: **59 sel kode, tanpa error**. Salinan `submission.csv`
cocok dengan CSV eksploratif di root; seluruh submission lama dan notebook analisis
pengguna tetap utuh. [Verifikasi notebook](../artifacts/optimization/kumo_hybrid/notebook_verification.json)
dan [audit model serta CSV](../artifacts/optimization/kumo_hybrid/verification.json).

## 11 — Pilot fine-tuning Kumo

[11_kumo_finetune_pilot.ipynb](11_kumo_finetune_pilot.ipynb) menunjukkan training
dua layer ICL terakhir beserta norm dan kepala klasifikasi. Episode training memakai
1.024 konteks dan 512 query dari inner-fitting, AdamW, BF16, dan gradient clipping.
Checkpoint 0/32/64/128 dipilih dengan AUC pada 15% inner-validation.

Hanya outer fold pertama yang dijalankan. Checkpoint **langkah 0 terpilih** karena
AUC internal checkpoint hasil training sedikit lebih rendah. Prediksi outer/test
pilot memakai kembali bobot pretrained. Pilot tidak mengekspor submission baru dan
belum menilai fine-tuning pada semua outer fold.

Regenerasi: `python scripts/build_finetune_pilot_notebook.py --execute`.
Pipeline `scripts/run_kumo_finetune.py` dapat meneruskan seluruh fold secara terpisah;
itu merupakan eksperimen tambahan yang tidak dibutuhkan resep notebook 10.

Notebook 11 sudah dieksekusi: **51 sel kode, tanpa error**. [Verifikasi](../artifacts/optimization/kumo_finetune/notebook_verification.json).

## 12 — Riset menuju KS OOF 0,54

[12_goal054_research.ipynb](12_goal054_research.ipynb) menyertakan EDA hingga kode
prediksi, audit, dan ekspor kandidat. Memuat TabPFN 3.5, EBM, koreksi Kumo dan
kalender, serta Kumo Relational dengan riwayat sebelum keputusan. Gunakan
Python 3.11, `environments/teman/requirements-goal054.txt` dan GPU CUDA. Flag training default `False`
untuk review hasil yang tersedia. Aktifkan `BUILD_RELATED_TABLES` untuk membuat
riwayat graf dari raw data; preflight relasional tidak menghitung KS.

Target berlaku untuk **pooled KS seluruh lima fold seed 31027**, bukan satu fold
atau rata-rata fold. Notebook menghitung ulang audit hash/ID dan metrik pada cache.
CSV kandidat goal diekspor jika pooled KS mencapai 0,54. Flag interim dapat mengekspor kandidat yang pooled KS-nya naik dan membaik pada 5/5 fold, dengan status target tetap belum tercapai. Checkpoint public
0,56393 tetap terpisah; public kandidat baru belum diketahui.

Regenerasi dan verifikasi: `python scripts/build_goal054_notebook.py --execute`.
[Progres](../artifacts/optimization/goal054/README.md),
[laporan](../artifacts/optimization/goal054/laporan_riset.md), dan
[verifikasi notebook](../artifacts/optimization/goal054/notebook_verification.json).
