# Perbandingan basis riset — 21 Oktober 2026

**Pembaruan setelah audit162:** user meneruskan izin panitia untuk pseudo-labeling
dan teknik semi-supervised lainnya, lalu meminta pengembangannya. Larangan SSL
dari lampiran awal yang dikutip di laporan ini kini diganti izin tersebut. Data
panitia saja; teacher/threshold/sampling nested, outer untuk evaluasi. Reference
train-only tetap kontrol berguna terhadap preprocessing transduktif. Temuan162
tidak otomatis membuktikan leakage target/future atau menyatakan SSL tidak sah.

Audit dilakukan sebelum pemindahan kode. Sumber primer lokal: inventaris SHA256
`inventaris_awal.csv`, `duplikat_checksum.json`, `git_status_awal.txt`, dan
`git_diff_awal.patch`. Folder teman merupakan **snapshot tanpa `.git`**; tidak ada
riwayat commit yang dapat digabungkan. Versi `utama:v5` dan `teman:v5` berbeda.

## Kelengkapan bukti

| Aspek | Utama | Teman |
|---|---|---|
| File terinventaris | 2.391 (termasuk data/cache lokal) | 508, 19.499.369 byte |
| Data mentah | 9 CSV di `data/`, kamus data; cache `data_pq/` | Tidak ada dalam snapshot |
| Fitur | `eda/features.py`, 232 prediktor (228 numerik + 4 kategori) | `scripts/analyze.py`, `extend_features.py`, `build_features_v3/v4/v5.py`; fitur V5 90 kolom dipilih dari fitur lebih banyak |
| Validasi | `eda/valid2.py`, `run_valid2.py`, `blend2.py` | `crossvalidate_ks.py`, `model_v5.py`, `kumo_tabular.py`, `improve_kumo.py`, `research_*054.py` |
| Notebook | `notebook/`, builder per versi, builder submission | 12 notebook pipeline, 1 notebook pembaca hasil, 8 builder/review builder |
| OOF/test | Banyak parquet ID-bearing di `results/`; beberapa `.npy` lama | **0 parquet**, tidak ada OOF/test per fold yang dapat dipakai langsung |
| Model/cache | Environment dan bobot lokal, cache prediksi dan fitur | Tidak disertakan; metadata fingerprint bukan pengganti prediksi/model |
| Submission | `results/vN/`, bukti skor di handoff | 10 CSV root, laporan dan metadata public (perlu verifikasi checksum/format saat migrasi) |

SHA256 seluruh file yang diaudit memberi **68 kelompok duplikat**, **0 kelompok
lintas sumber**. Jadi tidak ada dasar menyatukan dua implementasi hanya dari nama.
Sebagian besar duplikat utama adalah snapshot resep dan hasil run yang sengaja
dikunci; jangan dibuang. Pada teman, `submission.csv` identik dengan
`submission_v4.csv`; metadata identik pada lokasi berbeda tetap punya peran provenance.

## Fitur dan cutoff

Keduanya memakai 8 tabel panitia, satu selected application/customer. `application_id`
dan `customer_id` adalah join/alignment, bukan prediktor.

- Utama: fitur `bc_/bp_/bh_` memisahkan biro aplikasi terpilih dan historis;
  `rp_` perilaku cicilan/active obligation; `ac_` rekening; `ec_` belanja; `pp_`
  harga relatif kategori. Teman: latest report per provider, debt/income,
  shrinkage late-event matang, equal-order/equal-category log price, completed dan
  cancelled baskets dipisahkan. Indikator `has_*` eksplisit.
- Utama memfilter biro `available_datetime <= cutoff`; snapshot fitur rekening
  memakai bulan selesai (`month + 1mo <= cutoff`), repayment due `<= cutoff`.
  Teman umumnya mensyaratkan timestamp **strict `< cutoff`**, dan mengaudit
  report/available/payment secara terpisah; rekening bulan selesai `<= cutoff`.
  Perbedaan inklusivitas belum terbukti menyebabkan gain: audit153 pada data yang
  tersedia mencatat **0 pelanggaran waktu**. Jangan mengubah resep utama diam-diam.
- Kode teman V3–V5 menghitung prior pembayaran dan reference median kategori dari
  **seluruh customer train** (tanpa target), sebelum outer split. Ini bukan target
  leakage, tetapi fit preprocessing melibatkan covariate outer-valid. Adaptasi baru
  wajib memfit reference pada inner-fit / outer-train masing-masing.
- Audit153: 44 kolom baseline kosong seluruhnya pada thin-file; audit157: float32
  menambah ties pada 99/228 numerik, termasuk 48 baseline100, tetapi **42 kolom
  uang tidak berubah KS univariat**. Tidak ada bukti precision uang menjelaskan gap.

### Tindak lanjut audit162: reference populasi harga

`eda/features.py:v5_features` membangun mean/std kategori per expanding month dari
raw histori seluruh customer sebelum join cutoff, tanpa filter customer train.
Audit162 menghitung keanggotaan saja (tanpa fit mean/std atau model): dari
4.359.817 event, **30,0570%–30,1146%** event reference bulanan berasal dari customer
test. Waktu event tetap sebelum awal bulan keputusan; temuan ini bukan bukti
future leakage atau besaran optimisme skor. Saat audit dilakukan, lampiran awal
masih melarang fitting distribusi test. Setelah izin SSL terbaru, populasi ini
harus dinyatakan sebagai preprocessing transduktif dan diaudit sesuai prosedurnya.

Baseline historis dipertahankan untuk provenance. RunVIPv48 memakai cache lama,
sehingga hasilnya harus dilabeli dengan preprocessing legacy transduktif, bukan
strictly inductive. Reference train-only per fitting partition (inner-fit untuk
selection,outer-train untuk evaluation) adalah kontrol ilmiah yang belum dibuat.
Reference teman juga perlu diikat pada kebijakan eksperimen yang eksplisit.
Baik reference maupun label/threshold/feature selection tidak boleh dituning dari
outer-score; tidak ada gerbang breakthrough yang sudah lolos saat ini. Lihat
[audit162](../../eda/out/162_reference_population_audit.json) dan
[protokol reference](protokol_reference_bersarang.md).

## Input, context, dan inferensi Kumo

| Faktor | Utama | Teman |
|---|---|---|
| Bobot | Kumo-Tabular **large**, SDM0.2.1 | Kumo-Tabular **large**, SDM0.2.1, bobot v1.0.1, metadata commit `3c3e10bbdb590ace29e7026847f92db3c603096d` |
| Fitur | per fold: top-KS univariat + greedy abs Spearman<0,95; 100 numerik +4 kategori | 90 fitur V5 tetap dari recipe JSON; bukan identik baseline100 utama |
| Kategori | float codes; SDM diberi semua fitur numerical | string, SDM categorical; vocabulary disejajarkan dengan konteks |
| Uang/missing | raw, NaN dipertahankan | raw numerik, Inf→NaN; explicit has-history |
| Context | penuh56k per outer (v28m), subsample stratified pada screening | cap50k; eksperimen penuh56k, sampling negatif:positif1/2 |
| Estimator | umumnya e4; generator seed=100×outer seed (+bag offset) | e8; uji e16; generator2026+fold, estimator_batch_size1 |
| Inferensi | legacy direct per query chunk menggambar preprocessing ulang | fit/predict cache satu draw tetap; float16, chunk8192 historis |
| Seed | outer42/7/2026; pisahkan outer seed dari generator draw | outer8247 (3 fold),31027 (5 fold); generator2026+fold |

`eda/152` dan `154` sudah menyediakan engine cache bersama. Smoke terdahulu:
direct/cache max error0; beda query chunk max0,000153. Itu bukti teknis, bukan
gain. Perubahan cache/chunk/e4→e8 adalah faktor eksperimen sendiri.

Audit CPU168 kemudian memverifikasi cache lokal large: 854.850.034 byte,
SHA256 `cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613`.
Loader lokal memakai v1.0.1 dan snapshot3c3e10bb…; deklarasi tiga fold teman
menyebut revisi/commit/SDM yang sama. Bobot aktual teman tidak disertakan,
sehingga kesamaan byte historis belum terbukti. Ini mempersempit dugaan
perbedaan checkpoint, bukan bukti kualitas atau kesamaan runtime.
[Laporan168](../../eda/out/168_kumo_checkpoint_audit.json).

## Validasi dan bias seleksi

- Utama validation_v2: outer5 fold per seed42/7/2026; inner1/7 untuk stopping dan
  seleksi; refit outer-train budget tetap; blend cross-fit. Seleksi banyak fitur
  wajib bersarang. v31 3-seed ICL hanya unggul~0,0017 dari v8, P0,88.
- Teman: holdout42k/10,5k/17,5k awal, kemudian3fold8247,10fold,5fold31027.
  Perubahan training fraction dan tuning bobot dari OOF membuat skor lintas tahap
  tidak sebanding. Laporan teman sendiri mengakui label/fold dipakai berulang.
  TabM memakai inner stopping/refit; Kumo tidak melakukan stopping bobot pretrained.
- **Jangan concatenate OOF teman**: prediksi tidak ada; split tidak sama; fitur90
  dibekukan dari riwayat seleksi berbeda. Perlu rebuild dan run di validation_v2.
- Angka public historis utama **0,56405 v31**, teman Kumo **0,56262**, Kumo+spline
  **0,56393**. Bukan bukti teman lebih baik. Petunjuk user tentang tim lain
  **0,57065 Kumo3seed** belum terikat pada resep snapshot ini.
- Rank #33/118 dan top1 0,57788 adalah keadaan **19 Oktober**, tidak diverifikasi
  sebagai keadaan terkini dalam audit ini.

## Eksperimen null yang mempersempit pencarian

| Sumber | Eksperimen | Bukti dan keputusan |
|---|---|---|
| teman | core37 | Δpooled−0,002829 vs Kumo90,0/3 fold; jangan ulang |
| teman | denoised / laten100 | −0,001874 / −0,000334; tidak terbukti |
| teman | e16 vs e8 | −0,001079; menambah estimator bukan alasan breakthrough |
| teman | balanced context, blend25% | screening+0,000492; konfirmasi5fold−0,000204; null |
| teman | context50k→56k | Δ−0,000173,3/5 fold; null |
| teman | TabICLv2 | v5−0,006423; denoised−0,003869; null |
| teman | Kumo+spline25% | Δ+0,001287,1/5 fold, public+0,00131 historis; tidak lolos gerbang saat ini |
| teman | fine-tune pilot | inner memilih step0; tidak ada pembaruan bobot |
| teman | EBM / TabPFN3.5 | keluarga tunggal lebih buruk, blend kecil tidak kuat |
| teman | koreksi Kumo offset | +0,000789,5/5 fold;95% bootstrap[−0,001286;+0,002729]; bukan breakthrough |
| teman | relational | memori gagal / tidak lengkap5fold; bukan null kualitas yang terbukti |
| utama | fine-tune / feature bagging / ~20 keluarga fitur | jurnal18–19: tidak memperbaiki keluarga pohon; jangan ulang tanpa mekanisme baru |
| utama | seleksi residual global | +0,0036 palsu→nested−0,0005; wajib nested |
| utama | ICL public transfer | Kumo-only0,56056, Causilo-only0,55333; local gain tidak menjamin public |

Rujukan lengkap: jurnal utama18–20, laporan teman
`kumo_improvement/laporan_improvement.md`, `goal054/laporan_riset.md`, dan
`goal054/README.md`. Semua tetap dipertahankan setelah pemindahan.

## Keputusan integrasi dan riset

1. **Integrasikan struktur** dahulu: scripts→root/scripts, notebooks→notebook,
   artifacts→artifacts, third_party→third_party, submission→results/teman,
   requirements→environments/teman. Pertahankan atribusi per file dan checksum.
2. **Adaptasi, jangan overwrite** fitur/pipeline utama. Kode berbeda tetap hidup
   sebagai implementasi bernama asal; root README dan indeks lintas riset menjadi
   pintu masuk tunggal. Data mentah tetap `data/`, dipakai kedua pipeline.
3. **Hipotesis H1** kategori native/cache fixed-draw memperbaiki kelemahan input.
   Uji104 kolom/context/e4 identik, satu faktor kategori; keputusan inner, outer
   screening0. Jangan membandingkan e8-teman langsung dengan e4-utama.
4. **H2** fitur V5 teman menambah informasi komplementer (harga equal-order,
   shrinkage mature repayment), bukan sekadar menambah seed. Rebuild/audit fitur,
   reference fit nested, fixed input family/context/inference. Ini belum diuji setara.
5. **H3** VIP feature/context optimization. Lanjutkan checkpoint152 yang sudah
   terkunci, bukan mengulang koalisi dari nol. Pencarian AUROC/e1 perlu confirmation
   KS/e4/context48k. Tidak memakai pseudo-label test atau fitting distribusi test.
6. Gain>0,003 pada screening yang menjanjikan→5fold×minimal3seed; pooled/fold KS,
   variasi seed, paired bootstrap baris aplikasi/customer (satu baris/customer),
   rata-rata prediksi seed **sebelum** bootstrap; unit bukan seed×baris. Butuh
   P(Δ>0)≥0,9, mayoritas fold, audit leakage dan pembanding Kumo/v31 sesuai anggaran.

Paper [VIP-COp v2](https://arxiv.org/html/2605.12904v2) mengoptimalkan context
keras secara black-box; model yang dievaluasi di paper bukan Kumo kompetisi ini.
Kode152 merupakan adaptasi, bukan reproduksi hasil paper. Bobot publik dan API
dasar dikonfirmasi dari [model card NVIDIA](https://huggingface.co/nvidia/Kumo-Tabular).
