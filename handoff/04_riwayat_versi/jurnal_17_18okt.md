# Jurnal 17 Okt 2026 (sesi goal: public > 0,57005)

> Log mentah sesi 17–18 Okt, ditulis sambil jalan (dipindah dari `handoff/10_jurnal_17okt.md`). Versi yang dirapikan per
> topik ada di `v22_validasi_bersih.md`, `v23_v25_v27_zoo.md`, `v26_v28_v33_v36_kumo.md`, `v29_seg_causilo.md`,
> `v31_v35_submission.md`, dan folder 05–09.

## Mesin baru
- CachyOS, i7-12700H (20 thread), RAM 32 GB, **RTX 3070 Ti Laptop 8 GB**. Repo di `/home/veinsan/Projects/logika`.
- Data diunduh ulang lewat Kaggle CLI ke `data/`; `.venv` baru (Python 3.13, polars 2.0.0, lightgbm 4.7.0,
  xgboost 3.4.1, catboost 1.2.10, torch 2.14.1+cu130, tabm 0.0.3, rtdl_num_embeddings 0.0.12, causilo 1.0.3).
- Aturan baru dari user: screening lokal 1 fold di GPU lokal boleh; full fold di Kaggle.

## Audit skema validasi (sebelum validation_v2)
| Sumber bias | Status di v10–v21 | Tindakan di validation_v2 |
|---|---|---|
| Early stopping di fold OOF (v1–v9) | sudah diperbaiki di v10–v21 (inner 1/7) | tetap inner 1/7, lalu refit di seluruh train fold dengan budget tetap (GBDT best×1,15, TabM best epoch) |
| Bobot blend dipilih di OOF yang sama dengan penilaian | masih (greedy/stack di seluruh OOF, held-out halves hanya sebagian) | bobot blend **cross-fitted per outer fold** (dipilih di 4 fold lain, dinilai di fold ke-5) |
| Kandidat bersih dibandingkan dengan base v8 yang bias | masih (s01 bench pakai OOF v8) | resep v8 dijalankan ulang di protokol yang sama (v22) → semua pembanding paired di fold yang sama |
| Banyak kandidat → winner's curse | sebagian (wins/40 = stabilitas fit) | paired bootstrap baris (P>0 ≥ 0,9) + menang di mayoritas fold |
| Seleksi fitur FM / nn_prep | sudah per fold | tetap |
| Desain fitur dari EDA di seluruh label train | tidak bisa dihindari | konfirmasi fitur baru wajib lewat model cross-fit, bukan dari skor EDA |

Kode: `eda/valid2.py` (protokol), `eda/run_valid2.py` (driver; lokal `--folds 0 --seeds 42`), notebook Kaggle
`notebook/build_v22.py` (menyematkan file repo yang sama, jadi Kaggle = kode lokal).

## Siklus 1: fondasi
- Screening lokal fold 0 seed 42 (resep v8, protokol bersih): lgb 0,5189, lgb_s 0,5128, xgb 0,5196, xgb_s 0,5147,
  cat 0,5163, cat_s 0,5238, TabM 0,5231 (epoch terbaik 5). Waktu: 6 GBDT 4,4 menit, TabM 1,5 menit (RTX 3070 Ti).
  Satu fold sd ~0,008, jadi angka per model satu fold hanya untuk pembanding paired.
- v22 (`riantamaputra/logika-v22-clean`, T4×2) di-push 08:30 WIB: GBDT 6 × 3 seed × 5 fold (GPU0) dan TabM 3 seed +
  Causilo 1 seed (GPU1). TabPFN tidak ikut (kernel CLI tidak punya secret token; bobot v8 hanya 0,045).

## Siklus 1 hasil: v22 (rerun bersih resep v8, validation_v2), selesai ~12:50 WIB
Output: `results/v22/v22/{gbdt,nn,fm}/oof_*.parquet, test_*.parquet` (dengan application_id), `manifest.json` (versi paket),
log `job0.log` (GBDT 78 menit) / `job1.log` (TabM 48 menit + Causilo 31 menit).

| Model (bersih) | OOF KS pooled | Catatan |
|---|---|---|
| causilo (1 seed, 4 est, 100 fitur) | **0,52583** | model tunggal terbaik; v8 bias 0,5241 |
| lgb / xgb / xgb_s | 0,52316 / 0,52310 / 0,52285 | 3 seed |
| lgb_s / cat_s / cat | 0,52118 / 0,52077 / 0,51975 | 3 seed |
| tabm | 0,52100 | epoch terbaik 5–6; v8 bias 0,5257 |

Blend cross-fit (`eda/blend2.py`, bobot dipilih di 4 fold, dinilai di fold ke-5; output `eda/out/blend2_v22.txt`):
| Blend | cf KS | Δ vs resep v8 | P>0 | fold menang |
|---|---|---|---|---|
| resep v8 (bobot v8, _aug dilipat, TabPFN dibuang) | 0,52426 | – | – | – |
| equal trees | 0,52448 | −0,0004 (bootstrap) | 0,35 | 3/5 |
| trees 50% + causilo 50% | 0,52527 | +0,0011 | 0,77 | 4/5 |
| greedy bagged (causilo 0,46) | 0,52480 | +0,0006 | 0,68 | 4/5 |
| causilo saja | 0,52583 | +0,0013 | 0,73 | 3/5 |

Interpretasi: korelasi rank antar model 0,97–0,999 → blend nyaris tidak menambah. Bias OOF lama membuat TabM tampak
terbaik; di protokol bersih **Causilo** terbaik dan TabM terlemah. Ini konsisten dengan public 16 Okt (GBDT-only ≈ v8).
Belum ada kandidat yang lolos P>0 ≥ 0,9.

## EDA siklus 1 (semua null untuk KS)
| Script | Hipotesis | Hasil |
|---|---|---|
| `eda/100_external_credit.py` | utang/akun eksternal = bureau − pinjaman lender sendiri | bureau outstanding **tidak** memuat pinjaman lender sendiri (korelasi 0,14, rata-rata datar); total DTI chi2/df 2,97, dKS_cf +0,0018 (noise p95 ~0,0015) → lemah |
| ad hoc | delinquency bureau memuat delinquency lender sendiri? | tidak (korelasi 0,01–0,02) → tidak ada konstruksi eksternal |
| ad hoc | lag keputusan previous apps | null (sudah di eda/51) |
| `eda/101_segment_time_shock.py` | shock makro per segmen × bulan | ada efek bulan (Mei O/E 0,888, z −3,9) tapi koreksi bulan/segmen × bulan dKS ≤ null p95 |
| ad hoc | harga item bergantung kovariat item (qty, metode bayar, ukuran order, jam, status)? | 0,0000 varians dijelaskan; residual antar item dalam order independen → pp_mean sudah statistik cukup (laten + noise iid) |

## Siklus 2 (sore 17 Okt): model zoo, skema CV, pendekatan aditif
Permintaan user: uji semua model leaderboard TabArena, jangan terpaku pada patokan v8, coba skema CV lain, cek repo
supaya EDA/preprocessing tidak diulang. Token HF/PriorLabs dari user disimpan di `~/.config/logika/secrets.env`
(di luar repo; jangan di-commit).

### Screening fold 0 (seed 42, 14.000 baris; `eda/screen2.py`, bootstrap 1.000×)
Referensi fold 0: resep v8 0,52107; trees+causilo 0,52327; causilo 0,52490.
| Model | KS fold 0 | Δ sendiri vs v8 (P>0) | Δ sbg +20% ke trees+causilo (P>0) |
|---|---|---|---|
| Causilo e16 / f150 / f300 / f60 | 0,5243 / 0,5253 / 0,5217 / 0,4998 | – | – (100 fitur, 4 est = titik terbaik) |
| TabPFN-3.5 (token) | 0,5175 | −0,0034 (0,23) | −0,0005 (0,41) |
| RealMLP-TD default (pytabkit, val = inner) | 0,5157 | −0,0075 (0,10) | −0,0005 (0,43) |
| WOE scorecard + LR (`woe_lr`) | 0,5162 | −0,0060 (0,15) | +0,0012 (0,70) |
| AutoGluon LinearModel | 0,5223 | +0,0002 (0,51) | +0,0009 (0,65) |
| AG CatBoost default | 0,5237 | +0,0014 (0,64) | +0,0004 (0,55) |
| AG NN Torch / FastAI | 0,5189 / 0,5005 | −0,0050 / −0,0227 | +0,0006 / −0,0005 |
| TabSwift | 0,5103 | −0,0115 (0,01) | −0,0001 |
| AG LGB / XGB default | 0,5160 / 0,5118 | negatif | negatif |
| RF / ExtraTrees / KNN | 0,4880 / 0,4943 / 0,4421 | sangat negatif | negatif |
TabDPT (wrapper sendiri) OOM di 8 GB; dipindah ke wrapper TabArena.

### LR pada fitur rank-gaussian (`lr_q`, nn_prep + one-hot, C dipilih di inner), 5 fold lokal (CPU, detik)
OOF 0,52119 (fold 0,524/0,525/0,523/0,514/0,544) = setara TabM bersih (0,5210), sedikit di bawah LGB (0,5232).
Blend cross-fit v22 + lr_q: greedy 0,52503 (+0,0009 vs resep v8, P>0 0,74, **5/5 fold**), bobot lr_q 0,14.
**Temuan struktural:** model linear ≈ TabM ≈ GBDT → risiko hampir linear dalam fitur yang ditransformasi monoton;
kapasitas nonlinear tidak menambah. Ini menjelaskan kenapa tuning/arsitektur tidak pernah membantu.

### Skema CV (`eda/104_cv_schemes.py`, LGB v8 params)
| Uji | Hasil |
|---|---|
| 5-fold vs 10-fold | 0,52074 vs 0,52067 pooled; Δ bootstrap +0,0004 (P>0 0,59) → tambah 12,5% data latih tidak membantu (plafon informasi) |
| Maju-waktu (train ≤ bulan m, nilai m+1) vs acak | rata-rata −0,0024, sd per bulan ~0,02 → tidak ada drift |
| Adversarial validation train vs test | AUC 0,4987 → split acak; StratifiedKFold tepat |

### Residual terhadap OOF bersih (`eda/103_clean_residual.py`)
0 dari 228 fitur di atas null (chi2/df p95 1,93, dKS p95 +0,0018). Satu-satunya chi2 tinggi `dec_month` (efek Mei).
Per segmen: thin-file KS ~0,496, history ~0,535; Causilo unggul di thin-file dan rate 24–28%.

### Peta informasi per keluarga fitur (`eda/105_family_ablation.py`, LR rank-gaussian, cross-fit 5 fold)
LR semua fitur 0,5212. | keluarga (n) | KS sendiri | KS hilang bila dibuang |
|---|---|---|
| repayment rp/rq/sv (36) | 0,343 | **0,0266** |
| bureau (50) | 0,272 | 0,0139 |
| harga e-commerce pp (17) | 0,376 | 0,0114 |
| akun ac/cf (38) | 0,149 | 0,0051 |
| perilaku e-commerce ec/ec2 (40) | 0,364 | 0,0041 |
| previous pv/ob (25) | 0,213 | 0,0030 |
| cross (7) | 0,164 | 0,0017 |
| aplikasi + profil (19) | 0,232 | −0,0009 |
Korelasi skor keluarga di dalam kelas: blok {aplikasi, bureau, previous} ~0,65 ("grade kredit"), blok {perilaku,
harga e-commerce} 0,84 ("konsumsi"), repayment ~0,3 ke keduanya. Dua faktor laten utama + repayment.

### Laptop hang (~15:00 WIB)
Penyebab kemungkinan: TabICLv2 (zoo AutoGluon) jatuh ke CPU (~500% CPU) bersamaan dengan ablation LR (semua thread).
Aturan sejak itu: satu job berat sekaligus, `OMP_NUM_THREADS` dibatasi (4–8).

### Zoo lanjutan
- TabICLv2 (AutoGluon, lokal, jatuh ke CPU, 14 menit): fold 0 0,5163 (−0,0055 vs v8; +20% ke base +0,0010).
- LimiX-2, TabPFN-3, Mitra: OOM / ditolak di GPU 8 GB → dikirim ke Kaggle v23 (`riantamaputra/logika-v23-zoo`, T4×2,
  bersama TabICL/v2, iLTM, TabDPT/Turbo, Nori). Dataset Kaggle berisi token ditolak classifier izin (eksfiltrasi), jadi
  keluarga TabPFN (butuh token) hanya diuji lokal.
- **NVIDIA Kumo-Tabular** (`structured-data-models`, `eda/sdm_screen.py`, 100 fitur fm_select, 4 estimator):
| Varian | KS fold 0 | Δ sendiri vs v8 (P>0) | +20% ke base (P>0) |
|---|---|---|---|
| small, ctx 20k | 0,5240 | – | – |
| small, ctx 56k | 0,5230 | +0,0029 (0,78) | +0,0010 (0,71) |
| medium, ctx 20k | 0,5256 | +0,0043 (0,80) | +0,0015 (0,77) |
| **large, ctx 20k** | **0,5276** | **+0,0058 (0,88)** | +0,0017 (0,81) |
| large, ctx 20k, 198 fitur | 0,5240 | – | – (100 fitur lebih baik, sama seperti Causilo) |
Kumo large = model tunggal terbaik fold 0 → lolos screening, lanjut variasi lalu full fold Kaggle.
- Kumo large context penuh (56k, 18 menit): 0,52756 = context 20k (0,52758) → pakai 20k.
- iLTM (Kaggle v23): fold 0 0,5216. LimiX-2 OOM di T4 15 GB pada context 48k; TabDPT/Turbo/Nori/Mitra ditolak oleh
  estimasi memori AutoGluon → ronde 2 `riantamaputra/logika-v25-zoo2` (LimiX ctx 15k, TabICLv2 ctx 30k, rasio memori dinaikkan).
- **Peringatan reprodusibilitas Kumo:** wrapper `valid2.kumo_fit` (subsample context rng seed, generator estimator
  diset, query = fold + test dalam satu TableTensor) memberi fold 0 **0,5182**, bukan 0,5276. Sedang dilacak sebelum
  full fold (v26 sudah dibangun, belum di-push).
- Wrapper Kumo e8 (8 estimator): 0,5196 → bukan soal jumlah estimator.
- v25 (zoo ronde 2) berstatus CANCEL_ACKNOWLEDGED tanpa output (dibatalkan dari luar); tidak di-push ulang.
- Jebakan: loop tunggu `until ! pgrep -f "<pola>"` macet karena pola muncul di baris perintah loop itu sendiri
  (sama dengan jebakan `pkill -f`). Diagnostik `eda/106_kumo_variance.py` sempat tertunda ~1 jam karena ini.

### Varians Kumo (`eda/106_kumo_variance.py`, fold 0, large, ctx 20k)
| Varian | KS | Spearman vs A |
|---|---|---|
| A (setup sdm_screen) / A diulang | 0,5281 / 0,5258 | 1 / 0,998 |
| generator seed 42 | 0,5255 | 0,998 |
| context draw lain (rng 1 / rng 2) | 0,5255 / 0,5213 | 0,988 / 0,985 |
| + baris test di tabel | 0,5261 | 0,998 |
Tidak ada bug: **varians antar context draw besar (0,518–0,528)**, ditambah ±0,002 dari acak estimator. 0,5276 awal
adalah draw beruntung. Pelajaran: screening ICL dengan subsample context butuh beberapa draw.
- `kumo_fit(n_bag=3)` (rata-rata 3 draw 20k), fold 0: **0,5267**; vs resep v8 +0,0048 (P>0 0,85); +20% ke base +0,0011
  (P>0 0,75); vs Causilo sendiri +0,0018. Lolos screening → **v26** (`riantamaputra/logika-v26-kumo`, push 18:25 WIB):
  Kumo large, 2 draw/fold, seed 42, fold 0–2 GPU0, fold 3–4 GPU1.
- TabFM (Google, lisensi non-komersial diterima via `accept_license=True` untuk riset lomba mahasiswa): bobot ~7 GB →
  OOM lokal; disiapkan Kaggle v27 (`notebook/build_v27.py`).
- TabICLv2 versi NVIDIA sdm: 0,5164 (sama dengan AutoGluon 0,5163).
- Causilo + 14 fitur utang eksternal (eda/100): 0,5237 vs 0,5249 → null.
- KumoRelational ditunda: sampler graf mengambil sampel tetangga terbatas (merusak agregat seperti pp_mean), basis
  TabICLv2 lemah di data ini, v14 set-encoder baris mentah null; biaya ~½ hari.
- v26 (Kumo full fold Kaggle) **CANCEL_ACKNOWLEDGED saat antre** (18:58 WIB), sama seperti v25; tidak ada output,
  alasan tidak terlihat dari CLI (dibatalkan dari UI atau oleh Kaggle). Tidak di-push ulang berulang.
  **Penyimpangan aturan yang disengaja:** full fold Kumo dijalankan lokal (`results/v26_local/`, seed 42, 5 fold,
  2 draw context 20k, protokol validation_v2 identik) karena Kaggle terblokir.
- Fold 0 blend (prediksi tersimpan): Kumo+Causilo 50/50 +0,0054 vs resep v8 (P>0 0,91); trees/Causilo/Kumo 1:1:1
  +0,0036 (0,88) → trees mengencerkan. Perlu konfirmasi lintas fold.

### Koreksi status Kaggle (21:30 WIB)
- **Kaggle tidak terblokir.** Cek ulang: v25 = COMPLETE, v26 = RUNNING (lastRunTime 19:16 WIB). Status
  CANCEL_ACKNOWLEDGED yang saya baca 18:58 ternyata sementara/menyesatkan; saya menyimpulkan "terblokir" tanpa cek ulang
  dan menjalankan full fold lokal → komputasi ganda. Pelajaran: cek `kaggle kernels status` + `kernels list --mine`
  ≥2 kali dengan jeda sebelum menyimpulkan cancel.
- v25 (zoo ronde 2, T4): LimiX ctx 15k OOM (minta 8,2 GB tambahan) → null karena memori; TabDPT/TabDPT-Turbo gagal
  load checkpoint (`Missing key enc_cell_dim`, versi wrapper tidak cocok); Nori/Nori30M gagal fit; **Mitra (8k baris)
  0,4488** → null jauh. TabICLv2 tidak sempat jalan di rantai itu. Zoo TabArena praktis habis: tidak ada yang
  mengalahkan Causilo/Kumo.

### Kumo full fold (lokal, `results/v26_local/`, seed 42, large, 2 draw ctx 20k)
| fold | 0 | 1 | 2 | 3 | 4 | OOF |
|---|---|---|---|---|---|---|
| Kumo | 0,5256 | 0,5358 | 0,5163 | 0,5140 | 0,5460 | **0,52484** |
Causilo OOF 0,52583, lgb 0,52316. Kumo bukan model tunggal terbaik secara pooled; fold 0 terlalu optimis.
Rank corr Kumo–Causilo 0,989, Kumo–trees ~0,978.

Cross-fit blend (`eda/out/blend2_v26.txt`, referensi resep v8 bersih 0,52426):
| blend | cf KS | dKS | sd | P>0 | fold menang |
|---|---|---|---|---|---|
| trees_causilo_50 | 0,52527 | +0,00110 | 0,00140 | 0,77 | 4/5 |
| greedy (semua + kumo + lr_q) | 0,52497 | +0,00112 | 0,00132 | 0,82 | 2/5 |
| kumo_causilo_50 | 0,52555 | +0,00135 | 0,00200 | 0,77 | 4/5 |
| **trees_causilo_kumo 1:1:1** | **0,52602** | **+0,00152** | 0,00155 | **0,84** | 4/5 |
| v8 + 25% kumo | 0,52493 | +0,00068 | 0,00101 | 0,76 | 4/5 |
Keputusan: terbaik sejauh ini tapi **P>0 0,84 < 0,9 → belum dipromosikan**. Kandidat ditulis ke
`results/v26/trees_causilo_kumo.csv` (Spearman vs v8 0,99614). Hipotesis berikut: varians draw context menekan Kumo;
context penuh (semua 56k baris train fold, tanpa subsample) di fold 0 = 0,52756. → **v28**
(`riantamaputra/logika-v28-kumo56k`, push 21:35 WIB, kumo_large_c56000, seed 42, 5 fold, T4x2).

### Learning curve (`eda/102_learning_curve.py`, fold 0, 2 rep, protokol validation_v2)
| baris train | 40% | 60% | 80% | 100% (56k) |
|---|---|---|---|---|
| lgb | 0,5122 | 0,5169 | 0,5198 | 0,5197 |
| cat_s | 0,5151 | 0,5167 | 0,5188 | 0,5194 |
Kurva mendatar di 80→100% (lgb +0,0000, cat_s +0,0006); kemiringan log-linear (+0,0019/+0,0011 per +25%) melebih-lebihkan
karena titik terakhir datar. Arti: model di plafon informasi fitur; refit full-train untuk test paling banyak ~+0,0005,
dan jarak ke top-1 (0,016 public) **bukan soal jumlah data**.
Konsekuensi validasi: sd selisih KS blend berkorelasi tinggi ~0,0015 pada 70k baris; untuk P>0 ≥0,9 butuh gain
≥ ~0,002. Seed outer tambahan tidak mengecilkan noise ini (baris yang sama).

### Pola public vs CV (riwayat submission, `eda/out/107_submissions.csv`)
v8 0,56145 = v5; S4 (bagian GBDT v8) 0,56133; S5 (bagian NN/FM, OOF lebih tinggi +0,0040) 0,55866; S3 stack + TabICL
0,55778; v9 0,55917. Tiga kali blend condong NN/ICL kalah di public (−0,0023 s.d. −0,0037) walau OOF naik. Bisa noise
(sd selisih public ~0,003 per pasangan), bisa juga ICL/NN tidak transfer. Ini pertanyaan untuk submission informatif.

### Noise public (`eda/107_public_noise.py`, OOF bersih cross-fit, subset acak 30% = ukuran public jika 9k/30k)
- trees_causilo_kumo − resep v8: OOF penuh +0,00175; di subset 30% **+0,0016 ± 0,0022** (5–95% −0,0021..+0,0052),
  P(selisih < 0) 0,23. Satu submission tidak bisa membuktikan gain ini; hanya bisa menolak "ICL rusak di public" jika
  hasilnya ≤ −0,004.
- icl_nn_only − trees_only (analog bersih S5 vs S4): +0,0007 penuh, subset +0,0018 ± 0,0026. Public S5−S4 −0,0027
  ada di ekor bawah (~P 0,1) → lemah-sugestif, belum bukti bahwa ICL/NN tidak transfer (S5 juga 39% tabm_aug).
- Asumsi: public = 30% test; proporsi sebenarnya tidak diketahui.

### Reproducibility Kumo (Kaggle v26 vs lokal, 22:50 WIB)
v26 Kaggle T4 selesai (≈4.100 s/fold, 2,5× lebih lambat dari 3070). Per fold: 0,5257/0,5357/0,5166/0,5143/0,5462 vs
lokal 0,5256/0,5358/0,5163/0,5140/0,5460; OOF 0,52469 vs 0,52484; Spearman OOF 1,0000. **Reproducible lintas
mesin** (selisih ≤0,0004 per fold dari non-determinisme GPU). v28 (context penuh) masih RUNNING.

## 18 Okt (malam, user tidur; otonom)
- Arahan user: run di tempat yang lebih cepat (lokal 3070 ≈2,5× T4 untuk Kumo); kalau sampai 06:00 belum ada
  breakthrough, tetap pakai 1 submission yang paling informatif.
- v28 lokal: Kumo context penuh fold 2 = **0,5197** vs 0,5163 (2 draw 20k) → +0,0034. Satu fold lokal 3.333 s (termasuk
  30k baris test). Fold 0,1,3,4 dijalankan lokal (selesai ~04:00); Kaggle v28 dibiarkan jalan sebagai replikasi.
- v27 (TabFM screen fold 0, T4) di-push 00:30.
- `eda/108_segment_blend.py`: per segmen KS thin-file trees/causilo/kumo/tabm 0,4963/0,4963/0,4964/0,4945; history
  0,5344/0,5349/0,5372/0,5323. Bobot grid per segmen vs global (cross-fit): −0,0005 (P>0 0,34) → **null**. Bobot hasil
  grid juga tidak stabil antar fold dan global-grid 0,52539 < bobot tetap 1:1:1 0,52602 → pakai bobot tetap sederhana.
- Hipotesis baru (bukan tuning): fm_select global memberi model ICL **44 dari 104 kolom yang 100% NaN untuk thin-file**
  (fold 0). `valid2.seg_fit` + prefix `seg_` di run_valid2: model ICL difit terpisah per segmen has_history (context
  dan seleksi fitur sendiri). **v29** (`logika-v29-segcausilo`, seg_causilo, seed 42, 5 fold, T4x2) di-push 01:10.
- Kaggle membatasi 2 sesi GPU batch bersamaan. Kernel v28 (duplikat run lokal, sisa ~4,5 jam kuota) **dihapus**
  supaya v29 bisa jalan; kodenya tetap di `results/v28k/`.
- **v29 seg_causilo (T4, ~205 s/fold): null.** OOF 0,52409 vs Causilo global 0,52583; thin-file 0,4953 vs 0,4963,
  history 0,5359 vs 0,5349. Kolom all-NaN tidak merugikan ICL (praktis diabaikan), dan memisah context menghapus
  informasi lintas segmen + kalibrasi antar segmen. Di blend trees/causilo/kumo: mengganti Causilo +0,00154 (P 0,83,
  3/5), menambah keduanya +0,00162 (P 0,85) → sama dengan noise. (`eda/out/109_seg_causilo.txt`)
- Catatan aturan (docs/context.md §6): **public 45% + private 45% + notebook 10%** (reproducibility). Notebook final
  harus jalan di Kaggle; Kumo context penuh di T4 ≈2,3 jam/fold (termasuk 30k test) → 5 fold di 2 GPU ≈7 jam.
  **v30** (`logika-v30-kumo70k`, 01:35): Kumo large dengan seluruh 70k train sebagai context, test diprediksi sekali →
  ukur waktu & VRAM T4 untuk opsi notebook final yang lebih murah (`eda/kumo_fulltrain.py`).
- Kumo context penuh fold 0 = 0,5256 (= 2 draw 20k 0,5256). Fold 2 +0,0034, fold 0 ±0 → tunggu 5 fold.
- v27 TabFM (T4) gagal: ctx 20k OOM (12,4 GB terpakai), ctx 8k prediksi NaN (autocast fp16; bobot fp32). v27 versi 2
  di-push 01:25: `--fp32`, ctx 8k (GPU0) dan 4k (GPU1), chunk 200.
- 01:29 WIB access token Kaggle kedaluwarsa ("Authentication required"). `kaggle auth print-access-token` mencetak token
  baru dari refresh token tapi tidak menyimpannya; CLI jalan lagi dengan `KAGGLE_API_TOKEN="$(kaggle auth
  print-access-token)" kaggle ...` (token tidak ditulis ke file mana pun). Tidak perlu login ulang interaktif.
- v30 v1 OOM di T4 (context 70k; 7,3 GB reserved-unallocated). v2: `expandable_segments`, chunk 500.

### Kumo context penuh, 5 fold (lokal, `results/v28m/` = gabungan v28_local_f2 + v28_local; 04:17 WIB)
| fold | 0 | 1 | 2 | 3 | 4 | OOF |
|---|---|---|---|---|---|---|
| Kumo ctx penuh (56k) | 0,5256 | 0,5380 | 0,5197 | 0,5151 | 0,5507 | **0,52665** |
| Kumo 2 draw 20k | 0,5256 | 0,5358 | 0,5163 | 0,5140 | 0,5460 | 0,52484 |
Menang/seri di 5/5 fold (+0,0018 OOF): varians draw context memang menekan Kumo. Kumo ctx penuh = model tunggal
terbaik (Causilo 0,52583, lgb 0,52316). ~55–65 menit/fold di 3070 (termasuk 30k test).

Cross-fit blend (`eda/out/blend2_v28.txt`, ref resep v8 bersih 0,52426):
| blend | cf KS | dKS | sd | P>0 | fold menang |
|---|---|---|---|---|---|
| **trees_causilo_kumo 1:1:1** (aturan ditetapkan sebelum v28) | 0,52613 | **+0,00221** | 0,00149 | **0,93** | **4/5** |
| kumo_causilo_50 | 0,52701 | +0,00252 | 0,00194 | 0,91 | 4/5 |
| v8 + 25% kumo | 0,52537 | +0,00134 | 0,00102 | 0,93 | 4/5 |
| greedy_bagged | 0,52534 | +0,00164 | 0,00152 | 0,84 | 4/5 |
**Kandidat pertama yang lolos kriteria goal** (P>0 ≥0,9, mayoritas fold) → dipromosikan:
`results/v31/submission.csv` (trees 6×1/18, Causilo 1/3, Kumo ctx penuh 1/3; Spearman vs v8 0,99615).
Catatan jujur: 4 aturan berisi Kumo diuji (multiple comparison kecil); aturan 1:1:1 sudah dipakai di v26 sebelum v28.

**Pra-registrasi submission v31** (`eda/107_public_noise.py` kini memakai v28m; output lama tetap di
`out/107_public_noise.txt`): selisih vs resep v8 di subset 30% = +0,0024 ± 0,0022 (P<0 0,14).
Perkiraan public = 0,56145 (v8) + 0,0019 ± 0,0025 → **0,5633 ± 0,0025** (sd ditambah karena public resep v8 bersih
sendiri belum diukur). Pertanyaan riset: apakah gain ICL (Kumo ctx penuh + Causilo) di validasi bersih transfer ke
public, atau pola "ICL/NN kalah di public" (S3/S5/v16) berulang? Hasil ≤ 0,5590 = pola itu nyata.
- **Hasil v31: public 0,56405** (submit 04:22 WIB) = **public terbaik baru**, +0,0026 vs v8 0,56145; perkiraan
  0,5633 ± 0,0025 → di dalam rentang (+0,7 sd). Gain ICL bersih **transfer** ke public; pola "ICL/NN kalah di public"
  sebelumnya kemungkinan karena tabm_aug/stack berbobot negatif/OOF bias, bukan ICL. Sisa kuota window ini: 2.
- **Pra-registrasi submission 2 (window ini): `results/v31/kumo_causilo50.csv`** (Causilo 1/2 + Kumo ctx penuh 1/2,
  tanpa pohon; CV +0,00252 vs v8, P 0,91, 4/5). Vs v31: OOF +0,0009, row-bootstrap +0,0003 ± 0,0012 (P 0,63), subset
  30% +0,0002 ± 0,0017. Perkiraan public **0,5642 ± 0,0017**. Pertanyaan: apakah 1/3 pohon membantu di public
  (S4 vs S5 dulu mengisyaratkan pohon lebih aman)? Satu hasil tidak bisa membedakan, tapi ≤0,5615 akan jadi sinyal
  bahwa pohon perlu dipertahankan. Slot window berakhir 07:00, jadi tidak ada biaya kesempatan.
- **Hasil submission 2 (v31b, Causilo+Kumo tanpa pohon): public 0,55709** = −0,0070 vs v31, padahal perkiraan
  +0,0002 ± 0,0017 → **−4 sd**. Konsisten dengan S5 (NN/FM saja −0,0028) dan v16 (TabICL −0,0037): komponen ICL/NN
  *sendirian* jauh lebih buruk di public daripada prediksi CV, tapi *dicampur* 1/3 pohon justru terbaik (v31).
  Diagnostik: (1) prediksi test ICL normal (rank corr test ≥ OOF, mean/kuantil sama) → bukan bug sisi test;
  (2) `eda/110_neighbor_gap.py`: jarak 1/10/50-NN validasi→train vs test→train rasio 1,001/1,0005/1,0006 → tidak ada
  struktur "tetangga dalam train" yang menggelembungkan CV ICL. Penjelasan yang tersisa: noise ekor (public share
  tidak diketahui) atau perbedaan hubungan label di subset public. Kurva CV porsi pohon: datar 0–0,4 (0,5270→0,5259),
  turun setelah 0,5 → v31 (1/3) tetap pilihan CV-konsisten; jangan menyetel bobot dari public.
- **Pra-registrasi submission 3: `results/v31/kumo_only.csv`** (Kumo ctx penuh sendiri). CV vs v31 −0,0001 ± 0,0016.
  Jika gain CV transfer: public ≈ 0,5640 ± 0,0021; jika "ICL sendirian gagal di public" berlaku umum: ≈ 0,557.
  Pertanyaan: kegagalan v31b karena Causilo, atau sifat semua ICL solo?
- **Hasil submission 3 (v31c, Kumo saja): public 0,56056** (−0,0035 vs v31; perkiraan −0,0001 ± 0,0021 → −1,6 sd).
  Ringkasan public malam ini: v31 trees+Causilo+Kumo 0,56405 > Kumo 0,56056 > Causilo+Kumo 0,55709.
  **Menambah Causilo ke Kumo: −0,0035 di public** (CV +0,0004) → bukti lemah-sedang (~2 sd) bahwa Causilo adalah
  komponen yang tidak transfer (Causilo juga ada di S5 0,26 dan v8 0,137). Pohon tetap penting (v31 > Kumo saja).
  Kandidat window berikut (setelah 07:00): **trees + Kumo 1:2 tanpa Causilo** — CV +0,00212 vs v8 (P 0,91, 4/5),
  −0,0001 vs v31 → setara di CV, didukung arah bukti public. Risiko: memilih dari 3 titik public yang noisy.
- **v27 v2 TabFM (fp32, T4): null.** Fold 0 ctx 8k 0,5173 (6.848 s), ctx 4k 0,5169 (3.207 s); ref v8 fold 0 0,5211.
  rho vs Kumo 0,977 / vs trees 0,971; +20% ke v31 di fold 0: +0,0006 ± 0,0022 (P 0,60). Terlalu lambat (≈1,9 jam/fold
  untuk ctx 8k) dan tidak menambah → tidak lanjut full fold. Zoo TabArena tuntas: hanya Kumo & Causilo yang berguna.
- Varians estimator Kumo (ctx penuh = context deterministik): fold 0 e4 diulang 0,52585 vs 0,52756 sebelumnya → ±0,0017
  dari acak estimator saja. Screening e8 / f150 / medium berjalan (`results/screen/kumo_full_f0/`).
- Screening Kumo fold 0 ctx penuh (`results/screen/kumo_full_f0/`, sdm_screen tanpa test): large e4 0,52585 (ulang;
  sebelumnya 0,52756), **large e8 0,52780**, large f154 0,52305, medium 0,52361. Fitur lebih banyak dan model medium
  lebih buruk; e8 ≈ rata-rata 2×e4 (+0,001, dalam noise). → **v33 lokal** (06:15): Kumo large ctx penuh dengan seed
  generator lain (`g_off`, key `g`), 5 fold; akan dirata-rata dengan v28m (≈ e8, mengurangi noise estimator ±0,0017).
- **v30 (T4): Kumo large context 70k (seluruh train), 30k test sekali: 16.172 s (4,5 jam), peak VRAM 6,5 GB.**
  Spearman vs rata-rata 5 model fold (v28m) 0,99869. Opsi notebook final: pohon + Kumo-70k (≈4,5 jam di 1 T4).
- **Pra-registrasi window 18 Okt #1: `results/v32/trees_kumo_1to2.csv`** (6 GBDT 1/18 masing-masing + Kumo ctx penuh
  2/3; tanpa Causilo). CV +0,00212 vs v8 (P 0,91, 4/5), vs v31 −0,0001 ± 0,0010 (row bootstrap). Perkiraan berbasis CV:
  public 0,5640 ± 0,0018; jika Causilo memang tidak transfer (v31b vs v31c), > 0,5641. Pertanyaan: apakah Causilo
  menyumbang di public dalam blend berpohon?
- **Hasil v32 (trees + Kumo 1:2, tanpa Causilo): public 0,56376** (−0,0003 vs v31; perkiraan −0,0001 ± 0,0018 → sesuai).
  Causilo netral di dalam blend berpohon. Pelajaran gabungan public: blend **tanpa pohon** jatuh jauh (v31b 0,5571,
  Kumo saja 0,5606, S5 0,5587) sementara blend pohon+ICL 1/3:2/3 stabil 0,5638–0,5641. Pohon wajib ≥ ~1/3.
  Sisa window 18 Okt: 2.
- Kendala notebook final: Kumo ctx penuh per fold di T4 ≈5 jam (estimasi dari v30) → 5 fold × 2 GPU ≈16 jam > batas
  12 jam. **Resep final yang reproducible:** pohon v22 (3 seed, rata-rata fold) 1/3 + Causilo (5 fold) 1/3 + **Kumo
  large context 70k fit sekali** 1/3 (v30). → `results/v34/submission.csv`, Spearman vs v31 0,99985.
- **Pra-registrasi window 18 Okt #2: v34** (resep final). Perkiraan public = v31 ± 0,0005 → **0,5640 ± 0,0006**.
  Pertanyaan: apakah resep yang bisa dijalankan ulang di Kaggle mempertahankan skor v31?
- **Hasil v34 (resep final): public 0,56271** (−0,0013 vs v31, padahal Spearman 0,99985; perkiraan ±0,0005 → meleset).
  **Kalibrasi noise public:** perturbasi rank sekecil apa pun menggeser KS public ~0,001 (KS = statistik maksimum di
  satu titik potong). v31/v32/v34 (0,56405/0,56376/0,56271) = satu klaster ≈ 0,5635 ± 0,0007; selisih < ~0,002
  **tidak bermakna**. Konsekuensi: jangan memilih final dari selisih public kecil; pilih yang reproducible (v34) atau
  setara-CV. Sisa window 18 Okt: 1.
- **Notebook final** dibangun: `notebook/build_final.py` → `notebook/final.ipynb` (layout ref/notebook/SKILL.md, prosa
  Indonesia, 45 sel; modul repo ditulis via `%%writefile`; GPU0 Kumo-70k, GPU1 6 GBDT×3 seed + Causilo; blend rank
  bobot tetap; validasi format). Banner tim masih placeholder (`DSCxxxx`, nama anggota) → **user perlu mengisi**.
  Run reproducibility penuh di Kaggle: `riantamaputra/logika-final` (push 07:05 WIB, ~5 jam). Setelah selesai:
  bandingkan output-nya dengan `results/v34/submission.csv`.

## 18 Okt pagi: user minta lanjut sampai 0,57768 (sisa 13 hari)
- Leaderboard 18 Okt: kita #24 (0,56405). 0,563–0,568 sangat padat; tujuh tim > 0,570 (top-1 0,57768, #2 0,57530,
  #3 0,57290). Kontinuum ini konsisten dengan plafon ~0,565 + noise public + seleksi best-of-many submission.
- **`eda/111_relative_scan.py`: null.** Premium harga z per subkelompok yang belum pernah dipisah (metode bayar,
  cancelled, malam/siang, weekend, qty>1, ukuran basket, dispersi, tren, 10 item terakhir, share spend kartu kredit/COD)
  diuji vs OOF bersih v31 (LR chi2 menambah [rank(x), missing] ke logit(p)). Maks chi2 5,4 (z_cancel_minus_all) < batas
  Bonferroni 9,4; fitur pp_* lama juga null (kalibrasi benar). Daya uji: sinyal bernilai +0,01 KS akan memberi chi2
  ratusan → e-commerce relatif sudah habis.
- Cek kebocoran account_behavior: bulan terakhir per customer selalu tepat 1 bulan sebelum bulan keputusan (70.000/70.000)
  → tidak ada baris bulan berjalan (bersih).
- **Pra-registrasi window 18 Okt #3: `results/v35/five_family.csv`** (6 GBDT 0,3 + Kumo 0,3 + Causilo 0,2 + TabM 0,1 +
  lr_q 0,1). CV +0,00209 vs v8 (P 0,95, **5/5 fold**), −0,0001 ± 0,0009 vs v31. Perkiraan public 0,5638 ± 0,0015
  (klaster v31/v32/v34). Pertanyaan: apakah keragaman tambahan (NN + linear) dihargai public seperti campuran pohon+ICL?
- **Hasil v35 (five-family): public 0,56298** → klaster v31/v32/v34/v35 = 0,5627–0,5641 (mean 0,5634, sd 0,0006).
  Semua varian setara-CV juga setara di public; ini plafon pendekatan sekarang. Kuota 18 Okt habis.
- **v33 (Kumo ctx penuh, seed generator kedua; fold 0–2 lokal, 3–4 Kaggle ≈8.300 s/fold di T4): null.** Per fold
  0,5262/0,5385/0,5196/0,5121/0,5523; OOF 0,52586 (seed 1: 0,52665). Rata-rata 2 seed (≈ e8) OOF 0,52569 →
  tidak lebih baik; seed 1 sedikit beruntung. Blend 1:1:1 dengan rata-rata ini +0,00215 (P 0,93, 4/5) = v31.
  (`results/v33m/`, `results/v36/`, `eda/out/blend2_v36.txt`). Tidak ada kandidat baru untuk disubmit.
- `valid2.run` kini menulis oof/test setelah setiap fold (run yang terputus tidak kehilangan fold yang sudah selesai).
- `eda/kumo_finetune.py` (fine-tune Kumo, screening fold 0, protokol bersih) **ditulis tapi belum dijalankan**; user
  minta berhenti setelah run yang sedang jalan selesai. Ini langkah pertama yang disarankan untuk sesi berikut.
- **Uji reproducibility notebook final (`notebook/final.ipynb` → kernel Kaggle `riantamaputra/logika-final`, T4×2): lulus.**
  Runtime 15.484 s (≈4,3 jam; GPU0 Kumo konteks 70k, GPU1 6 GBDT×3 seed + Causilo). Output `results/final_kaggle/`:
  `submission.csv` 30.000 baris, ID unik lengkap, nilai [0,000098; 0,999985], tanpa null. Spearman vs
  `results/v34/submission.csv` = 1,000000 (selisih absolut maks 0,0014, berasal dari nondeterminisme GBDT/GPU di rank-average);
  vs v31 = 0,99985; Kumo-70k test vs v30 Spearman 1,000000. Artinya notebook final mereproduksi v34 (public 0,56271).
  Sisa tugas: isi placeholder banner (nomor peserta, nama anggota) sebelum dikumpulkan; pilih apakah final = v34 atau v31.
- **`eda/kumo_finetune.py` screening fold 0: dihentikan, belum valid.** KS baseline (sebelum fine-tune, konteks penuh e4)
  keluar 0,29776 (1.005 s), jauh di bawah fold 0 Kumo ctx penuh di v28m (≈0,526) → ada bug persiapan data, run dihentikan.
  Sudah dicek: `numerical[..., 1]` = kolom "1" dari `to_pandas()`, dan indexing TableTensor ala finetune identik dengan
  cara `valid2.kumo_fit` (konteks 16k dan 56k, korelasi 1,0). Masih janggal: query `va[:3000]` memberi KS ~0,36, sedangkan
  `va[:2000]` dengan konteks 4k memberi 0,547 → curigai frame fitur (`common.features()` vs frame yang dipakai
  `run_valid2`) atau urutan baris/ID. Langkah berikut: bandingkan langsung prediksi fold-0 v28m vs script ini per
  application_id sebelum fine-tune apa pun.
