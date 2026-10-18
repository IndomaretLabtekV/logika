# Notebook final (`notebook/build_final.py` → `notebook/final.ipynb`)

Notebook ini wajib menghasilkan file yang disubmit (10% nilai = reproducibility). Status 18 Okt: **sudah lulus run
penuh di Kaggle** dan outputnya identik dengan `results/v34/submission.csv` (public 0,56271).

## Resep yang dijalankan
| Komponen | Detail | Bobot rank |
|---|---|---|
| 6 GBDT | lgb, lgb_s, xgb, xgb_s, cat, cat_s; seed 42/7/2026; validation_v2 (stop di inner 1/7, refit ×1,15); test = rata-rata model fold | 1/18 masing-masing (total 1/3) |
| Causilo | 4 estimator, 100 fitur `fm_select`, seed 42, 5 fold; test = rata-rata 5 model fold | 1/3 |
| Kumo-Tabular large | **seluruh 70k train sebagai context**, prediksi 30k test sekali (`eda/kumo_fulltrain.py`) | 1/3 |

Blend = rata-rata berbobot dari rank/30.000 tiap kolom test, lalu divalidasi terhadap `sample_submission.csv`
(30.000 baris, ID sama dan unik, tanpa null, nilai di [0,1]) dan ditulis ke `/kaggle/working/submission.csv`.

Kenapa Kumo-70k, bukan Kumo 5 fold seperti v31: Kumo context penuh per fold di T4 ≈ 2,3 jam untuk 14k baris fold + 30k test
(≈ 8.300 s/fold di v33), jadi 5 fold ≈ 16 jam di satu GPU / ≈ 8 jam di dua GPU bersaing dengan GBDT — melewati batas
12 jam sesi bila digabung pipeline lain. Kumo-70k fit sekali = 16.172 s (4,5 jam) di satu T4, peak VRAM 6,5 GB, Spearman
0,99869 terhadap rata-rata 5 model fold. Harga yang dibayar: public v34 0,56271 vs v31 0,56405 (−0,0013, masih dalam
noise public ±0,001; lihat `07_validasi_dan_leaderboard/noise_public_17_18okt.md`).

## Struktur notebook (45 sel, layout `ref/notebook/SKILL.md`, prosa Bahasa Indonesia)
1. Banner 3 sel: judul, **`DSCxxxx - lagi mager lomba bentar`** (placeholder), nama anggota (placeholder), daftar isi.
2. Introduction: Overview, Aim, Metric (KS), Dataset, Approach (rank blend GBDT + foundation model).
3. Initialization: `!nvidia-smi`, `%pip install -q polars==2.0.0 lightgbm xgboost catboost structured-data-models`,
   `%pip install -q --no-deps causilo`, import, seed, Settings (`CFG`: path data/output/kode/run, `TREES`, `TREE_SEEDS`,
   `ICL_SEED`, `WEIGHTS`), load dataset (8 CSV dari input kompetisi).
4. Pipeline modules: `%%writefile` modul repo ke `CFG.CODE_DIR` (`features.py`, `nn_prep.py`, `valid2.py`,
   `run_valid2.py`, `kumo_fulltrain.py`) → notebook menjalankan **kode yang sama** dengan lokal.
5. Feature engineering: `build_all` → `features.parquet`, insight singkat.
6. Model training: dua proses paralel via `subprocess.Popen` dengan `CUDA_VISIBLE_DEVICES`:
   GPU0 = `kumo_fulltrain.py`; GPU1 = `run_valid2.py` 6 GBDT × 3 seed lalu Causilo seed 42. Log `gpu0.log` / `gpu1.log`,
   ringkasan `log_*.csv`.
7. Blending dan submission (validasi format), insight.
8. Conclusion.

## Hasil run Kaggle (`riantamaputra/logika-final`, push 18 Okt 07:05 WIB, selesai ~11:20 WIB)
- Runtime 15.484 s (≈ 4,3 jam) di T4×2.
- `results/final_kaggle/submission.csv`: 30.000 baris, ID unik lengkap, nilai [0,000098; 0,999985], tanpa null.
- Spearman vs `results/v34/submission.csv` = 1,000000; selisih absolut maks 0,0014 (nondeterminisme GBDT di GPU yang
  merambat lewat rank-average). Spearman vs v31 = 0,99985. Kumo-70k test vs run v30 Spearman 1,000000.

## Yang masih harus dikerjakan sebelum dikumpulkan
1. Isi banner: nomor peserta (`DSCxxxx`) dan nama anggota tim — **butuh input user**.
2. **Pin versi paket** di sel `%pip` (sekarang lightgbm/xgboost/catboost/structured-data-models tanpa versi). Versi yang
   terbukti jalan: lightgbm 4.6.0, xgboost 3.4.1, catboost 1.2.10, structured-data-models 0.2.1, causilo 1.0.3
   (`--no-deps`), polars 2.0.0. Bobot Kumo diunduh dari Hugging Face saat runtime (perlu Internet On).
3. Putuskan resep final (v34 atau v31; lihat `09_langkah_berikutnya/rekomendasi.md`). Kalau resep berubah, ubah
   `WEIGHTS`/langkah training di builder dan jalankan ulang penuh di Kaggle.
4. Nama notebook resmi `ParticipantNumber_TeamName_Notebook` (contoh `DSC0001_RandomSeed_Notebook`).
5. Jalankan ulang sekali setelah semua perubahan, bandingkan output dengan submission yang dipilih (Spearman ≈ 1).

## Cara build dan push ulang
```
cd notebook && ../.venv/bin/python build_final.py            # tulis notebook/final.ipynb (hanya compile sel)
cp final.ipynb ../results/finalk/logika-final.ipynb          # staging (notebook staging dihapus saat bersih-bersih)
KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" kaggle kernels push -p ../results/finalk
```
`results/finalk/kernel-metadata.json` sudah ada (T4, Internet on, private, sumber kompetisi).
