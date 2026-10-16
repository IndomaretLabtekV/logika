# Langkah berikutnya (per 16 Okt 2026 malam; deadline notebook 30 Okt 23:59 WIB)

## Inti
Setelah 21 versi, satu hal yang jelas: **sumber informasi (fitur) sudah jenuh**, tapi **cara validasi kita bias**.
Semua OOF v1–v9 optimis karena early stopping di fold validasi, terutama TabM. Akibatnya bobot blend dan semua
keputusan "lebih bagus dari v8" salah arah di public (lihat `07_validasi_dan_leaderboard/bias_oof_early_stopping.md`).

Jangan berharap lompatan ke 0,5777 dari sini. Realistisnya: memperbaiki validasi bisa memberi blend yang lebih jujur dan
sedikit lebih baik/aman di private (dan mungkin public +0,001–0,003), bukan +0,016. Tetap jujur soal ini ke user.

## Prioritas 1: rerun pipeline dengan validasi bersih (butuh GPU, tunggu kuota reset)
Satu notebook Kaggle T4×2 (~4–5 jam) berbasis `notebook/build_notebook.py` (v9) dengan perubahan:
1. **Early stopping di inner split.** Untuk tiap outer fold: pisahkan 1/7 train fold sebagai stop split
   (`splits(y[tr], 7, 300 + fold)` seperti v15–v21). LGB/XGB/CatBoost stop di situ. Opsi lebih murni: pilih jumlah
   ronde di inner split lalu refit di seluruh train fold dengan ronde tetap (×1,1).
2. **TabM:** pilih epoch di inner 80/20 lalu refit epoch tetap di seluruh train fold (protokol v11, kode di
   `notebook/v11_experiment.py`). TabM overfit sangat dini (epoch terbaik ~6), jadi refit epoch tetap penting.
3. TabPFN, Causilo tetap (tanpa early stopping, sudah bersih). TabPFN butuh secret token di UI Kaggle.
4. Model: lgb, lgb_s, xgb, xgb_s, cat, cat_s (3 seed), TabM (3 seed), TabPFN, Causilo. Varian `_aug` boleh ikut kalau
   mau mereproduksi v8 (bobot v8 54,8% di model _aug), tapi efeknya ≈ 0 dan protokolnya transduktif; lebih bersih tanpa.
5. Simpan OOF + test per model dengan ID, per seed, plus manifest versi paket.

Lalu di lokal:
- Ganti base `eda/s01_stack_bench.py` ke OOF bersih.
- Bandingkan blend **rank-average bobot positif** (equal trees, trees + x% TabM, greedy dengan bagging) di held-out
  halves dan bootstrap. Hindari stack LR dengan bobot negatif.
- Cek apakah porsi GBDT optimal di OOF bersih lebih tinggi dari v8 (0,467). Data public 16 Okt mengarah ke sana:
  α_GBDT 0 → 0,55866, 0,467 → 0,56145, 1 → 0,56133.
- Submit kandidat hanya kalau OOF bersih-nya ≥ v8 yang dihitung ulang dengan protokol sama, dan bobotnya wajar.

## Prioritas 2: notebook final yang reproducible (wajib, 10% nilai)
- Notebook yang dikumpulkan harus menghasilkan **file submission yang dipilih**. v8 public terbaik (0,56145) tapi blend
  v8 dibuat lokal setelah kernel mati di TabICL; `results/v8/v8_run.ipynb` sendiri tidak menghasilkan file itu.
- Opsi: (a) notebook rerun bersih dari Prioritas 1 jadi notebook final, dengan blend yang dipilih di OOF bersih;
  (b) kalau tetap pakai v8, notebook harus mereproduksi blend v8 (bobot rank di
  `07_validasi_dan_leaderboard/probe_submission_16okt.md`) dari model-model v8 dan dijalankan ulang utuh.
- Nama `ParticipantNumber_TeamName_Notebook`, isi banner tim, patuhi `ref/notebook/` dan `docs/context.md`.
- Kumpulkan sebelum 30 Okt 23:59 WIB (telat −1 poin/menit). Sisakan 2 hari untuk rerun kalau ada error.

## Prioritas 3 (opsional, ekspektasi kecil)
- **S4-like submission dengan GBDT bersih** sebagai pembanding (bagian GBDT v8 saja hampir sama dengan v8 di public).
- Pemilihan 2 submission akhir (kalau Kaggle minta memilih): satu yang terbaik di OOF bersih + satu yang paling
  konservatif (rank-average bobot positif, GBDT-heavy). Bahas hanya kalau user minta; jangan terus bilang "final".
- v20 (Causilo 16 estimator) hanya kalau kuota GPU masih sisa setelah Prioritas 1.

## Jangan dihabiskan waktunya lagi
EDA residual / fitur baru (40+ analisis null, dan v14 yang membaca baris mentah pun null), tuning LGB/TabM (v10–v12
null), EBM, objective ranking, stacker L2, linear_tree/monotone/ExtraTrees, pseudo-label, fitur mature-history,
pruning fitur, TabICL, stack LR dengan bobot negatif, stack per segmen, augmentasi historis, menambah seed, LB probing
label. Detail: `08_pelajaran/yang_tidak_efektif.md` dan `status_rekomendasi_lama.md`.

## Kalender usulan
| Tanggal | Kegiatan |
|---|---|
| 17–18 Okt | Tunggu kuota GPU reset (cek dengan push kecil). Siapkan builder rerun bersih (Prioritas 1), uji sintaks lokal saja |
| 18–20 Okt | Jalankan rerun bersih di Kaggle; analisis blend lokal di OOF bersih |
| 20–24 Okt | Maksimal 1–2 submit per hari untuk kandidat yang lolos OOF bersih (batas 3/hari) |
| 25–27 Okt | Pilih pendekatan final, rakit notebook final yang menghasilkan file yang disubmit |
| 28–30 Okt | Rerun reproducibility notebook final di Kaggle, rename, banner, kumpulkan sebelum 30 Okt 23:59 WIB |
