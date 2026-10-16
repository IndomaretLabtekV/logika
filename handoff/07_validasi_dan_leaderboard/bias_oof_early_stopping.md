# Temuan terpenting: OOF v1–v9 bias karena early stopping di fold validasi

Ditemukan 16 Okt 2026 setelah 3 probe submission (lihat `probe_submission_16okt.md`). Temuan ini menjelaskan kenapa
**setiap** kandidat yang "lebih bagus dari v8 di OOF" justru turun di public.

## Apa yang salah
Semua model di notebook v1–v9 (`notebook/build_notebook.py`, `build_notebook_v8.py`) memilih iterasi/epoch terbaik
memakai **label fold validasi yang sama dengan yang kemudian dipakai untuk OOF**:

| Model | Kode (v8/v9) | Akibat |
|---|---|---|
| LightGBM | `lgb.train(..., valid_sets=[dva], callbacks=[lgb.early_stopping(200)])`, `dva` = baris `vai` | `best_iteration` dipilih pakai label OOF |
| XGBoost | `xgb.train(..., evals=[(dva,'va')], early_stopping_rounds=200)` | sama |
| CatBoost | `m.fit(pool_tr, eval_set=pool_va)` dengan `use_best_model=True, od_wait=200` | sama |
| TabM | `auc = roc_auc_score(y[vai], ...)`, simpan state terbaik, patience 8 | epoch terbaik dipilih pakai label OOF; kurva AUC TabM berisik dan memuncak dini (epoch ~6), jadi bias paling besar |
| TabPFN / Causilo / TabICL | tanpa early stopping | bersih |

Prediksi **test** tidak bias dengan cara yang sama (iterasinya dipilih di fold, lalu dipakai untuk test). Yang bias adalah
**angka OOF**, dan semua keputusan yang diambil dari OOF: bobot blend greedy, stacking, dan perbandingan antar versi.

## Seberapa besar
Pembanding: run v10/v11 memakai early stopping di **inner split** (fold validasi tidak pernah dipakai untuk stopping).

| Model | OOF v8 (stop di fold validasi) | OOF kontrol bersih | Selisih |
|---|---|---|---|
| LightGBM (param v8) | 0,5239 (3 seed) | 0,5191 (v10 control, 1 seed, fit di 2/3 train fold) | ~0,005 |
| TabM (config v8) | 0,5257 (3 seed) | 0,5175 (v11 control, 1 seed, inner 80/20 lalu refit epoch tetap) | ~0,008 |

Sebagian selisih karena jumlah seed dan ukuran data latih berbeda, jadi angkanya kasar. Arahnya konsisten: **TabM
lebih "menggelembung" daripada GBDT** (~0,003 lebih besar).

## Bukti dari public (16 Okt)
v8 = 0,467 bagian GBDT + 0,533 bagian NN/FM (bobot rank direkonstruksi dari submission v8, korelasi 0,99999999993).

| Submission | OOF KS | Public |
|---|---|---|
| v8 penuh | 0,52703 | 0,56145 |
| S4 = bagian GBDT v8 saja | 0,52376 (−0,0033) | 0,56133 (−0,0001) |
| S5 = bagian NN/FM v8 saja | 0,52776 (+0,0007) | 0,55866 (−0,0028) |

Menurut OOF, NN/FM unggul 0,0040 atas GBDT. Di public GBDT unggul 0,0027. Ayunan 0,0067 dengan sd 0,0036 → **z = 2,3,
p ≈ 0,03**. Ini konsisten dengan bias OOF TabM yang lebih besar.

Rangkaian yang sekarang terjelaskan:

| Submission | Δ OOF vs v8 | Δ public vs v8 |
|---|---|---|
| v6 | ~+0,001 | −0,0002 |
| v9 (stack LR 11 model) | +0,0007 (held-out vs v8 blend) | −0,0023 |
| S3 (stack LR + TabICL17) | +0,0011 | −0,0037 |
| S5 (NN/FM saja) | +0,0007 | −0,0028 |

Semua kandidat yang dioptimasi di OOF v8 condong ke TabM (bobot besar TabM, atau bobot negatif yang memanfaatkan pola
OOF), dan semuanya turun di public.

## Konsekuensi praktis
1. **Angka OOF v1–v9 tidak bisa dipercaya untuk memilih bobot blend atau membandingkan keluarga model.** Perbandingan
   dalam satu keluarga (misal LGB vs LGB) kira-kira masih valid karena biasnya mirip.
2. Bobot blend v8 (TabM + tabm_aug 35%) kemungkinan terlalu berat ke NN. Public mengisyaratkan blend yang lebih berat
   ke GBDT lebih aman.
3. Stack logit-LR dengan bobot negatif (v9, S3) paling rentan karena mengeksploitasi pola OOF yang bias.
4. Semua bench lokal `eda/s01_stack_bench.py` memakai OOF v8 sebagai base, jadi juga terkena bias ini. Eksperimen
   v10–v21 sendiri memakai inner split (bersih), tapi dinilai dengan menambahkannya ke stack v8 yang bias.

## Perbaikan yang disarankan (belum dikerjakan)
Jalankan ulang pipeline v8/v9 di Kaggle dengan:
- early stopping di **inner split** dari train fold (misal 1/7 train fold, seperti `splits(y[tr], 7, 300 + fold)` di
  v15–v21), atau jumlah iterasi tetap hasil inner CV;
- TabM: pilih epoch di inner split lalu refit epoch tetap di seluruh train fold (protokol v11);
- simpan OOF + test per model dengan ID;
- lalu pilih bobot blend dari OOF bersih ini (rank-average bobot positif, dinilai di held-out halves), bukan dari public.
Butuh GPU (TabM, TabPFN, Causilo). Kuota GPU Kaggle mingguan habis 16 Okt, reset mingguan.
