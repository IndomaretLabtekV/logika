# v22 (17 Okt): rerun bersih resep v8 di validation_v2

## Pertanyaan
Berapa sebenarnya kekuatan tiap keluarga model resep v8 kalau tidak ada keputusan apa pun yang memakai label fold OOF?
Dan berapa skor resep v8 itu sendiri di protokol bersih, sebagai pembanding semua kandidat berikutnya?

## Setup
- Kernel `riantamaputra/logika-v22-clean` (T4×2), push 08:30 WIB, selesai ~12:50 WIB. Builder `notebook/build_v22.py`
  menyematkan `features.py`, `nn_prep.py`, `valid2.py`, `run_valid2.py` → kode sama dengan screening lokal.
- GPU0: lgb, lgb_s, xgb, xgb_s, cat, cat_s × seed 42/7/2026 × 5 fold (78 menit). GPU1: TabM 3 seed (48 menit) + Causilo
  1 seed (31 menit). TabPFN tidak ikut (kernel CLI tanpa secret token; bobotnya di v8 hanya 0,045).
- Protokol: StratifiedKFold 5 per seed; early stopping di inner 1/7 train fold; refit di seluruh train fold dengan budget
  tetap (GBDT best iteration × 1,15; TabM best epoch). Output per model dengan application_id + `manifest.json`.
- Screening lokal fold 0 seed 42 sebelumnya: lgb 0,5189, lgb_s 0,5128, xgb 0,5196, xgb_s 0,5147, cat 0,5163,
  cat_s 0,5238, TabM 0,5231 (epoch terbaik 5); 6 GBDT 4,4 menit, TabM 1,5 menit. Satu fold sd ~0,008.

## Hasil (OOF pooled)
| Model | OOF bersih | OOF v8 (bias) |
|---|---|---|
| causilo | **0,52583** | 0,5241 |
| lgb / xgb / xgb_s | 0,52316 / 0,52310 / 0,52285 | 0,5239 / 0,5224 / 0,5241 |
| lgb_s / cat_s / cat | 0,52118 / 0,52077 / 0,51975 | 0,5245 / 0,5208 / 0,5213 |
| tabm | 0,52100 | 0,5257 |

Blend cross-fit (`eda/out/blend2_v22.txt`): resep v8 bersih **0,52426** (pembanding tetap); equal trees 0,52448
(−0,0004, P 0,35); trees 50% + causilo 50% 0,52527 (+0,0011, P 0,77, 4/5); greedy bagged 0,52480; causilo saja
0,52583 (+0,0013, P 0,73, 3/5). Belum ada yang lolos P>0 ≥ 0,9.

## Kesimpulan
- Urutan model terbalik dari kesan v3–v9: di protokol bersih **Causilo terbaik, TabM terlemah**. Konsisten dengan probe
  16 Okt (GBDT saja ≈ v8 di public, NN/FM saja turun).
- Korelasi rank antar model 0,97–0,999 → blend nyaris tidak menambah.
- Kandidat lokal ditulis ke `results/v24/` (`v8recipe_clean.csv`, `trees_causilo50.csv`), tidak disubmit.
- OOF/test v22 menjadi fondasi semua blend berikutnya (`results/v22/v22/{gbdt,nn,fm}`).
