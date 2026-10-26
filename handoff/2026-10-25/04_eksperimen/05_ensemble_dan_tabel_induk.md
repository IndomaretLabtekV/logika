# Eksperimen 5 — Ensemble + tabel induk semua eksperimen sesi ini

## 5.1 Stacking cross-fit (235)
- `eda/235_crossfit_member_stack.py`: L2-logistic di atas logit-rank 9 member Kumo strict (+ pohon6, Causilo), fit di
  4 fold, nilai di fold ke-5 (bukan nested end-to-end). KS 0,52596–0,52652, **Δ vs K64 2-seed −0,0009…−0,0014**
  (P 0,38–0,52); bobot pohon di stack ~0,04 → null.

## 5.2 Blend terkunci lain
- Semua di `eda/221_evaluate_recipe_candidates.py` (lihat `03_posisi_sekarang/kandidat_oof.md`). Menambah pohon v22 ke
  K64 (KT/KTC) **menurunkan** OOF (KT64 2-seed 0,52507; KTC 0,52544), kontras dengan public di mana pohon membantu.

## Tabel induk (urut waktu)
| # | Script | Eksperimen | Skala | Hasil utama | Status |
|---|---|---|---|---|---|
| 1 | 219 | build pool strict per outer fold | CPU | 5 pool, f2 dikarantina setelah hang | infra |
| 2 | 212/221 | resep acuan: cap500@8k | 5f | 0,51885 | null |
| 3 | 212/221 | resep acuan: nested_small union | 5f | 0,50936 | null |
| 4 | 223 | pp strict vs legacy | CPU | ρ 1,0000 | diagnosis |
| 5 | 212 | baseline protokol main100 e4 | f0 | 0,52629 ≈ legacy | diagnosis |
| 6 | 222 | context 8k→16k (cap500) | inner | +0,0094 | **positif** |
| 7 | 222 | median impute | inner | ±0 / −0,0037 | null |
| 8 | 224 | cap500_ctx16k | 5f | 0,52426 (+0,0054 vs 8k) | **positif** |
| 9 | 224 | main_small e8 | 5f | 0,52553 | member |
| 10 | 221 | K62 | 5f | 0,52662 → public 0,56203 | submit |
| 11 | 222 | SSL S3 pseudo tersampel | inner | −0,0022 | null |
| 12 | 226 | cap300@28k / cap200@40k | inner | 0,52533 / 0,51831 | **positif** (cap300) |
| 13 | 224 | cap300_ctx28k | 5f | 0,52549 | **positif** |
| 14 | 221 | K64 / K64w | 5f | 0,52689 / 0,52724 | kembar K62 |
| 15 | 224 | seed 2027 | 5f | K64 2-seed 0,52740 → public 0,56168 | submit |
| 16 | 227 | e16 | inner | OOM | engineering |
| 17 | 228 | pool 777 dedup | inner | −0,0052 | null |
| 18 | 229 | main150 / cap128@48k e8 | inner | +0,0034 / OOM | — |
| 19 | 224 | main150 5f ×2 seed | 5f | 0,52597/0,52606; K65 0,52568 | null |
| 20 | 231 | 8×e1 sekuensial; cap128@48k | inner | setara e8; 0,52224 | plateau |
| 21 | 232 | anchored union | inner | −0,0086 | null |
| 22 | 233/234 | overdispersion / offset kelompok | CPU 5f | bulan +0,0007 | null |
| 23 | 235 | stacking cross-fit | 5f | −0,0009 | null |
| 24 | 236 | DART | f0 | −0,0062 | null |
| 25 | 237/238 | series OOF ecom / cicilan | f0 / 5f | −0,0003 / −0,0013 | null |
| 26 | 239 | KNN target mean | f0 | −0,0040 | null |
| 27 | 240 | CoLES ecom/cicilan/gabungan | 5f | −0,0026/−0,0007/−0,0007 | null |
| 28 | 241 | simulasi ruin arus kas | 5f | −0,0004 | null |
| 29 | 242/243 | GRU-NN | 5f | +0,0019 vs LGB; −0,0001 vs K64 | null utk blend terbaik |
| 30 | 224 | seed 2028 (3-seed) | 5f | 0,52685 | null |
| 31 | 231 | 24 estimator sekuensial | inner | AUC datar | null |
| 32 | 246 | 42 agregat di pita KS | 5f | −0,0024 | null |
| 33 | 247 | target analog | 5f | +0,0006 | null |
| 34 | 248 | bobot kelas / label smoothing | 5f | +0,0001 / −0,0026 / −0,0003 | null |
| 35 | 249 | seleksi nested fitur teman (pohon) | 4/5f | ≈ +0,0002 | null |
| 36 | 225 | submisi goal054 teman | – | public **0,56470** | **rekor** |

Inner = inner-stop 8k fold 0 (noise KS ±0,01; AUC lebih stabil). 5f = 5-fold outer pooled 70k. f0 = outer fold 0
saja (sd ~0,008 — terbukti menipu pada #25).
