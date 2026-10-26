# Pipeline teman (goal054 / kumo_hybrid) — detail lengkap

Anggota tim yang sama (bukan tim 0,57065). Kode ada di repo sejak konsolidasi 21 Okt; **OOF dan prediksi test per fold
tidak tersimpan** (hanya `definition.json`, `fold*_metadata.json`, dan CSV submisi final di `results/teman/`).

## Riwayat skor teman
| Versi | Isi | OOF | Public |
|---|---|---:|---:|
| V3 | GBDT 10 fold | 0,51901 (10f) | 0,55845 |
| V5 | GBDT + TabM, 10 fold | 0,52446 (10f) | 0,55680 |
| Kumo large | V5 90 fitur, ctx 50k, e8, 3 fold seed 8247 | 0,52807 (3f) | 0,56262 |
| 75% V5 + 25% Kumo (`submission_kumo_blend.csv`) | | 0,52674 (3f) | belum diketahui |
| **Kumo 75% + spline 25%** (`submission_kumo_spline_exploratory.csv`) | 5 fold seed 31027 | 0,52851 | **0,56393** |
| **goal054 kumo_offset_l20p003_unweighted_d0p5** | + koreksi offset laten | **0,52930** (mean fold 0,53291, 5/5 fold membaik) | **0,56470** (disubmit sesi ini) |
| goal054 calendar_transfer_l20p003_unweighted_d1p0 | + koreksi kalender | 0,52917 (3/5) | tidak ada CSV |

## Resep kumo_offset (yang mendapat 0,56470)
1. **Split:** StratifiedKFold 5, shuffle, **seed 31027** (`FOLD_SEED` di `scripts/research_goal054.py`).
2. **Kumo-Tabular large** (sdm 0.2.1, bobot v1.0.1) pada ~90 kolom fitur **V5** (lihat `columns` di
   `artifacts/optimization/goal054/cv5_seed31027/kumo_offset_*/definition.json`: demografi, bunga/cicilan/rasio,
   skor biro, utilisasi, posterior keterlambatan v3, beban cicilan, saldo 3 bulan, ecom recency/spend, harga v4,
   kategori), **context 50.000 baris** berlabel diambil stratified dari outer-train (`context_indices(..., 50000,
   2026 + fold)`), **e8**, generator/torch seed **2026 + fold**.
3. **Koreksi offset**: logistic ridge (penalty l2 **0,003**, unweighted, kekuatan **0,5**) di atas logit Kumo memakai
   12 kolom: `re_price_log_posterior`, `re_price_posterior_sd`, `re_bureau_score_posterior`, `re_bureau_posterior_sd`,
   `re_income_geometric_mean`, `re_declared_income_log_gap`, `re_outflow_income_geometric_ratio`,
   `re_late_1_posterior`, `re_late_30_posterior`, `installment_observed_income_ratio`,
   `v2_existing_total_burden_observed`, `employment_type`. Difit di **±6.000 label outer-train di luar context**
   (fold validasi tidak pernah dipakai).
4. **Blend:** 75% Kumo-offset + 25% **spline-logistic** (n_knots 3, degree 2; `scripts/spline_model.py`, "reverse
   spline" dari `artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline`).
5. **Seleksi:** dipilih dari ±24 varian (8 kumo_offset: penalty 0,003/0,03 × balanced/unweighted × kekuatan 0,5/1,0;
   8 calendar_transfer; EBM; TabPFN-3.5) di fold yang sama → ada optimisme seleksi (CI bootstrap Δ melewati nol), tapi
   gain ternyata transfer ke public.

## Seed ringkas
| Bagian | Seed |
|---|---|
| Fold | 31027 |
| Sampling context 50k | 2026 + fold |
| Generator Kumo | 2026 + fold |
| Seed model (manifest) | 2026 ("2026+fold") — **satu seed** per fold (bukan ensemble) |
| Riset lama | 3 fold seed 8247 |

## File yang tersedia lokal
- Fitur teman per outer fold **seed 42** (sama dengan fold user!): `results/v56/teammate_features/outer_s42_f{0..4}/
  data/processed/` berisi `application_features.parquet`, `_extended`, `_v3`, `_v4`, `_v5`, `bureau_latest.parquet`,
  `prefix_features_extended.parquet` (1,4 GB per fold). Juga seed 7 (f0–f2) dan seed 2026 (f0–f1).
- Fitur laten `re_*` dibuat oleh `scripts/reverse_synthetic_patterns.py` / `validate_reverse_features.py`
  (`data/processed/application_features_reverse.parquet` — **tidak ada di lokal**, harus dibangun ulang).
- Script: `scripts/research_goal054.py`, `research_context_correction054.py`, `hybrid_kumo.py`, `kumo_tabular.py`,
  `spline_model.py`, `build_features_v3/v4/v5.py`, `extend_features.py`, `analyze.py`. Env: Python 3.11,
  `environments/teman/requirements-goal054.txt`.
- CSV submisi teman: `results/teman/*.csv` (lihat tabel di atas).

## Mengapa penting
- Pipeline ini **terbaik di public** dan sangat berbeda dari user (fitur V5 + laten `re_*`, spline, offset). Spearman
  test goal054 vs v31 0,98670, vs K62 0,98924 → ruang keragaman untuk blend dengan pohon user masih ada.
