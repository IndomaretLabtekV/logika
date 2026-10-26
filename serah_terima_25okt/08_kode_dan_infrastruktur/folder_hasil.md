# Peta folder hasil (tidak disalin; ada di disk)

| Folder | Isi |
|---|---|
| `results/v60/strict_tfm/pool_outer_f{0..4}/` | pool strict 896 kolom per outer fold (seed 42) + manifest hash |
| `results/v60/strict_tfm/pool_inner_f0/` | pool inner fold 0 (48k fit) untuk screening |
| `results/v60/strict_tfm/outer_f*/` | member resep acuan (pool_cap500_s2026, nested_small_s2026, baseline f0) |
| `results/v60/strict_tfm/quarantine_crash_2026-10-24/` | pool f2 rusak (hang #2) |
| `results/v60/candidates/{K,KT,KTC}__<tag>/` | OOF, report.json, submission.csv kandidat terkunci (1m, 2m, recipe_s2026, ctx16k_only, v62_s2026, v64_K64, v64_K64w, K64_2seed, K64w_2seed, K64_3seed, K65_s2026, K65_2seed, main_small_2seeds) |
| `results/v61/*_f0/` | screening inner fold 0 (context_ssl, width_context, estimators, dedup_pool, full_context_budget, sequential_estimators, anchored_union) |
| `results/v62/strict_variant/` | cap500_ctx16k_s2026, main_small_s2026/s2027 |
| `results/v63/{k62_kumo_ctx16k, v70b_k64_2seed}/` | submisi K62 & K64 2-seed (+ preflight, analisis) |
| `results/v64/strict_variant/` | cap300_ctx28k_s2026/s2027 |
| `results/v65/strict_variant/` | main150_s2026/s2027 |
| `results/v66/` | dart, series_oof (fitur per fold: ecom, repay, knn, coles_*, ruin, hunt, analog), series_oof_eval (lgb_base4, lgb_so_*), class_weight, teammate_select |
| `results/v67/coles/` | embedding CoLES |
| `results/v68/gru_nn/` | OOF GRU-NN per fold |
| `results/v69/strict_variant/` | seed 2028 (main_small, cap300_ctx28k) |
| `results/v70/goal054_kumo_offset/` | submisi goal054 teman (0,56470) + preflight/analisis |
| `results/v56/teammate_features/outer_s{42,7,2026}_f*/` | build fitur teman per outer fold (V3/V4/V5, 1,4 GB/fold) |
| `results/v22/v22/{gbdt,fm,nn}/` | OOF/test pohon 6 GBDT, Causilo, TabM (validation_v2, 3 seed outer) |
| `results/v28m/` | Kumo legacy context penuh 5 fold seed 42 |
| `results/dsc0152/` | output sumber DSC0152: `agas/<v>/output` (preds npz, summary, log), `agas/shared/harness_output`, `dimaz/model_output`, `logs/` |
| `results/teman/*.csv` | CSV submisi teman (termasuk goal054 kumo_offset) |
| `results/v31/`, `results/v32/`, `results/v41/` | submisi v31/v31b/v31c, v32, v59 (v31_lowvar.csv) |
| `artifacts/optimization/continuation_23okt/`, `continuation_24okt/` | laporan & protocol |
| `artifacts/optimization/goal054/`, `kumo_hybrid/`, `kumo/`, `kumo_improvement/` | metadata riset teman |
| `eda/out/` | log (`220_recipe.log`, `237_series.log`, `240b.log`, `242_gru.log`, `244_seed2028.log`, …), telemetri guard (`2xx_telemetry.csv`), gambar |
