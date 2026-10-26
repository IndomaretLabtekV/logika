# Peta script yang dibuat/dipakai sesi ini (`eda/`)

Semua dijalankan dari `eda/` dengan `../.venv/bin/python`, pola `OMP_NUM_THREADS=4 taskset -c 0-11 nice -n 10`.

## Inti (pustaka)
| File | Fungsi |
|---|---|
| `valid2.py` | protokol validation_v2: `outer(y, seed)` StratifiedKFold 5, `inner(y_tr, seed, fold)` split 1/7, `frames()`, `lgb_fit`, `run()` (OOF/test, **tulis non-atomik**), param LGB/XGB/Cat, `fm_select`, `ks`, `ks_nan` |
| `common.py` | ROOT, DATA, `ks` |
| `features.py` | builder fitur main232 (`v5_features(..., reference_customers=None)` — parameter baru, default = resep historis) |
| `strict_feature_pool.py` | `build_pool(fit_ids, out_dir)` → pool strict 896 kolom (main232 reference fit-only + prefix teman), manifest hash |
| `strict_kumo_engine.py` | `predict(frame, fit_rows, query_rows, labels, columns, model_seed, n_estimators, max_context, column_cap, chunk, memory_fraction)` Kumo large terverifikasi; `prepare_inputs`, `technical_preflight` |
| `nested_input_pools.py` | `select_columns(X, y, fit_rows, n_numeric=100, extend_shortlist)` (fm_select KS univariat + Spearman<0,95) |
| `thermal_guard.py` | guard termal (lihat `cara_menjalankan.md`) |

## Script eksperimen sesi ini
| Script | Fungsi | Output |
|---|---|---|
| `212_strict_tfm_views.py` | resep protocol 23 Okt per member/fold (`--stage outer --fold k --member m --memory-fraction`) | `results/v60/strict_tfm/outer_f*/` |
| `213_evaluate_strict_tfm.py` | evaluator resep 6 member (dari 23 Okt; tidak dipakai karena resep tak lengkap) | – |
| `219_build_outer_pools.py` | build pool strict 5 outer fold (`os.sync`) | `results/v60/strict_tfm/pool_outer_f*/` |
| `221_evaluate_recipe_candidates.py` | kandidat terkunci K/KT/KTC/REF dari member mana pun (`root::member`, `--tag`), bootstrap, CSV | `results/v60/candidates/{K,KT,KTC}__<tag>/`, `eda/out/221_candidates_<tag>.png` |
| `222_context_ssl_screen.py` | screening inner f0: C2 context, M median impute, S3 SSL | `results/v61/context_ssl_screen_f0/` |
| `223_strict_vs_legacy_features.py` | diagnosis pp strict vs legacy | `eda/out/223_strict_vs_legacy_pp.png` |
| `224_strict_tfm_variant.py` | varian 212 dengan protocol view kustom (`columns: all/small/main_small/main_150`, `impute`, `memory_fraction`, `max_context`) | `results/v62`, `v64`, `v65`, `v69` `/strict_variant/` |
| `225_prepare_submission.py` | salin + validasi CSV kandidat, perkiraan public, preflight JSON | `results/v63/<nama>/` |
| `226_width_context_tradeoff.py` | screening cap300@28k / cap200@40k | `results/v61/width_context_f0/` |
| `227_cap300_estimators.py` | e16 (OOM) | `results/v61/estimators_f0/` |
| `228_dedup_pool_cap300.py` | pool 777 | `results/v61/dedup_pool_f0/` |
| `229_full_context_budget.py` | main100/main150 full ctx, cap128@48k (OOM) | `results/v61/full_context_budget_f0/` |
| `231_sequential_estimators.py` | E fit e1 berurutan (8x1, 24x1, cap128 full) | `results/v61/sequential_estimators_f0/` |
| `232_anchored_union_view.py` | anchored union | `results/v61/anchored_union_f0/` |
| `233_group_overdispersion.py` | dispersi residual per kelompok | `eda/out/233_group_overdispersion.png` |
| `234_crossfit_group_offsets.py` | offset logit kelompok cross-fit | `eda/out/234_group_offsets.{json,png}` |
| `235_crossfit_member_stack.py` | stacking LR cross-fit | `eda/out/235_member_stack.{json,png}` |
| `236_dart_screen.py` | LGB DART vs base (juga menghasilkan lgb_base 8 thread) | `results/v66/dart/` |
| `237_series_oof.py` | series OOF per baris (ecom/repay), nested | `results/v66/series_oof/<tabel>_f<k>.parquet` |
| `238_series_oof_eval.py` | evaluator LGB main232 + fitur tambahan per fold, paired vs `lgb_base4` (dibuat ulang otomatis) | `results/v66/series_oof_eval/` |
| `239_knn_target_mean.py` | KNN target mean nested | `results/v66/series_oof/knn_f*.parquet` |
| `240_coles_embeddings.py` | CoLES (juga dipakai 242 untuk encoder & prep urutan) | `results/v67/coles/`, `series_oof/coles_*` |
| `241_cashflow_ruin_sim.py` | simulasi ruin arus kas | `series_oof/ruin_f*.parquet` |
| `242_gru_series_nn.py` | GRU-NN + blend 0,75 LGB | `results/v68/gru_nn/` |
| `243_gru_blend_check.py` | GRU di atas REF / K64 | stdout |
| `246_error_band_hunt.py` | 42 agregat baru, KS pita | `series_oof/hunt_f*.parquet`, `eda/out/246_band_hunt.png` |
| `247_analog_target.py` | target analog pinjaman lama | `series_oof/analog_f*.parquet` |
| `248_class_weight_screen.py` | bobot kelas | `results/v66/class_weight/` |
| `249_nested_teammate_select.py` | seleksi nested fitur teman untuk pohon (tak selesai) | `results/v66/teammate_select/log_lgb_tsel30.csv` |

Nomor 220, 230, 244, 245 dipakai sebagai label antrean (skrip dihapus, isi di `skrip_antrean_dihapus.md`).

## Protocol/amendemen yang dikunci sebelum skor
`artifacts/optimization/continuation_24okt/`: `amendment_01.json` (kandidat K/KT/KTC, deviasi engineering, perubahan
urutan, kandidat v62), `protocol_v62_ctx16k.json`, `protocol_v64_cap300_ctx28k.json`, `protocol_v65_main150.json`,
`protocol_v69_seed2028.json`, `notebook_verification_24okt.json`. Desain SSL/context:
`docs/integration/protokol_context_ssl_24okt.md` (+ amendemen 02:10).
