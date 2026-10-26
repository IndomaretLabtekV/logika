# Kandidat dan OOF (pooled KS 70.000 baris)

Semua pipeline user: StratifiedKFold 5, shuffle, **seed 42**, protokol `validation_v2`. Pipeline teman: seed **31027**
(tidak bisa dibandingkan persis, tetapi skema sama). OOF lama v1–v9 bias (tidak dipakai).

## Ranking val tertinggi (semua sumber)
| Kandidat | OOF KS | AUC | Public | Status |
|---|---:|---:|---:|---|
| goal054 kumo_offset_l20p003_unweighted_d0p5 (teman) | **0,52930** | 0,84720 | **0,56470** | disubmit |
| goal054 calendar_transfer_l20p003_unweighted_d1p0 (teman) | 0,52917 | – | – | **tidak ada CSV test** (hanya metadata) |
| goal054 kumo_offset_l20p03_unweighted_d0p5 (teman) | 0,52908 | – | – | tidak ada CSV |
| teman Kumo 75% + spline 25% (public_056393_recipe) | 0,52851 | – | 0,56393 | disubmit (teman) |
| v56 Kumo seed 42 (legacy reference, 11/15 fit) | 0,52781 | – | – | tidak lengkap |
| **K64 2-seed** (user, cap300@28k ×2 + main_small ×2) | **0,52740** | 0,84797 | 0,56168 | disubmit |
| K64w (K64 s2026 + cap500@16k) | 0,52724 | 0,84789 | – | kembar K64 (ρ 0,9995) |
| v31b Kumo + Causilo | 0,52701 | – | 0,55709 | disubmit lama |
| K64 3-seed | 0,52685 | 0,84811 | – | kembar K64 |
| v28m Kumo legacy (fm_select 104, ctx 56k, e4) | 0,52665 | – | 0,56056 (v31c) | |
| K62 | 0,52662 | 0,84757 | 0,56203 | disubmit |
| REF = v59-analog (pohon6/Causilo/Kumo legacy 1/3) | 0,52625 | 0,84685 | (v59 0,56296) | pembanding utama |
| v31 OOF analog | 0,52613 | – | 0,56405 | |

## Member Kumo strict (pooled OOF, seed 42)
| Member | s2026 | s2027 | s2028 | Folder |
|---|---:|---:|---:|---|
| pool_cap500 @8k (protokol 23 Okt) | 0,51885 | – | – | `results/v60/strict_tfm/outer_f*/` |
| nested_small (union 104) | 0,50936 | – | – | idem |
| baseline protokol (main100, e4) | f0 saja 0,52629 | – | – | idem |
| cap500_ctx16k | 0,52426 | – | – | `results/v62/strict_variant/` |
| main_small (main100, e8, ctx penuh) | 0,52553 | 0,52644 | 0,52668 | v62 / v69 |
| cap300_ctx28k | 0,52549 | 0,52387 | 0,52713 | `results/v64/strict_variant/`, v69 |
| main150 | 0,52597 | 0,52606 | – | `results/v65/strict_variant/` |

## Kandidat blend terkunci (eda/221; per-cohort rank, bobot tetap)
| Kandidat | OOF | Δ REF (P>0, fold menang) |
|---|---:|---|
| K (recipe 2 member s2026) | 0,51960 | −0,0066 (0,00) |
| KT / KTC (recipe) | 0,52328 / 0,52311 | −0,0030 / −0,0031 |
| K62 / KT62 / KTC62 | 0,52662 / 0,52654 / 0,52634 | +0,0004 / +0,0003 / +0,0001 |
| K64 / K64w / KT64 (s2026) | 0,52689 / 0,52724 / 0,52564 | +0,0006 / +0,0010 / −0,0006 |
| **K64 2-seed** / K64w 2-seed | **0,52740** / 0,52677 | +0,0011 (0,67, 4/5) / +0,0005 |
| K64 3-seed | 0,52685 | +0,0006 (0,78) |
| K65 2-seed (cap300 + main150) | 0,52568 | −0,0006 |
| B2 = 0,8 K64 2-seed + 0,2 GRU-NN | 0,52730 | −0,0001 vs K64 |

## LGB (main232, v22 params, 4 thread) — basis eksperimen fitur siang hari
- `lgb_base4` pooled **0,52074** (fold 0,51892 / 0,52932 / 0,51759 / 0,51164 / 0,54362),
  `results/v66/series_oof_eval/oof_lgb_base4.parquet`.

## Kesimpulan posisi
- Target 0,53–0,54 belum tercapai di sumber mana pun (tertinggi 0,52930 teman).
- Tidak ada kandidat val tinggi tersisa yang punya CSV dan belum disubmit (sisa kembar rank atau tanpa CSV test).
