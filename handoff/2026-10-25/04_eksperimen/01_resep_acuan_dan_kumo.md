# Eksperimen 1 — Resep acuan dan variasi Kumo-Tabular (24 Okt)

Semua: Kumo-Tabular large v1.0.1 (checkpoint SHA `cc214843…`, diverifikasi `eda/strict_kumo_engine.py`), pool strict
896 kolom (main232 dengan reference harga fit-only + 664 fitur prefix teman `t__*`) yang dibangun ulang **per outer
fold** (`eda/219_build_outer_pools.py` → `results/v60/strict_tfm/pool_outer_f*/`), outer seed 42, kategori native
(vocab dari context). Test = rank per fold lalu rata-rata fold.

## 1.1 Resep acuan (protocol 23 Okt) — member-major, s2026
- `eda/212_strict_tfm_views.py` (+ `--memory-fraction`, fsync), driver antrean (dihapus, isinya di
  `08_kode_dan_infrastruktur/skrip_antrean_dihapus.md`).
- **pool_cap500 @8.192 context, e8: pooled 0,51885** (fold 0,5264/0,5291/0,5142/0,5014/0,5448) vs Kumo legacy v28m
  0,52665 (fold 0,5256/0,5380/0,5197/0,5151/0,5507) → kalah 4/5.
- **nested_small** (select_columns union → 104 kolom, context penuh 56k, e8): **0,50936**, kalah 5/5 (−0,017…−0,023
  per fold). 104 kolom didominasi fitur teman `t__*`.
- **Diagnosis:** `eda/223_strict_vs_legacy_features.py` — 17 fitur `pp_*` strict vs legacy Spearman 1,0000 (ΔKS
  univariat −0,00001) → reference strict bukan penyebab. Member `baseline` protokol (main232→100, e4) f0 **0,52629**
  ≈ legacy f0 0,52556 → pipeline strict sehat; penurunan berasal dari **seleksi union** (ranking KS univariat mengisi
  slot dengan varian DPD/ecom teman yang mirip) dan **context 8k** pada view lebar.
- Kandidat terkunci (amendment_01): K 0,51960, KT 0,52328, KTC 0,52311 (semua < REF 0,52625).
- **pool_uncapped** (896 kolom, 8k) tidak dijalankan 5-fold: OOM di 0,7 (23 Okt) dan view lebar 8k sudah terbukti
  kalah; seed 2027 resep juga tidak dijalankan → **penyimpangan dari resep acuan** (dicatat).
- **Median impute** (TLDR user) tidak ada di protocol 23 Okt → diuji terpisah (1.3): netral/negatif.

## 1.2 Context (C2) — inner fold 0 lalu 5-fold
- `eda/222_context_ssl_screen.py`, inner f0 (48k fit / 8k stop), s2026, e8, `results/v61/context_ssl_screen_f0/`.
- cap500: **8.192 → 16.384: KS 0,50494 → 0,51436 (+0,0094, CI −0,0021…+0,0240, P>0 0,95), AUC 0,83243 → 0,83875**.
  24.576 tidak dicoba (buffer 24.576×504×256×2 B ≈ 6,3 GB + overhead ~1,9 GB > VRAM).
- 5-fold (`eda/224_strict_tfm_variant.py`, protocol `protocol_v62_ctx16k.json`, fraction 0,85 + expandable segments):
  **cap500_ctx16k 0,52426** (fold 0,5307/0,5292/0,5187/0,5080/0,5464; menang 5/5 vs 8k, +0,0054). Peak 6,10 GiB.

## 1.3 Median impute (arm M)
- Inner f0: M_ctx8192 0,50283 vs 0,50494 (−0,0021, P 0,35); M_ctx16384 0,51448 vs 0,51436 (+0,0001, P 0,44);
  M_small 0,49676 vs 0,50047 (−0,0037, P 0,25) → **tidak dipakai**.

## 1.4 Lebar vs context pada VRAM tetap (226) — temuan positif
- `eda/226_width_context_tradeoff.py`, inner f0: **cap300 @28.672: KS 0,52533 / AUC 0,84142**; cap200 @40.960
  0,51831 / 0,84164; cap500 @16.384 0,51436 / 0,83875. (cap = 300 kolom acak per estimator dari 896, Latin shuffle.)
- 5-fold (`protocol_v64_cap300_ctx28k.json`): **cap300_ctx28k s2026 0,52549** (fold 0,5320/0,5317/0,5215/0,5116/
  0,5513; menang 5/5 vs cap500@16k).

## 1.5 View main_small dan main150
- main_small = main232 → select_columns 100 numerik + 4 kategori, context penuh, e8: s2026 0,52553, s2027 0,52644,
  s2028 0,52668.
- main150 (budget 150): inner +0,0034 (P 0,63, `eda/229`), 5-fold s2026 0,52597 / s2027 0,52606 ≈ main100 → null;
  K65 2-seed (cap300 + main150) 0,52568 < K64.

## 1.6 Seed tambahan
- Seed 2027: main_small 0,52644, cap300 0,52387 → **K64 2-seed 0,52740** (tertinggi protokol user).
- Seed 2028 (`protocol_v69_seed2028.json`, motivasi tim 0,57065 "3 seed"): main_small 0,52668, cap300 0,52713 →
  **K64 3-seed 0,52685** < 2-seed. Seed = noise (member 0,5239–0,5271).

## 1.7 Batas memori dan estimator
- e16 untuk cap300@28k: **OOM** (`eda/227`): +3,04 GiB di atas 4,80 GiB.
- **Penemuan:** `model.fit(num_estimators=E)` menyimpan cache KV per estimator sampai predict → memori naik dengan E.
  `eda/231_sequential_estimators.py`: E fit e1 berurutan (generator 2026·1000+b), rata-rata probabilitas → puncak =
  satu estimator. cap300@28k 8×e1 KS 0,52102 / AUC 0,84152 vs e8 0,52533 / 0,84142 (Δ −0,0043, P 0,26 → setara).
- Dengan itu **context penuh 48k pool lebar muat**: cap128@48k puncak 4,43 GiB, tetapi KS 0,52224 / AUC 0,84074
  (Δ −0,0031, P 0,34). 24 estimator sekuensial (cap300@28k): AUC 0,84135 → **datar**.
- Wide cap128@48k sebagai e8 biasa (`eda/229`): OOM.

## 1.8 Pool 777 (dedup + affine) dan anchored union
- `eda/228`: cap300@28k pada pool 777 (resep fit di 48k inner-fit) KS 0,52012 vs 0,52533 (−0,0052, P 0,18) → null.
- `eda/232`: anchored (104 main_small tetap + 196 union acak per member, 8×e1, ctx 28k) KS 0,51671 / AUC 0,84146
  (Δ −0,0086, P 0,08) → null.

## Kesimpulan bagian ini
- Tuas yang terbukti: **ukuran context** untuk view lebar (8k→16k +0,0054; trade kolom→context +0,0012).
- Plateau: semua varian pool lebar inner AUC **0,8407–0,8416**; seed/estimator/impute/dedup tidak menambah.
- Di public, Kumo-only strict ≈ 0,562 (K62 0,56203, K64 0,56168) — tidak lebih baik dari blend dengan pohon.
