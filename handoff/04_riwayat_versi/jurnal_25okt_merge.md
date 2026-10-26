# Jurnal 25 Okt — merge tiga pipeline tim

Mulai 01:11 WIB. Semua angka OOF di sini: pooled KS 70.000 baris, evaluasi pada StratifiedKFold 5 **seed 42**
(`valid2.outer`), rank per cohort (per outer fold) kecuali disebut lain. REF = 0,52625 (pohon6/Causilo/Kumo legacy 1/3).

## 1. Review tiga pipeline

| Aspek | Utama (eda/, v22–v69) | Teman (goal054) | DSC0152 (Agas v66) |
|---|---|---|---|
| Fitur | `features.py` ±800 kolom, pp_* (harga relatif kategori), rp_*, pool strict 896 | V3–V5 (±1.000), 90 kolom untuk Kumo; laten `re_*` (posterior empiris-Bayes harga/biro/telat, income geometrik) | `features.py` + 14 blok extra (598 kolom), **factor model** 12 sumber risiko (loading per fold dari label train fold) |
| Referensi tanpa label | strict: difit per partition | V5 per outer fold (sudah ada di results/v56), `re_*` prior dari customer train | **populasi penuh (train+test)**, tanpa label — bukan kebocoran label tapi tidak per-partition |
| Model | 6 GBDT (lgb/xgb/cat ×2), Causilo, Kumo large (context ≤56k) | Kumo large ctx 50k e8 + koreksi offset logistic (6k holdout) + spline-logistic 25% | LGB full/lean/fac, lgbx, Cat, XGB, spline-linear; blend `tm30` + ekstrapolasi `ext0.5` menjauh dari lgb_lean |
| Validasi | 5-fold seed 42, early stop inner 1/7 + refit ×1,15, OOF pohon rata-rata 3 seed outer | 5-fold seed 31027, satu seed, OOF/test per fold tidak tersimpan | 10-fold × 6 fold seed; **early stopping pada fold validasi** (OOF agak optimistis) |
| OOF terbaik | K64 2-seed 0,52740 | 0,52930 (seed 31027) | ext0.5·tm30·lgb_lean 0,52871 (global rank) / 0,52775 (rank per cohort seed 42) |
| Public terbaik | v31 0,56405 | **0,56470** | 0,56424 |
| Kekuatan | protokol paling bersih, Kumo strict, banyak null tercatat | koreksi offset laten yang terbukti transfer; fitur laten unik | pohon terkuat (KS kaki tunggal 0,524–0,526), ekstrapolasi transfer ke public |
| Kelemahan | Kumo-only mentok 0,562 public; pohon v22 lebih lemah dari DSC0152 | seleksi dari ±24 varian di fold yang sama; tidak ada OOF per fold | early stop di fold val; reference populasi; OOF tertinggi tidak menang public |

Overlap: harga relatif kategori (pp_* = ec_price_rel ≈ v4/v5 price), posterior keterlambatan (rp_* ≈ rh_rate1_smooth ≈
v3_repay_*_posterior ≈ re_late_*). Belum pernah dicoba: blend model lintas tiga pipeline; offset laten di atas
Kumo/pohon user; kaki DSC0152 di skema fold user; factor model DSC0152 sebagai input Kumo/offset.

## 2. Langkah (i): blend OOF yang sudah ada — `eda/250_merge_blend_existing.py`
Protocol dikunci sebelum skor: `artifacts/optimization/continuation_25okt/protocol_250_blend_existing.json`
(bobot sama, rank per cohort). **Eksplorasi**: kaki DSC0152 = OOF 10-fold×6 seed dengan early stop di fold val.

| Kandidat | OOF KS | AUC | Δ REF (P>0) | Δ vs komponen terbaik (P>0) | ρ test vs goal054 / v31 |
|---|---:|---:|---|---|---|
| D (DSC0152 v66 resep) | 0,52775 | 0,84686 | +0,0015 (0,75) | – | 0,9811 / 0,9846 |
| TD | 0,52581 | 0,84695 | −0,0004 (0,45) | −0,0019 (0,11) | 0,9841 / 0,9930 |
| **DK** (D + K64) | **0,52783** | **0,84865** | +0,0016 (0,92) | +0,0001 (0,66) | 0,9899 / 0,9948 |
| TDK | 0,52702 | 0,84804 | +0,0008 (0,79) | −0,0007 (0,39) | 0,9887 / 0,9967 |
| DCK | 0,52740 | 0,84809 | +0,0011 (0,88) | −0,0004 (0,48) | 0,9895 / 0,9980 |
| TDCK | 0,52658 | 0,84776 | +0,0003 (0,72) | −0,0012 (0,34) | 0,9887 / 0,9986 |
Komponen (rank global): T 0,52448, C 0,52493, KL 0,52665, K64 0,52740, D 0,52871 (kaki D: lgb_lean 0,52531,
xgb 0,52554, fac 0,52421). Bukti: `eda/out/250_blend_existing_v66.{json,png}`, CSV `results/v71/merge_blend/v66/`.
Kesimpulan: blend murah belum mencapai 0,53; pohon v22 (T) menurunkan OOF bila ditambah ke D; DK punya AUC tertinggi.

## 3. Langkah (ii): reproduksi resep teman di seed 42 — `eda/251_teammate_recipe_s42.py`
Protocol: `protocol_251_teammate_s42.json`. `re_*` dibangun ulang per fold (prior dari customer outer-train, tanpa
label; fungsi audit `distribution_summary` di-stub karena inkompatibel scipy). Kumo via `strict_kumo_engine.predict`.
Waktu: Kumo ±15–20 menit/fold (guard pause ±52% karena spike CPU), peak VRAM 3,58 GiB.

| Varian (seed 42) | Pooled KS | AUC | Mean-fold | Fold KS |
|---|---:|---:|---:|---|
| K5 Kumo V5 mentah | **0,52698** | 0,84690 | 0,52951 | 0,5316 0,5373 0,5176 0,5112 0,5498 |
| K5o + offset re_* | 0,52673 | 0,84659 | 0,52814 | 0,5297 0,5320 0,5187 0,5128 0,5475 |
| S spline | 0,51703 | 0,84112 | 0,52054 | |
| R0 0,75 K5 + 0,25 S (resep public 0,56393) | 0,52677 | 0,84701 | 0,52961 | |
| **R** 0,75 K5o + 0,25 S (resep goal054) | 0,52693 | 0,84677 | 0,52881 | 0,5279 0,5349 0,5186 0,5152 0,5475 |
Kesimpulan jujur: di seed 42 resep goal054 = 0,52693 (bukan 0,52930 seperti seed 31027) — setara K64 2-seed
(0,52740). Koreksi offset −0,0003 (tidak mereplikasi +0,0008) → gain offset di seed 31027 sebagian besar optimisme
seleksi/noise; transfer public +0,0008 tetap fakta, tapi satu titik. Bukti: `eda/out/251_teammate_s42.json`,
OOF/test `results/v71/teammate_s42/{oof,test}.parquet`.

## 4. Merge 252 (protocol `protocol_252_merge.json`, dikunci sebelum skor 251) — D masih eksplorasi (preds_v66)
| Kandidat | OOF KS | AUC | Δ REF (P>0) | Δ komponen terbaik (P>0, fold menang) | ρ test goal054 / v31 / K64 |
|---|---:|---:|---|---|---|
| RK | 0,52760 | 0,84820 | +0,0014 (0,89) | +0,0005 vs K64 (0,82, 3/5) | 0,9974 / 0,9937 / 0,9973 |
| **RD** | **0,52888** | 0,84826 | +0,0026 (0,93) | +0,0011 vs D (0,77, 3/5) | 0,9950 / 0,9900 / 0,9906 |
| **RDK** (primer) | 0,52837 | **0,84876** | +0,0021 (0,96) | +0,0006 vs D (0,72, 3/5) | 0,9955 / 0,9941 / 0,9958 |
| RDKC | 0,52764 | 0,84839 | +0,0014 (0,96) | −0,0001 | |
| RDKT | 0,52788 | 0,84838 | +0,0016 (0,92) | +0,0001 | |
Belum ada yang ≥0,53. Menunggu D seed 42 (inner-stop) untuk angka bersih.

## 5. DSC0152 v66 di fold seed 42, bersih (02:33–04:56 WIB)
`scripts/dsc0152/agas/v66/training/train_submission_v66.py --folds 5 --fold-seeds 42 --lgb-seeds 0,1,2,3,4
--valid2-folds --inner-stop --n-jobs 4 --out-prefix v66s42` (opsi baru opt-in; default tidak berubah). Cache fitur
dibangun ulang (`results/run_dsc_s42.sh`): full 598 / lean 490 / fac 553 = identik v66. Early stopping di inner 1/7
fold-train + refit ×1,15 (protokol valid2). Output `results/dsc0152/agas/v66/output/preds_v66s42.npz`.

| Kaki | KS | AUC | (v66 eksplorasi 10f×6 seed, early stop di val) |
|---|---:|---:|---:|
| fac_full | 0,52520 | 0,84588 | 0,52421 |
| lgb_full | 0,52487 | 0,84598 | 0,52497 |
| lgb_lean | 0,52457 | 0,84605 | 0,52531 |
| lgbx_full | 0,52401 | 0,84527 | 0,52409 |
| cat_full | 0,52213 | 0,84424 | 0,52523 |
| xgb_full | 0,52561 | 0,84595 | 0,52554 |
| linear | 0,51943 | 0,84167 | 0,52271 |
| **D resep ext0.5·tm30·lgb_lean** | **0,52614** | 0,84600 | 0,52871 |
→ optimisme versi eksplorasi ±0,0026 KS untuk resep D.

## 6. Merge 252 dengan D bersih (semua seed 42) — `eda/out/250_blend_existing_merge252_ds42.json`
| Kandidat | OOF KS | AUC | Δ REF (P>0) | Δ komponen terbaik (P>0, fold menang) | mean fold | ρ test goal054 / v31 / K64 |
|---|---:|---:|---|---|---:|---|
| RK | 0,52760 | 0,84820 | +0,0014 (0,89) | +0,0005 vs K64 (0,82, 3/5) | | 0,9974 / 0,9937 / 0,9973 |
| **RD** | **0,52852** | 0,84792 | +0,0023 (0,91) | +0,0020 vs R (0,90, **5/5**) | 0,53182 | 0,9950 / 0,9900 / 0,9906 |
| **RDK** (primer) | 0,52846 | **0,84856** | +0,0022 (**0,97**) | +0,0013 vs K64 (0,90, 3/5) | 0,53158 | 0,9955 / 0,9942 / 0,9958 |
| RDKC | 0,52792 | 0,84826 | +0,0017 (0,96) | +0,0008 (0,83) | | |
| RDKT | 0,52741 | 0,84824 | +0,0012 (0,88) | +0,0003 (0,73) | | |
Kesimpulan: merge tiga pipeline adalah OOF tertinggi di skema fold user (RD/RDK ±0,5285 vs K64 0,52740, REF 0,52625),
tetapi **belum 0,53** → tidak disubmit (aturan kuota). Menambah pohon v22 (T) atau Causilo menurunkan OOF.

## 7. Bobot cross-fit (255, protocol dikunci 05:01) dan Kumo V5 + factor (253)
- 255 (R, D, K64; grid simplex 0,1, D ≥ 0,2, pilih AUC di 4 fold lain): bobot terpilih ±R 0,2 / D 0,3 / K64 0,5;
  pooled **0,52774** < RDK bobot sama 0,52846 (Δ −0,0007, P>0 0,18). → bobot sama dipertahankan.
- 253 KF = Kumo V5 90 kolom + 6 kolom factor model DSC0152 (loading dari label outer-train): **0,52624** / AUC 0,84674
  vs K5 0,52698 → −0,0007, null (Kumo sudah menangkap sinyal faktor dari kolom mentah).
  Blend terkunci: KF_DK 0,52884 (Δ REF +0,0026 P 0,96), R_KF_DK 0,52845. KF_DK ≈ RDK + noise (KF kembar R) →
  tidak dipromosikan.

## 8. Satu entry point: `eda/260_merge_pipeline.py` + `artifacts/optimization/continuation_25okt/recipe_merge.json`
`--stage plan|build|blend|verify`. Satu skema fold (valid2.outer seed 42). Resep berisi semua perintah pembangun famili
(K64: 219 pool + 4 member 224 + 221; R: 166 fitur teman + 251 reverse/kumo/post/eval; D: `results/run_dsc_s42.sh`;
KF/KD opsional), output yang diharapkan (cache dilewati), dan blend terkunci RDK. `blend` menulis
`results/v71/merge_pipeline/{submission.csv,oof.parquet,manifest.json}` (SHA256 input/output); `verify` menghitung ulang
di memori dan mencocokkan byte: **lulus** (submission SHA `adee1103…`, OOF KS 0,52846).

## 9. KD = Kumo pada fitur DSC0152 (254; 598 kolom + 6 factor, view cap300@28k e8 s2026; peak 5,96 GiB, ±15–20 mnt/fold)
| Kandidat (protocol 254) | OOF KS | AUC | Δ REF (P>0) | Δ komponen terbaik (P>0, fold menang) | ρ test goal054 / v31 / K64 |
|---|---:|---:|---|---|---|
| KD | 0,52556 | 0,84763 | −0,0007 (0,46) | – | 0,9910 / 0,9903 / 0,9942 |
| **RDK_KD** | **0,52909** | **0,84874** | **+0,0029 (0,96)** | +0,0020 vs K64 (0,90, 3/5) | 0,9952 / 0,9941 / 0,9963 |
| RD_KD | 0,52903 | 0,84850 | +0,0028 (0,95) | +0,0026 vs R (0,95, 4/5) | 0,9953 / 0,9917 / 0,9934 |
Fold KS RDK_KD: 0,5304 0,5358 0,5276 0,5146 0,5507 (mean 0,53182). KD lemah sendiri tapi menambah keragaman (view Kumo
ketiga dengan feature engineering berbeda). OOF tertinggi seed 42 sejauh ini; **belum 0,53**.

## 10. R seed 2027 (256) — R2 = rata-rata 2 seed
| Varian s2027 | Pooled | AUC | | Kandidat 256 | OOF KS | AUC | Δ REF (P>0) | Δ vs R2 (P>0, fold) |
|---|---:|---:|---|---|---:|---:|---|---|
| K5 | 0,52833 | 0,84715 | | R2 | 0,52722 | 0,84714 | +0,0010 (0,72) | – |
| K5o | 0,52672 | 0,84685 | | **R2D** | **0,52942** | 0,84804 | **+0,0032 (0,94)** | +0,0022 (0,85, 4/5) |
| R0 | 0,52876 | 0,84719 | | R2DK | 0,52886 | 0,84861 | +0,0026 (0,97) | +0,0016 (0,84, 3/5) |
| R | 0,52631 | 0,84696 | | R2DK_KD | 0,52896 | 0,84876 | +0,0027 (0,96) | +0,0017 (0,84, 3/5) |
- Test ρ R2 vs submisi goal054 (0,56470) = **0,9997** → reproduksi resep teman di seed 42 setia (perbedaan OOF 0,5293 →
  0,5272 = efek skema fold/seed, bukan bug).
- Koreksi offset kembali negatif di seed 2027 (K5 → K5o −0,0016). Dua seed: rata-rata −0,0010. Gain offset di seed
  31027 tidak tereplikasi di seed 42.
- Kandidat merge di plateau 0,5289–0,5294; perbedaan antar kandidat < sd bootstrap (±0,0015) → anggap setara.

## 11. Notebook
`notebook/build_final.py`: default tetap resep v59 (14 sel; isi sel identik dengan sebelumnya, hanya id acak).
`LOGIKA_NOTEBOOK_RECIPE=merge` → `notebook/final_merge.ipynb` (11 sel): embed 260/250/221/251/254/valid2/common +
`recipe_merge.json` (SHA256, harus sama dengan repo), mode `replay` (re-blend cache + cek SHA CSV vs manifest) atau
`fresh` (`260 --stage build` semua famili, berjam-jam). Uji nbclient replay: **CSV byte-identik** (SHA `adee1103…`,
resep RDK). Diagnostik 258: korelasi rank OOF antar famili 0,975–0,991; D×R2 0,978 (pasangan paling beragam).

## 12. KD seed 2027 (257) dan bobot cross-fit kriteria KS (259)
| Kandidat | OOF KS | AUC | Δ REF (P>0) | Δ komponen terbaik (P>0) |
|---|---:|---:|---|---|
| KD2 (KD 2 seed; s2027 sendiri 0,52512) | 0,52593 | 0,84819 | −0,0003 (0,48) | – |
| RDK_KD2 | 0,52854 | 0,84880 | +0,0023 (0,96) | +0,0014 vs K64 (0,88) |
| R2DK_KD2 | 0,52892 | **0,84882** | +0,0027 (0,97) | +0,0017 vs R2 (0,82) |
| 259 cross-fit KS (R2, D, K64, KD2) | 0,52796 | 0,84844 | +0,0017 (0,93) | −0,0009 vs R2DK (P 0,31) |
Bobot cross-fit tidak stabil antar fold (KD2 0–0,3; R2 0,2–0,6) → bobot sama lebih baik (dua kali: 255 AUC, 259 KS).

## 13. Resep yang dipromosikan: **merge_R2DK** (R2 + D + K64, rank sama)
- Pooled OOF KS **0,52886**, AUC 0,84861, fold 0,5292 0,5362 0,5268 0,5153 0,5520; Δ REF +0,0026 (P 0,97).
- Dipilih karena = struktur primer yang dikunci (protocol 252: sepertiga tiap pipeline) + seed kedua (protocol 256);
  bukan maksimum OOF (R2D 0,52942) — semua kandidat di plateau 0,5285–0,5294, selisih < sd bootstrap.
- `eda/260_merge_pipeline.py` + `recipe_merge.json` (name merge_R2DK, output `results/v71/merge_pipeline_R2DK/`):
  blend + verify byte-identik (SHA `0edd6fef…`); notebook `final_merge.ipynb` replay nbclient **identik**.
- Preflight (`results/v71/merge_pipeline_R2DK/submission_preflight.json`): ρ test vs goal054 0,9954, v31 0,9943,
  K64 0,9959. Perkiraan public 0,5663 ± 0,0020. **Tidak disubmit**: aturan kuota hanya untuk OOF 0,53–0,54.

## Ringkasan sesi (per 08:53 WIB)
- Target 0,53–0,54 **belum tercapai**: plateau merge ±0,529 (maks R2D 0,52942). Di skema fold yang sama, resep teman
  goal054 = 0,52722 (2 seed), K64 0,52740, D 0,52614, REF 0,52625 → merge menambah ±+0,0015–0,002.
- Null baru: offset re_* di seed 42 (−0,0003/−0,0016), factor DSC0152 sebagai input Kumo (−0,0007), bobot cross-fit
  (AUC −0,0007, KS −0,0009), pohon v22/Causilo di atas merge (−0,0005…−0,0010).
- Ditemukan: kaki DSC0152 memakai early stopping di fold validasi → OOF v66 optimistis ±0,0026 KS (D 0,52871 → 0,52614).
- 261 (post-hoc, tidak dikunci): K5 mentah 2 seed tanpa offset/spline = 0,52747; K5x2_D 0,52870; K5x2_DK 0,52813 <
  R2DK 0,52886 → resep teman lengkap tetap dipakai.
- Belum dilakukan: commit (menunggu izin user), fresh run penuh `260 --stage build` dari nol (±8–10 jam GPU+CPU).

## 14. Iterasi lanjutan setelah hook (08:55–09:10 WIB)
- 262 T5 = LightGBM pada fitur V5 + re_* teman (1.027 kolom, inner stop 1/7 + refit ×1,15, 2 seed; protocol dikunci
  09:00): **0,52065** / AUC 0,84403 (fold 0,5260 0,5329 0,5161 0,5087 0,5397). R2DK_T5 0,52818 (< R2DK 0,52886),
  R2D_T5 0,52728 → null. Fitur V5 kuat untuk Kumo, tetapi pohon di atasnya lebih lemah dari pohon DSC0152 (D 0,52614).
- 263 audit referensi populasi DSC0152 vs per-partisi (fit customer outer-train): median harga kategori Δlog maks
  0,0000–0,0101, Spearman ec_price_rel ≥ 0,99999; gradenorm |Δz| maks 0,46–0,83 (outlier), Spearman ≥ 0,9999.
  → penyimpangan "reference difit per partition" pada D/KD **tidak material**; build ulang per partisi tidak
  diperlukan untuk klaim OOF. Bukti `eda/out/263_dsc_reference_audit.{json,png}`.
- Status: 13 kandidat/famili dievaluasi di seed 42; plateau merge 0,5285–0,5294. Target 0,53 tidak tercapai dengan
  bahan yang ada; tambahan famili (KF, KD, T5, seed kedua) memberi ≤ +0,0006 dan sebagian negatif.
- 264 blend logit-z per cohort (protocol dikunci 09:12; famili rank lewat probit): R2DK_logit 0,52899 (+0,0001 vs rank,
  P 0,46), R2DK_KD2_logit 0,52907 (+0,0002, P 0,45) → null; agregasi rank dipertahankan.

## 15. Uji reproducibility fresh (09:13–09:36 WIB) — `eda/265_fresh_check.py`, `results/v71/fresh_check/`
Fold 0 dibangun ulang dari nol ke folder terisolasi (`LOGIKA_MERGE_OUT`, opsi baru 251; default tidak berubah) lalu
dibandingkan dengan cache yang dipakai merge_R2DK:
| Unit | Byte identik | max abs diff | KS fold 0 |
|---|---|---:|---:|
| re_* (reverse, CPU) | ya | 0 | – |
| Kumo V5 s2026 (GPU fp16) | ya | 0 | 0,531569 |
| resep R (offset + spline) | ya | 0 | 0,527858 |
| K64 member cap300_ctx28k_s2026 (224, GPU) | ya | 0 | 0,532007 |
→ pipeline deterministik end-to-end, termasuk inferensi Kumo. D (DSC0152) fresh penuh berjalan (`--out-prefix v66s42fresh`).
