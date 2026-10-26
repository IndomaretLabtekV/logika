# Skor dan semua submisi (per 25 Okt 00:05 WIB)

## Public terbaik
- **0,56470 — goal054 kumo_offset (pipeline teman)**, ref 57600249, 24 Okt 23:21 WIB, file
  `results/v70/goal054_kumo_offset/submission.csv` (salinan dari
  `results/teman/submission_goal054_kumo_offset_l20p003_unweighted_d0p5_w1.csv`, SHA `5d4d8b81…`).
- Sebelumnya: v31 0,56405 (6 GBDT + Causilo + Kumo context penuh, 17 Okt).

## Kuota
- Window 24 Okt 07:00 → 25 Okt 07:00: **2/3 terpakai** (goal054, K64 2-seed); 1 slot tidak dipakai.
- K62 (05:15 WIB 24 Okt) masuk window sebelumnya (23 Okt 07:00 → 24 Okt 07:00).
- Catatan: `kaggle competitions submissions` hanya menampilkan submisi akun `riantamaputra`; submisi teman dari akun
  mereka (0,56393, 0,56262, 0,55845, 0,55680) tidak tampil tapi tetap memakai kuota tim yang sama.

## Submisi sesi ini (semua dengan perkiraan sebelum upload)
| Ref | Waktu WIB | Kandidat | OOF | Perkiraan | Public | Analisis |
|---|---|---|---:|---:|---:|---|
| 57573242 | 24 Okt 05:15 | **K62** Kumo-only strict: cap500 ctx16k + main_small (s2026) | 0,52662 | 0,5605 ± 0,0027 | **0,56203** | +0,0015 vs Kumo-only legacy v31c (0,56056) pada OOF setara; z +0,57 |
| 57600249 | 24 Okt 23:21 | **goal054 kumo_offset** (teman) | 0,52930 (seed 31027) | 0,5643 ± 0,0015 | **0,56470** | rekor tim; +0,00077 vs kembarannya 0,56393 (ρ 0,99965); z +0,27 |
| 57600277 | 24 Okt 23:22 | **K64 2-seed** Kumo-only strict: cap300 ctx28k ×2 + main_small ×2 | 0,52740 | 0,5628 ± 0,0015 | **0,56168** | gain OOF +0,0008 vs K62 tidak transfer; z −0,75 |

Bukti per submisi: `submission_preflight.json` + `post_submission_analysis.json` di
`results/v63/k62_kumo_ctx16k/`, `results/v70/goal054_kumo_offset/`, `results/v63/v70b_k64_2seed/`.

## Riwayat public lengkap (akun riantamaputra + teman yang tercatat)
| Public | Versi | Isi |
|---:|---|---|
| **0,56470** | goal054 (teman) | Kumo V5 + spline 25% + koreksi offset laten |
| 0,56424 | DSC0152 Agas v66_04 / v67_01 | ext0.5 · tm30 · lgb_lean (+ v70_07 0,56419, v68_04 0,56411, v61 0,56312) |
| 0,56405 | v31 | 6 GBDT 1/3 + Causilo 1/3 + Kumo full ctx 1/3 |
| 0,56393 | teman | Kumo 75% + spline 25% |
| 0,56376 | v32 | 6 GBDT 1/3 + Kumo 2/3 |
| 0,56337 | v39a | pohon saja |
| 0,56298 | v35 | 6 GBDT .3 + Kumo .3 + Causilo .2 + TabM .1 + LR .1 |
| 0,56296 | v59 | 6 GBDT / Causilo 5 seed / Kumo-70k 3 draw (resep notebook sekarang) |
| 0,56271 | v34 | resep notebook lama (Kumo-70k) |
| 0,56262 | teman | Kumo large murni |
| 0,56203 | K62 | (sesi ini) |
| 0,56168 | K64 2-seed | (sesi ini) |
| 0,56145 | v5/v8 | |
| 0,56133 | probe S4 | bagian GBDT v8 |
| 0,56123 | v6 | |
| 0,56056 | v31c | Kumo saja |
| 0,56046 | (21 Okt, tanpa deskripsi) | |
| 0,55917 | v9 | |
| 0,55866 / 0,55835 / 0,55778 | S5 / v39c / S3 | NN-part / LR rank-gauss / stack |
| 0,55845 / 0,55680 | teman V3 / V5 | |
| 0,55709 | v31b | Causilo + Kumo tanpa pohon |
| 0,55333 | v39b | Causilo saja |
| 0,54545 / 0,54192 / 0,53139 | v3 / v2 / v1 | |

Ledger lengkap sumber DSC0152 (root qc_00 0,56171, v61_00 0,56299, v70_06 0,56040, v70_01 0,55801, …): `docs/dsc0152/README.md`.

## Pola transfer CV → public yang terbukti
- Offset public − OOF: pohon +0,0389, LR +0,0372, Kumo legacy +0,0339, **Kumo strict +0,0354 (K62) / +0,0343 (K64)**,
  Kumo+spline teman +0,0354, goal054 +0,0354, Causilo +0,0292.
- Blend tanpa pohon jatuh di public (v31b). Pohon ≥ 1/3 di blend adalah aturan praktis lama.
- Kenaikan OOF kecil pada Kumo-only (+0,0008 K64 vs K62) **tidak** transfer; kenaikan koreksi offset teman (+0,0008)
  **transfer** (+0,0008).
