# Kronologi sesi 24 Okt 00:30 WIB – 25 Okt 00:05 WIB

Jurnal mentah dengan semua angka: `handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md`.

## Malam–pagi 24 Okt: resep acuan dan context
| Waktu | Kejadian |
|---|---|
| 00:32 | Goal dipasang. Audit: tidak ada submisi sejak v59 (21 Okt) → 3 slot. Baca laporan 23 Okt, protocol, handoff, SSL protokol, script 212–218. |
| 00:34 | Mulai `eda/219_build_outer_pools.py` (pool strict per outer fold, CPU ~30 dtk/fold). |
| ~00:35 | **Hang #2** (force shutdown user) saat pool f2; tidak ada job GPU. f2 manifest 0 byte → dikarantina, f0/f1 lolos hash, f2–f4 dibangun ulang. 212 diberi fsync, 219 `os.sync()`. |
| 00:41 | `run_220_recipe.sh` (member-major) + `eda/thermal_guard.py`. Amendemen `amendment_01.json` dikunci (kandidat K/KT/KTC). Guard awal pause tiap spike 97 °C → diubah ke "3 sampel berturut >95 °C". |
| 01:33 | **cap500 s2026 5-fold pooled 0,51885** (legacy Kumo 0,52665). Evaluator 221 mereproduksi REF v59-analog 0,52625. |
| 01:34 | Ganti antrean; insiden pgrep menangkap shell sendiri → 2 job GPU paralel ±30 dtk (dibunuh, cache aman). Guard pakai pola ter-anchor. |
| 01:44–02:28 | **nested_small s2026 pooled 0,50936** (kalah 5/5 vs legacy). Diagnosis 223: `pp_*` strict = legacy (ρ 1,0000). |
| 02:10 | Amendemen protokol: C2 max context 16.384 (24.576 tak muat), arm median impute. |
| 02:38 | Member baseline protokol f0 0,52629 ≈ legacy 0,52556 → penurunan berasal dari seleksi union (fitur teman). Bug 222 manifest (tuple vs list) diperbaiki. |
| 02:34–03:03 | Screening inner: C2 16k 0,51436 vs 8k 0,50494 (+0,0094); median impute ±0. |
| 03:05–05:05 | v62 5-fold: cap500_ctx16k 0,52426 (+0,0054, 5/5), main_small e8 0,52553; **K62 0,52662**, KT62 0,52654, KTC62 0,52634. |
| 05:15 | **Submit K62 → public 0,56203** (forecast 0,5605 ± 0,0027). KT62/KTC62 tidak (kembar v32/v59). |
| 05:05–05:45 | SSL S3 pseudo-label tersampel: −0,0022 vs tanpa SSL → null. |
| 05:55–06:05 | Screening 226: **cap300@28k 0,52533** vs cap500@16k 0,51436 vs cap200@40k 0,51831. |
| 06:06–07:03 | v64 cap300_ctx28k s2026 0,52549 (5/5 vs 16k). **K64w 0,52724**, K64 0,52689. Tidak disubmit (kembar K62). |
| 07:39–08:14 | Seed 2027: main_small 0,52644, cap300 0,52387 → **K64 2-seed 0,52740** (OOF tertinggi protokol user). |
| 08:20 | e16 OOM (227). Pool 777 dedup −0,0052 (228). |

## Siang 24 Okt: plateau, literatur, mekanisme
| Waktu | Kejadian |
|---|---|
| 08:31–08:47 | 229: main150 inner +0,0034; wide cap128@48k **OOM**. |
| 08:50–11:02 | v65 main150 s2026/s2027 0,52597/0,52606; **K65 2-seed 0,52568** (null). |
| 11:05–11:19 | 231: **8×e1 berurutan** membuat context penuh muat (cap128@48k puncak 4,43 GiB), tapi KS 0,52224 (tak naik). |
| ~11:05 | **User: submit hanya bila breakthrough / val 0,53–0,54 lebih bagus dari highest.** |
| 11:25 | 232 anchored union −0,0086. Plateau AUC 0,8407–0,8416. |
| ~11:25 | User: "pikirkan approach outstanding". 233 overdispersion (bulan φ 3,54); 234 offset cross-fit bulan +0,0007 (null). Kumo Relational diperiksa: backbone TabICLv2 → tidak dikejar. |
| 12:00 | 235 stacking LR cross-fit −0,0009. |
| ~12:05 | User: "masih 6 hari, cari write-up/paper". Baca kode juara 1 AmEx (DART, series OOF, GRU) & write-up juara 1 Home Credit (KNN target mean). |
| 12:15 | 236 DART f0 −0,0062. Mulai 237 series OOF. |
| ~12:20 | **Hang #3** (boot baru 13:29) saat 237 (LGB 8 thread, tanpa guard). Urutan: #1 23 Okt 23:49 (run 217), #2 24 Okt 00:35, #3 ~12:20, #4 ~18:09. |
| 14:22–14:55 | 237/238 ulang 4 thread + guard 1-sampel: series OOF ecom −0,0003 (f0), cicilan **+0,0055 di f0 tapi −0,0013 di 5-fold**; 239 KNN −0,0040. |
| 14:40–15:00 | 240 **CoLES** (ecom −0,0026, cicilan −0,0007, gabungan −0,0007). |
| 15:20 | 241 simulasi ruin arus kas −0,0004 (KS univariat 0,05–0,08). |
| 15:55–16:00 | 242 GRU-NN: blend dengan LGB +0,0019 (P 0,93), tetapi di atas K64/REF −0,0001/−0,0005 (243). |
| 16:05–17:30 | Seed 2028 (v69): **K64 3-seed 0,52685** < 2-seed. |
| 17:50 | e24 sekuensial: AUC datar (0,84135). |
| 18:05 | 246 berburu 42 agregat baru di pita KS −0,0024. |
| 18:15 | Notebook: banner tim ditambahkan, replay 5 sel lulus, CSV byte-identik v59. |
| 18:30 | 247 target analog pinjaman lama +0,0006 (KS univariat 0,226 tapi sudah tertangkap). |
| 18:50 | 248 bobot kelas null. |
| 18:05–18:09 | 249 seleksi nested fitur teman untuk pohon: f0–f3 rata-rata +0,0002. **Hang #4** ~18:09 dengan guard aktif & suhu normal (boot baru 20:25). |

## Malam 24 Okt – 25 Okt: pipeline teman, submisi, serah terima
| Waktu | Kejadian |
|---|---|
| 23:16 | User tanya val KS tertinggi & public → dijawab (tabel di `03_posisi_sekarang/`). |
| 23:30 | User tanya kenapa kandidat 0,5292 tidak disubmit → ditemukan CSV goal054 kumo_offset (OOF 0,52930), ρ 0,99965 vs submisi teman 0,56393. |
| 23:40 | User: "submit yang belum disubmit & val tinggi, terutama teman". **Submit goal054 → public 0,56470 (rekor tim)**, lalu **K64 2-seed → 0,56168**. Slot #3 ditahan (kandidat sisa kembar). |
| 23:55–00:02 | Tanya-jawab: OOF teman = StratifiedKFold 5 seed 31027; seed model 2026+fold; repo sudah satu tetapi model belum di-merge. |
| 00:03 | User minta rapikan repo + folder serah terima ini. Dihapus: `eda/__pycache__`, 16 file 0 byte, 14 skrip antrean `eda/run_2*.sh` (isi disalin ke `08_kode_dan_infrastruktur/skrip_antrean_dihapus.md`). |
