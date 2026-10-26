# Jurnal 24 Oktober 2026 — resep acuan lokal, context, SSL

Arahan user (/goal 24 Okt 00:3x): jalankan resep acuan 5-fold lokal sebagai baseline baru, lalu
context-optimization yang belum diuji, SSL nested, ranking 3 kandidat, submit ≤3 sebelum 07:00 WIB.

## Kronologi

- 00:32 audit awal: tidak ada submisi sejak v59 (21 Okt), 3 slot tersedia sampai 07:00.
- 00:34–00:35 build pool outer f0..f2 (`eda/219_build_outer_pools.py`, CPU saja, ~30 dtk/fold).
  **Laptop hang** saat f2 (force shutdown oleh user). Tidak ada job GPU yang berjalan;
  journal boot −1 berhenti 00:32:06 tanpa error kernel. Ini melemahkan hipotesis "hang =
  termal/VRAM dari job GPU" (run 217). Penyebab belum terbukti.
- f2 meninggalkan `manifest.json` 0 byte → dikarantina ke
  `results/v60/strict_tfm/quarantine_crash_2026-10-24/`. f0/f1 lolos verifikasi hash; f2–f4
  dibangun ulang. 219 sekarang `os.sync()` setelah tiap pool; 212 menulis cache dengan fsync+rename.
- 00:41 mulai `eda/run_220_recipe.sh` (member-major, lihat
  `artifacts/optimization/continuation_24okt/amendment_01.json`, dikunci sebelum skor).
- Watcher `eda/thermal_guard.py`: SIGSTOP bila GPU >85 °C atau CPU >95 °C selama 3 sampel
  (15 dtk). Satu core P (i7-12700H) mencapai 97–101 °C dalam ~15 dtk setiap kali GPU aktif dan
  turun ke 60–80 °C dalam 5 dtk saat pause → duty cycle ~75%. Telemetri `eda/out/220_telemetry.csv`.
- Desain tugas 2/3 ditulis sebelum skor: `docs/integration/protokol_context_ssl_24okt.md`
  (C2 context maksimum view lebar; S3 pseudo-label tersampel + kontrol permutasi).
- 01:33 **member `pool_cap500_s2026` lengkap 5 fold: pooled KS 0,51885**; fold 0,5264 / 0,5291 /
  0,5142 / 0,5014 / 0,5448. Kumo legacy v28m (fm_select 104, context 56k, e4) pooled 0,52665;
  cap500 kalah 4/5 fold. Evaluator `eda/221` mereproduksi REF v59-analogue 0,52625 (tercatat 0,52622).
  Interpretasi: view lebar dengan context 8.192 kehilangan sinyal; mendukung motivasi C2.
- 01:34 urutan antrean diubah (`eda/run_220b_queue.sh`): nested_small s2026 → screening C2 → S3 →
  uncapped s2026 → seed 2027. Kandidat terkunci tidak berubah. Insiden kecil: pgrep ikut menangkap
  shell tool, antrean baru sempat memulai fit f1 paralel ±30 dtk (dibunuh, tidak ada cache tertulis).
  Guard sekarang memakai pola ter-anchor ke proses python job.
- 01:44 `nested_small_s2026` f0: KS 0,50884 (legacy f0 0,52556, −0,0167); fit 56k context e8 hanya
  ~10,5 menit. 104 kolom terpilih didominasi fitur teman `t__*`.
- `eda/223_strict_vs_legacy_features.py` (outer-train f0 saja): 17 fitur `pp_*` strict vs legacy
  Spearman 1,0000, ΔKS univariat rata-rata −0,00001 → reference harga strict **bukan** penyebab.
  Antrean v3 (`run_220c_queue.sh`) menyisipkan member `baseline` protokol f0 (main→100, e4) untuk
  memisahkan efek fitur teman/selector.
- 02:28 `nested_small_s2026` lengkap: pooled **0,50936** (fold 0,5088/0,5146/0,5001/0,5086/0,5292 vs legacy
  0,5256/0,5380/0,5197/0,5151/0,5507; kalah 5/5). Kandidat terkunci dengan 2 member (s2026):
  K 0,51960 (AUC 0,84338), KT 0,52328, KTC 0,52311; REF v59-analog 0,52625. Δ KT−REF −0,00297
  (P>0 0,025), KTC−REF −0,00314 (P>0 0,013). Semua < 0,53 dan < REF.
- Ditambahkan arm median-impute (M_*) dan C2 16.384 pada screening (amendemen 02:10 di protokol).
- 02:38 member `baseline` protokol (strict main232→100 via select_columns, kategori native, context
  56k, e4) f0: **KS 0,52629** vs legacy v28m f0 0,52556 → pipeline strict setara legacy. Penurunan
  nested_small (0,50884 di f0) berasal dari 104 kolom yang dipilih dari pool union 896 (didominasi
  fitur teman `t__*`), bukan dari reference/encoding.
- 02:38 bug 222: manifest dibandingkan (tuple vs list JSON) → arm M_ctx8192 gagal "changed run".
  Diperbaiki dengan normalisasi JSON; manifest lama disimpan sebagai
  `manifest_v1_before_compare_fix.json` (prediksi C2_ctx8192 dibuat oleh versi sebelum perbaikan
  baris pembanding; logika inferensi identik). C2_ctx8192 inner-stop KS 0,50494 / AUC 0,83243.
- 02:34–03:03 screening inner f0 (`eda/222`, 8k inner-stop, s2026, e8; `results/v61/context_ssl_screen_f0`):

  | Arm | KS | AUC |
  |---|---:|---:|
  | C2_ctx8192 (cap500, NaN mentah) | 0,50494 | 0,83243 |
  | **C2_ctx16384** | **0,51436** | **0,83875** |
  | M_ctx16384 (median impute) | 0,51448 | 0,83877 |
  | S3_base_teacher (nested_small union) | 0,50047 | 0,83133 |
  | M_small (median impute) | 0,49676 | 0,83069 |

  Context 8k→16k: +0,0094 KS. Median impute: +0,0001 (lebar) / −0,0037 (kecil) → tidak dipakai.
  24.576 tidak dicoba (estimasi buffer > VRAM).
- 03:05 antrean v5 (`run_220e_queue.sh`): protocol `protocol_v62_ctx16k.json` (ditulis sebelum skor
  outer varian): `cap500_ctx16k` (context 16.384, fraction 0,85) dan `main_small` (seleksi main
  saja, context penuh, e8), s2026, 5 fold via `eda/224_strict_tfm_variant.py`. S3 pseudo ditunda.
- 04:06 **`cap500_ctx16k_s2026` (v62) lengkap: pooled 0,52426** (AUC 0,84591); fold 0,5307/0,5292/
  0,5187/0,5080/0,5464 vs cap500 8k 0,5264/0,5291/0,5142/0,5014/0,5448 (menang 5/5, +0,0054 pooled).
  Masih −0,0024 dari Kumo legacy. Dengan member ini saja: KT 0,52563, KTC 0,52696 (vs REF +0,0007,
  P>0 0,687, 3/5 fold). Konfirmasi outer bahwa context lebih besar membantu view lebar.
- 05:05 **kandidat v62 terkunci (5-fold, s2026)**: K62 0,52662 (AUC 0,84757; fold 0,5308/0,5326/0,5198/
  0,5119/0,5521), KT62 0,52654, KTC62 0,52634; REF 0,52625; member main_small e8 0,52553.
  Δ vs REF +0,0004/+0,0003/+0,0001 (P>0 0,74/0,61/0,65). Tidak ada yang ≥0,53.
- Kemiripan rank: K62 vs v31c 0,99721; KT62 vs v32 0,99843; KTC62 vs v59 0,99965.
- 05:15 **submit K62** (ref 57573242, `results/v63/k62_kumo_ctx16k/`), alasan: satu-satunya kandidat yang
  jelas berbeda (Kumo-only, pool/context baru). Forecast 0,5605 ± 0,0027. **Public 0,56203** (z +0,57);
  vs Kumo-only legacy v31c 0,56056 → +0,0015 pada OOF setara (offset +0,0354 vs +0,0339). Bukti lemah
  (~0,7 sd). KT62/KTC62 tidak disubmit (tidak jelas berbeda; tidak memilih berdasarkan LB).
- 05:05–05:45 screening S3 (pseudo-label tersampel Bernoulli(p_teacher), 30k test sebagai context
  tambahan, nested_small e8): d0 0,49885, d1 0,49405, rata-rata rank 0,49822; baseline 0,50047;
  permutasi 0,49103. Δ pseudo−base **−0,0022** (CI −0,0093…+0,0022, P>0 0,13); pseudo−perm +0,0072
  (P>0 0,93). **Null** (kriteria student > baseline gagal); tidak diteruskan. Bersama v49 (threshold),
  pseudo-context Kumo sudah gagal dalam dua bentuk.
- Bootstrap screening: C2 16k−8k +0,0094 (CI −0,0021…+0,0240, P>0 0,95); M8k−C2 8k −0,0021 (P 0,35);
  M16k−C2 16k +0,0001 (P 0,44); M_small−base −0,0037 (P 0,25).
- Ide baru 05:25 (`eda/226_width_context_tradeoff.py`): tukar lebar vs context pada VRAM tetap
  (cap300@28.672, cap200@40.960) vs cap500@16.384. Lalu seed 2027 untuk member v62.
- 05:55–06:05 screening 226 (inner f0, s2026, e8): **cap300@28.672 KS 0,52533 / AUC 0,84142**,
  cap200@40.960 0,51831 / 0,84164, cap500@16.384 0,51436 / 0,83875. Context besar dengan subset kolom
  acak per estimator mengalahkan lebar 500 (+0,011 KS untuk cap300). Puncak KS di cap300.
- 06:06 antrean v7 (`run_220g_queue.sh`): protocol `protocol_v64_cap300_ctx28k.json` (ditulis
  sebelum skor outer view ini; kandidat K64/K64w/KT64 terkunci) → cap300_ctx28k s2026 5-fold, lalu
  main_small s2027, cap300_ctx28k s2027. Fit main_small s2027 f0 yang baru 9 dtk dihentikan.
- 07:03 **`cap300_ctx28k_s2026` (v64) lengkap: pooled 0,52549** (fold 0,5320/0,5317/0,5215/0,5116/0,5513;
  menang 5/5 atas cap500@16k). Kandidat terkunci: **K64w 0,52724** (AUC 0,84789; Δ REF +0,0010,
  P>0 0,75, 4/5), K64 0,52689 (mean fold 0,53135), KT64 0,52564.
- Ranking OOF kandidat terkunci saat ini: K64w 0,52724 > K64 0,52689 > K62 0,52662 (> KT62 0,52654).
- Tidak disubmit: K64w/K64 vs K62 Spearman 0,99952/0,99934 (tidak jelas berbeda, ΔOOF +0,0006);
  KT64 strukturnya = v32/KT62. Antrean lanjut seed 2027 (main_small, cap300_ctx28k), ~10:30.
- 07:39 `main_small_s2027` (v62) lengkap: pooled 0,52644 (fold 0,5290/0,5379/0,5179/0,5139/0,5522);
  s2026 0,52553. Rata-rata rank 2 seed 0,52587 (mean fold 0,53026). Selisih antar-seed 0,0009 ≈ noise
  seed historis (~0,0015). Lanjut cap300_ctx28k s2027.
- 08:14 `cap300_ctx28k_s2027` lengkap: pooled 0,52387 (fold 0,5234/0,5295/0,5202/0,5078/0,5508; s2026
  0,52549). **K64 2-seed (cap300×2 + main_small×2) 0,52740** (AUC 0,84797; Δ REF +0,0011, P>0 0,67,
  4/5) — OOF tertinggi hari ini, masih <0,53. K64w 2-seed (+ctx16k) 0,52677; KT 0,52507; KTC 0,52544.
- Berikutnya: screening inner e16 vs e8 untuk cap300@28k (subset kolom acak per estimator → cakupan).
- 08:20 screening `eda/227` e16 untuk cap300@28k: **OOM** saat fit (minta 3,04 GiB; 4,80 GiB terpakai,
  batas 6,53 GiB pada fraction 0,85). Kegagalan engineering, bukan null kualitas; tidak diulang
  dengan pengurangan context/kolom. GPU idle sesudahnya.
- 08:35 screening `eda/228` cap300@28k pada pool 777 (dedup+affine, difit 48k inner-fit): KS 0,52012 vs
  pool 896 0,52533 (Δ −0,0052, CI −0,0111…+0,0045, P>0 0,18); AUC 0,84150 vs 0,84142. **Null** — duplikat di pool 896
  tidak merugikan subset acak per estimator. GPU idle sesudahnya.
- 08:31–08:47 screening `eda/229` (inner f0, s2026, e8, context penuh 48k): main100 0,51750 / AUC 0,84086;
  **main150 0,52092 / 0,84160** (Δ +0,0034, CI −0,0090…+0,0114, P>0 0,63); wide cap128 @48k **OOM**
  (alokasi 2,55 GiB di atas 4,89 GiB) → context penuh untuk pool lebar tidak muat di 8 GB.
- 08:50 protocol `protocol_v65_main150.json` dikunci; main150 s2026 lalu s2027 5-fold via 224.
- 09:53 `main150_s2026` (v65) lengkap: pooled 0,52597 (fold 0,5252/0,5354/0,5183/0,5131/0,5506) vs
  main100 0,52553 (+0,0004). K65_s2026 (cap300 + main150) 0,52653 (AUC 0,84848; Δ REF +0,0003, P>0
  0,59) < K64_s2026 0,52689. Budget 150: null. s2027 tetap dijalankan agar K65 lengkap.
- 11:02 `main150_s2027` lengkap: pooled 0,52606. **K65 2-seed 0,52568** (AUC 0,8484; Δ REF −0,0006,
  P>0 0,43) < K64 2-seed 0,52740. Budget main 150: null. GPU idle, guard dimatikan.

## Posisi 24 Okt 11:05

| Kandidat (OOF pooled 70k, terkunci) | KS | AUC | Δ REF (P>0) |
|---|---:|---:|---:|
| K64 2-seed (cap300@28k×2 + main_small×2) | **0,52740** | 0,84797 | +0,0011 (0,67) |
| K64w (cap300@28k + cap500@16k + main_small, s2026) | 0,52724 | 0,84789 | +0,0010 (0,75) |
| K62 (cap500@16k + main_small, s2026) — submit, public 0,56203 | 0,52662 | 0,84757 | +0,0004 (0,74) |

Belum ada kandidat ≥0,53. Satu-satunya tuas yang konsisten positif adalah ukuran context; di 8 GB
batasnya tercapai (pool lebar: 500 kolom ≤16k, 300 ≤28k; 128 kolom @48k dan e16 OOM). Jalan teknis
tersisa (chunking row-embedding context di dalam sdm) melanggar kontrak "tanpa ubah source paket"
dan butuh keputusan user. Notebook builder tidak diubah (tidak ada resep yang mengalahkan v31).
- 11:05–11:17 `eda/231` (inner f0): E fit e1 berurutan, probabilitas dirata-rata (puncak = satu estimator).
  cap300@28k 8×e1 KS 0,52102 / AUC 0,84152 vs e8 0,52533 / 0,84142 (Δ −0,0043, CI −0,0128…+0,0078,
  P>0 0,26 → setara). **Context penuh 48k untuk pool lebar kini muat** (cap128, puncak 4,43 GiB; e8
  OOM karena cache per-estimator menumpuk), tetapi KS 0,52224 / AUC 0,84074 — tidak lebih baik
  (Δ vs cap300@28k e8 −0,0031, CI −0,0139…+0,0098, P>0 0,34).
  Kurva lebar×context inner: 500@16k 0,5144; 300@28k 0,5253; 200@40k 0,5183; 128@48k 0,5222 → plateau.
- 11:20 user: kuota hari ini baru; submit hanya bila breakthrough / OOF 0,53–0,54 dan lebih baik dari
  tertinggi. Belum terpenuhi (terbaik K64 2-seed 0,52740) → 0 submit.
- 11:25 `eda/232` anchored union (104 main_small tetap + 196 kolom union acak per member, ctx 28k,
  8×e1): KS 0,51671 / AUC 0,84146 vs cap300@28k e8 0,52533 / 0,84142 → null.
- Ringkasan inner f0 pool lebar: AUC 0,8407–0,8416 untuk cap300@28k (e8/8×e1), cap200@40k, cap128@48k,
  anchored; selisih KS ±0,005 = noise 8k baris. Feeding Kumo plateau; bukan lagi kendala VRAM.
- 11:40 `eda/233` residual overdispersion K64 2-seed: bulan φ 3,54 (p 3e-5), minggu 1,66 (p 0,002),
  pekerjaan×bulan 1,95, rumah×bulan 1,91; region/jam/hari/tenor×bunga ≈1; autokorelasi lag-1 0,0066.
  `eda/234` offset logit ridge cross-fit: bulan +0,0007 (P 0,76, 3/5) — setara koreksi kalender teman;
  minggu ±0,0001; rumah×bulan −0,0023; region×bulan −0,0022; kombinasi −0,0012…−0,0015. Null praktis.
- 12:00 `eda/235` stacking LR cross-fit atas 9 member Kumo strict (+pohon6, Causilo): KS 0,52596–0,52652
  (Δ vs K64 2-seed −0,0009…−0,0014, P>0 0,38–0,52); bobot pohon ~0,04. Null. Caveat: bukan nested end-to-end.
- 12:05 user: 6 hari tersisa, lanjut eksperimen; cari write-up/paper yang bisa ditiru.
- 12:15 literatur: kode juara 1 AmEx (jxzly) → LGB DART ff0,05 + "series OOF" (model level baris riwayat
  dengan label customer, GroupKFold, agregasi OOF per customer) + GRU; juara 1 Home Credit → fitur rata-rata
  target 500 tetangga (EXT_SOURCE + rasio). `eda/236` DART f0: KS 0,51276 vs LGB v22 0,51892 (−0,0062) → null.
  Antrean CPU: 237 series OOF (ecom, repay) dan 239 KNN target mean, dievaluasi 238 (paired vs lgb_base f0).
- ~12:20 **hang ke-3** (force restart user, boot baru 13:29) saat 237 ecom f0 berjalan (LightGBM 8 thread,
  4,36 juta baris, **tanpa guard**). Ketiga hang hari ini terjadi di job CPU multithread berat (pool polars,
  217, 237); telemetri pagi mencatat CPU 104 °C. Dugaan kuat: beban CPU/termal tanpa throttle, bukan GPU.
  Mitigasi: job CPU 4 thread + guard termal wajib (pola 23[789]). 14:22 antrean 237/238/239 diulang.
- 14:26 guard diperketat: pause pada satu sampel CPU >95 °C (aturan 3-sampel membiarkan CPU >95 °C 73%
  waktu, maks 103 °C).
- 14:30 screening fold 0 (LGB v22, 232 fitur main + fitur baru; paired vs base 0,51892):
  series OOF **ecom 0,51860 (−0,0003)**; series OOF **cicilan 0,52441 (+0,0055)**; KNN target mean
  (top-5 univariat = varian pp + rp_frac_dpd1) 0,51495 (−0,0040). Cicilan diteruskan ke 5-fold (`run_237b.sh`),
  base dihitung ulang dengan thread sama (lgb_base4). Audit bocor: baris cicilan due < keputusan, DPD snapshot
  cutoff (nol pembayaran sesudah cutoff), model baris outer-val hanya dari customer outer-train, GroupKFold.
- 14:55 konfirmasi 5-fold series OOF cicilan: base4 0,52074 vs so_repay 0,51940 (Δ −0,0013, CI −0,0040…+0,0027,
  P>0 0,32); per fold +0,0055/−0,0046/−0,0003/+0,0003/−0,0027 → **null**; fold 0 kebetulan (sd 1 fold ~0,008).
  Pelajaran: kandidat murah (LGB) langsung dinilai 5-fold, bukan fold 0 saja.
- 14:40–15:00 **CoLES** (Babaev dkk. SIGMOD'22; GRU + supervised-contrastive, 5 slice acak/customer, tanpa label,
  100k customer; `eda/240`) → PCA-32 → LGB v22 5-fold paired vs base4 0,52074:
  ecom 0,51817 (Δ −0,0026, P>0 0,13, kalah 5/5); cicilan 0,52009 (Δ −0,0007, P>0 0,43); gabungan 0,52002. Null.
  Paper melaporkan gain terbesar pada data bank riil; di data sintetis ini fitur buatan tangan sudah menyerap sinyal.
- 15:20 kamus data: "selected loans have 12-month terms" → hipotesis generator mensimulasikan arus kas (default =
  saldo tak menutup cicilan ≥3 bulan). Fitur rasio statis (cf_inst_to_free, inst_to_avgbal, dti_total) sudah ada;
  `eda/241` simulasi ruin struktural (bootstrap arus kas bersih bulanan sendiri, 12 bln, 400 jalur, tanpa label):
  KS univariat 0,048–0,078 (vs pp_mean 0,37); LGB 5-fold Δ −0,0004 (P>0 0,52) → null. Arus kas bukan mekanisme utama.
- 15:55 `eda/242` GRU-NN ala AmEx #1 (GRU urutan ecom+cicilan + MLP 232 fitur, early stop inner 1/7, tanpa refit):
  NN sendiri pooled 0,51409 / AUC 0,8377 (fold 0,5111/0,5156/0,5165/0,5121/0,5376). Blend terkunci 0,75 LGB + 0,25 NN:
  **0,52266 vs LGB 0,52074 (+0,0019, CI −0,0007…+0,0043, P>0 0,93, 3/5)**. Kandidat dikunci sebelum dihitung:
  B1 = 0,8·rank(REF v59-analog) + 0,2·rank(NN); B2 = 0,8·rank(K64 2-seed) + 0,2·rank(NN).
- 16:00 `eda/243`: B1 (REF + 0,2 NN) 0,52577 vs REF 0,52625 (Δ −0,0005, P>0 0,49); B2 (K64 2-seed + 0,2 NN) 0,52730 vs
  K64 0,52740 (Δ −0,0001, P>0 0,67). GRU-NN menambah di atas LGB tunggal, **tidak** di atas blend yang sudah memuat
  Kumo/Causilo → keragamannya sudah tertangkap. Null untuk tujuan kita.
- 17:30 seed 2028 (protocol v69, dikunci sebelum skor; motivasi: tim 0,57065 memakai Kumo 3 seed): main_small_s2028 0,52668,
  cap300_ctx28k_s2028 0,52713. **K64 3-seed 0,52685** (AUC 0,84811; Δ REF +0,0006, P>0 0,78) < K64 2-seed 0,52740.
  Seed tambahan = noise (member 0,5239–0,5271); tidak menembus 0,53.
- 17:50 cap300@28k dengan 24 estimator sekuensial (e1×24, dulu e16 OOM; `eda/231` arm 24x1, member 0–7 = arm 8x1):
  inner KS 0,5174 / AUC 0,84135 vs 8×e1 0,52102 / 0,84152 vs e8 0,52533 / 0,84142 → null; AUC datar, estimator
  tambahan tidak menambah cakupan yang berguna.
- 18:05 `eda/246` berburu mekanisme di pita KS (rank OOF 0,50–0,85): 42 agregat mentah baru (timing bayar, partial,
  cure, harga relatif q90/max/tren, malam/weekend, drawdown saldo, dispersi skor biro, inquiry). KS dalam pita tertinggi
  0,062 (h_early_mean), lalu timing/recency cicilan. (Spearman vs residual = artefak korelasi dengan p, diabaikan.)
  LGB 5-fold main232 + 42 kandidat: 0,51840 vs base4 0,52074 (Δ −0,0024, P>0 0,27, kalah 4/5) → null.
- 18:15 notebook: banner tim ditambahkan di `notebook/build_final.py` ("DSCxxxx — lagi cape lomba asli", anggota akun
  Kaggle; nomor peserta masih placeholder). Build ulang 14 sel; replay nbclient 5 sel kode lulus, CSV byte-identik dengan
  submisi v59 (SHA d7e17b07…). Bukti: `artifacts/optimization/continuation_24okt/notebook_verification_24okt.json`.
  Fresh training tidak diulang. Sisa: nomor peserta + nama file `ParticipantNumber_TeamName_Notebook`.
- 18:30 `eda/247` target analog pada pinjaman sebelumnya (DPD≥90 dalam 12 bln sejak pencairan, sensor bila jendela
  belum selesai): KS univariat 0,226; 8.197 customer dengan default analog punya target 36,3% (vs 13,6%). LGB 5-fold
  +0,0006 (P>0 0,62, 2/5) → null; sudah tertangkap fitur repayment yang ada (rp_maxdpd_*, rp_n_dpd60).
- 18:50 `eda/248` bobot kelas LGB (5-fold, paired base4 0,52074): spw2 +0,0001 (P 0,66), spw6,4 −0,0026 (P 0,11),
  cross-entropy label-smoothing 0,02 −0,0003 (P 0,47) → null.
- 18:05–18:09 `eda/249` LGB main232 + 30 fitur teman dipilih nested via gain (inner-fit): fold 0–3 Δ vs base4
  +0,0015/−0,0023/+0,0028/−0,0014 (rata-rata ≈ +0,0002) → null praktis; fold 4 tidak selesai.
- ~18:09 **hang ke-4** (boot baru 20:25) saat 249 berjalan **dengan** guard 1-sampel >95 °C dan 4 thread → hipotesis
  "termal tanpa throttle" melemah; penyebab hang belum diketahui. `oof_lgb_tsel30.parquet` 0 byte (v.run menulis non-atomik).
- 23:16 user bertanya val KS tertinggi & public (dijawab, tanpa run baru).
- 23:30 user bertanya kenapa kandidat 0,5292 tidak disubmit. Kandidat tertinggi teman: `results/teman/
  submission_goal054_kumo_offset_l20p003_unweighted_d0p5_w1.csv` (OOF 0,52930, seed fold 31027, SHA 5d4d8b81…).
  Format lolos; Spearman vs Kumo+spline teman (public 0,56393) **0,99965**, vs v31 0,98670, vs K62 0,98924.
  Perkiraan public 0,5643 ± 0,0015 (≈ submisi 0,56393 yang sudah ada). Tidak lolos aturan submit (tidak jelas berbeda;
  <0,53, bukan breakthrough) → tidak diupload tanpa perintah eksplisit user. Siap disubmit bila user memutuskan.
- 23:45 user: "coba submit dulu aja deh yang belum di submit dan emang dapet val tinggi, terutama dari temen gua".
  **Submit #1 goal054 kumo_offset** (ref 57600249, `results/v70/goal054_kumo_offset/`), forecast 0,5643 ± 0,0015 →
  **public 0,56470 = rekor tim baru** (+0,00065 vs v31 0,56405; +0,00077 vs kembarannya 0,56393; z +0,27).
- 23:55 **Submit #2 K64 2-seed** (ref 57600277), forecast 0,5628 ± 0,0015 → **public 0,56168** (z −0,75; K62 0,56203).
  Gain OOF +0,0008 vs K62 tidak transfer; Kumo-only strict ≈ 0,562 public. Slot #3 ditahan: kandidat val tertinggi
  yang tersisa (K64w 0,52724, K64 3-seed 0,52685) kembar rank dengan K64 (ρ ≥0,999) → informasi ~nol.
