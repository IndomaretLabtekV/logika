# Temuan utama (dengan angka)

## Tentang data dan validasi
1. **Stratified 5-fold tetap skema yang benar**: train/test interleaved per bulan, customer disjoint, PSI maks 0,0016,
   adversarial AUC 0,4987. Forward CV tidak bisa mensimulasikan deployment (tidak ada label matang sebelum batas bulan).
2. **Satu fold menipu**: sd KS satu fold ≈0,008. Series OOF cicilan +0,0055 di fold 0 berubah jadi −0,0013 di 5-fold.
   Kandidat murah langsung dinilai 5-fold.
3. **Inner-stop 8k sangat bising untuk KS** (±0,01); AUC jauh lebih stabil — gunakan AUC untuk membaca screening inner.
4. Selisih KS ±0,0015 antar seed outer/model adalah noise (seed 2026/2027/2028 member 0,5239–0,5271).

## Tentang Kumo-Tabular
5. **Context lebih besar membantu view lebar**: cap500 8k→16k +0,0094 inner / +0,0054 5-fold; trade kolom→context
   (cap300@28k) +0,0012 lagi. Ini satu-satunya tuas positif yang konsisten di sesi ini.
6. **Memori Kumo**: `fit(num_estimators=E)` menyimpan cache KV per estimator → OOM pada e16/context besar. E fit e1
   berurutan (rata-rata probabilitas) setara e8 dan membuat context penuh 48k muat (cap128 4,43 GiB). Buffer row
   embedding = R × (C+4) × 256 × 2 byte (large, fp16) + overhead ~1,9 GB.
7. **Plateau Kumo**: semua varian pool lebar inner AUC 0,8407–0,8416; seed, estimator (24), impute, dedup, anchored,
   budget main150 tidak menambah.
8. **Seleksi union lewat KS univariat merusak** (nested_small −0,017…−0,023/fold): fitur teman `t__*` banyak varian
   DPD/ecom yang sangat berkorelasi; main-only selection (main_small) sehat.
9. Kumo-only strict ≈ **0,562 public** (K62 0,56203, K64 0,56168) meski OOF ≥ Kumo legacy (0,56056) — offset
   public−OOF Kumo strict +0,0343…+0,0354.

## Tentang transfer ke public
10. Pohon paling bagus transfer; blend tanpa pohon jatuh (v31b 0,55709). Rekor lama v31 (pohon+Causilo+Kumo).
11. **Koreksi offset laten di atas Kumo (pipeline teman) transfer**: +0,0008 OOF → +0,0008 public (0,56393 → 0,56470).
    Kenaikan OOF Kumo-only dari context/seed (+0,0008) **tidak** transfer.

## Tentang fitur dan mekanisme
12. Reference harga strict (fit-only) = legacy (ρ 1,0000) — perbaikan kontrak strict tidak mengubah sinyal.
13. Hampir semua mekanisme generator yang bisa ditebak (volatilitas income, keterjangkauan, premium bunga, delay review,
    target analog, timing bayar) **sudah dikodekan** di 232 fitur main atau pool teman; menambah varian baru
    mengencerkan pohon (−0,0024).
14. Residual OOF overdispersed per bulan (φ 3,5) tapi efek KS kecil (+0,0007) — setara koreksi kalender teman.
15. Arus kas bukan mekanisme default utama (fitur ruin KS univariat 0,05–0,08).
16. Fitur laten reverse engineering teman: gagal sebagai input pohon, berhasil sebagai **koreksi offset** Kumo.

## Tentang teknik literatur
17. DART/series-OOF/KNN target/CoLES menang di data bank riil berfitur mentah anonim; di data sintetis dengan fitur
    buatan tangan kuat semuanya null. GRU-NN menambah keragaman di atas LGB tunggal (+0,0019) tetapi tidak di atas
    blend yang sudah memuat Kumo/Causilo.

## Tentang tim dan proses
18. **Pipeline teman (Kumo V5 + spline + offset) adalah yang terkuat di public** dan belum pernah digabung dengan
    pohon/Causilo user. Ini peluang terbesar yang tersisa.
19. Tiga pipeline independen (strict user, legacy blend, teman) konvergen di OOF 0,527–0,529 / public 0,562–0,565 →
    plafon informasi yang terekstrak; tim top (0,5717–0,5779) kemungkinan punya mekanisme/fitur yang belum kita temukan
    atau kombinasi yang lebih baik.
