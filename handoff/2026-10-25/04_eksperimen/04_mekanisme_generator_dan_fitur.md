# Eksperimen 4 — Berburu mekanisme generator dan fitur baru (24 Okt siang–malam)

Data sintetis → lompatan besar hanya datang dari menebak mekanisme generator (price premium +0,016 public). Semua uji
LGB di bawah: 5-fold paired vs `lgb_base4` 0,52074 lewat `eda/238` (kecuali disebut lain).

## 4.1 Struktur kelompok/waktu (233/234)
- `eda/233_group_overdispersion.py`: dispersi residual OOF K64 2-seed (isotonic) per kelompok. Bulan **φ 3,54
  (p 3e-5)**, minggu 1,66 (p 0,002), pekerjaan×bulan 1,95, rumah×bulan 1,91, purpose×bulan 1,57; region/jam/hari/
  tenor×bunga ≈1; autokorelasi residual lag-1 0,0066 (≈1,7 se).
- `eda/234_crossfit_group_offsets.py`: offset logit ridge cross-fit (isotonic + intercept kelompok difit di 4 fold
  lain): bulan **+0,0007** (P 0,76, 3/5; setara koreksi kalender teman); minggu ±0,0001; rumah×bulan −0,0023;
  region×bulan −0,0022; kombinasi −0,0012…−0,0015 → null praktis. (eda/101 lama juga: efek Mei ada, tak bisa dipakai.)

## 4.2 Simulasi ruin arus kas (241) — hipotesis dari kamus data ("12-month terms")
- `eda/241_cashflow_ruin_sim.py`: per customer (70k punya data akun 8–23 bulan), bootstrap arus kas bersih bulanan
  sendiri 12 bulan × 400 jalur, mulai saldo akhir terakhir, dikurangi cicilan baru; fitur P(≥3 bulan negatif
  berturut), P(kekurangan ≥3 cicilan), P(negatif), kuantil saldo min/cicilan, waktu ke negatif. Tanpa label.
- KS univariat hanya **0,048–0,078** (vs pp_mean 0,37); LGB 5-fold Δ **−0,0004** (P 0,52) → arus kas bukan mekanisme
  utama. Fitur rasio statis keterjangkauan sudah ada (`cf_inst_to_free`, `cf_inst_to_minbal`, `inst_to_avgbal`,
  `dti_total`, `t__v2_income_after_existing_installments`, …).

## 4.3 Berburu 42 agregat baru di pita KS (246)
- `eda/246_error_band_hunt.py`: agregat mentah baru (timing bayar: early_mean/std, paid-same-day, partial/zero/
  overpay share, cure share, DPD 3 cicilan pertama, recency 90 hari; ecom: malam/weekend, harga relatif q90/max/std/
  tren, qty, COD cancel, kartu kredit, entropi kategori, gap; akun: tren saldo, drawdown, nol income; biro: rentang/
  perubahan skor, unscored, inquiry, delinquent). Diukur KS di pita rank OOF 0,50–0,85.
- KS dalam pita tertinggi **0,062** (h_early_mean), lalu timing/recency cicilan. (Spearman vs residual = artefak
  korelasi dengan p untuk y=0 — diabaikan.)
- LGB main232 + 42 kandidat: **0,51840 (−0,0024, P 0,27, kalah 4/5)** → null (dilusi).

## 4.4 Target analog pada pinjaman lama (247)
- `eda/247_analog_target.py`: rekonstruksi target persis (DPD≥90 dalam 12 bulan sejak pencairan, sensor bila jendela
  belum selesai sebelum keputusan) untuk tiap pinjaman sebelumnya → jumlah/rate/hari sejak hit.
- KS univariat **0,226**; 8.197 customer dengan default analog punya target **36,3%** (vs 13,6%). LGB 5-fold
  **+0,0006** (P 0,62, 2/5) → sudah tertangkap fitur repayment (`rp_maxdpd_*`, `rp_n_dpd60`).

## 4.5 Pemeriksaan "sudah ada sebagai fitur" (tanpa run model)
- Volatilitas pendapatan: `ac_income_cv/std/trend/min/l3` + puluhan varian teman → ada.
- Keterjangkauan saldo vs cicilan: ada (lihat 4.2).
- Premium bunga lender: `pv_rate_mean/max/last/delta`, `pv_rej_rate`, statistik teman; grid bunga cuma 4 level → ada.
- Delay aplikasi→keputusan (campuran 3 uniform, proxy review manual): `decision_lag_h`, `t__decision_delay_hours`,
  delay keputusan aplikasi lama → ada.

## 4.6 Bobot kelas (248) dan seleksi fitur teman untuk pohon (249)
- `eda/248_class_weight_screen.py`: scale_pos_weight 2 **+0,0001** (P 0,66), 6,4 **−0,0026** (P 0,11),
  cross-entropy label smoothing 0,02 **−0,0003** (P 0,47) → null.
- `eda/249_nested_teammate_select.py`: per fold, gain LGB (main + 664 `t__*`, inner-fit) → top-30 fitur teman
  ditambahkan. Fold 0–3: +0,0015/−0,0023/+0,0028/−0,0014 (≈ +0,0002) → null praktis (fold 4 tak selesai, hang #4).

## Catatan penting untuk pipeline teman
- Fitur laten reverse engineering teman (posterior harga/skor biro, rata-rata geometrik pendapatan, posterior
  keterlambatan) **gagal sebagai input pohon** (laporan reverse engineering teman: −0,0004…−0,0080), tetapi **berhasil
  sebagai koreksi offset kecil di atas logit Kumo** (goal054: +0,0008 OOF → +0,0008 public). Ini belum dicoba pada
  Kumo/pohon user — lihat `07_langkah_berikutnya/rencana_merge.md`.
