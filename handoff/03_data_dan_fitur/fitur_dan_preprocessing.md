# Fitur (`eda/features.py`) dan preprocessing

Semua agregasi difilter terhadap **cutoff milik customer itu sendiri** (`decision_datetime` aplikasi terpilih):
- bureau: `available_datetime` ≤ cutoff;
- account: bulan + 1 bulan ≤ cutoff (bulannya sudah lengkap);
- repayment: `due_date` ≤ cutoff;
- e-commerce: `transaction_datetime` ≤ cutoff;
- previous: `decision_datetime` ≤ cutoff.

## Keluarga fitur
| Versi | Prefix | Isi |
|---|---|---|
| v1 | (raw app) | loan_amount, rate, installment, purpose, profil, `decision_lag_h`, `app_hour`, `app_dow`, `dti_declared`, `lti_declared`, `tenure_over_age`, `income_per_head` |
| v1 | `bc_*` | pull saat ini: skor mean/min/max/std, active accounts, outstanding, limit/saldo revolving, utilisasi, inquiry, delinquency, staleness, delay |
| v1 | `bp_*` | pivot per provider: `bp_{score,util,inq,delinq,out}_{a,b,c}` |
| v1 | `bh_*` | histori pull: n_pulls, skor mean/min/last, delinquency maks, inquiry mean, pull 180 hari, delta skor, delta outstanding |
| v1 | `rp_*` | statistik maxdpd, count/fraksi dpd ≥1/30/60/90, dpd saat ini, unpaid, rasio bayar, hari telat (total / 6m / 12m / 3 terakhir), hari sejak delinquency, pinjaman aktif, cicilan aktif, sisa kewajiban |
| v1 | `ac_*` | rata-rata, agregat 3 bulan terakhir, baseline bulan 7–12, delta (out_in, lowdays, bal), slope, CV |
| v1 | `ec_*` | order dedup per `order_id`; jumlah order di 7/14/30/60/90 hari; `ec_base_wk` (laju 120–365 hari); spend 30/90 hari/total; AOV; recency; tenure; share metode bayar; share kategori; `ec_accel_30d/90d` = laju terbaru ÷ (base + 0,1) |
| v1 | `pv_*` | jumlah per keputusan, rejection rate, recency, aplikasi 90/180 hari, rejection 365 hari, rate mean/max/last/delta, approval ratio, nominal |
| v1 | cross | `has_history`, `dti_verified`, `dti_total`, `income_verif_ratio`, `debt_to_income`, `inst_to_avgbal`, `ecspend_to_income` |
| v2 (38) | `bz_` | tren skor bureau, range, skor terbaru, delinquency semua pull, tren active accounts, `bz_inq_external` (inquiry dikurangi aplikasi sendiri 6 bulan) |
| v2 | `ob_` | rata-rata tenor approved, hari sejak approval, nominal approved 90 hari, sisa kewajiban, rasio amount baru vs approved terakhir |
| v2 | `rq_` / `sv_` | cicilan pertama telat, pinjaman terakhir (fraksi telat, maxdpd), fraksi 1–29 dpd, tren telat, streak telat maks; keparahan (rata-rata maxdpd saat telat, hari telat, share ≥30 di antara yang telat, unpaid di antara yang telat) |
| v2 | `cf_` | share inflow non-income (total dan 3 bulan), outflow 3 bulan, free cash flow minimum, saldo minimum 3 bulan, cicilan baru vs free cash / saldo minimum, out/income 3 bulan |
| v2 | `ec2_` | cancel 28 hari, share order malam 90 hari, item per order 28 hari, pergeseran AOV, gap tersempit 28 hari, EWM halus 7/28/56 hari relatif terhadap baseline |
| v5 (19) | `pp_` | z = (log(item_total/qty) − μ_kategori) / σ_kategori. Agregat per customer: mean, trim 10–90, done (completed saja), min, max, 30 hari, 90 hari, old (>182 hari), qty per item (total dan 90 hari), dan mean per kategori (7). Ditambah `dec_month` |

Total **232 prediktor** (4 kategorikal: loan_purpose, region, employment_type, housing_status). `pp_median` dibuang karena redundan.

## Referensi harga (perbaikan penting di v6)
- **v5:** μ/σ per kategori dihitung dari seluruh item semua customer di semua tanggal. Astra menunjukkan ini memakai transaksi customer lain yang terjadi *setelah* cutoff aplikasi awal, jadi melanggar semangat aturan cutoff (bukan kebocoran target).
- **v6+:** referensi **expanding per bulan keputusan**, yaitu hanya item (semua customer, tanpa label) yang tanggalnya **sebelum hari pertama bulan keputusan**. KS pp_mean tetap 0,3689 (global 0,3684), Spearman vs global 0,9992. Diverifikasi di `eda/50` dan `eda/53`: tes injeksi 2 bagian dan independent recompute.
- Astra juga menguji versi fold-local (hanya customer training): korelasi rank 0,99999838. Perbedaannya tidak material.

## Preprocessing per model
| Model | Input |
|---|---|
| Tree | matriks raw. NaN dibiarkan; kategori sebagai pandas category (LightGBM/XGB) atau string (CatBoost) |
| TabM | `nn_prep` **fit di dalam fold** (sejak v6): ECDF mid-rank → normal, clip ±5, NaN → 0, plus satu indikator per pola missing yang unik (sekitar 38 grup). Hasilnya 266 kolom. Kategori di-embed. Verifikasi `eda/32`: rank-preserving, KS univariat tidak berubah |
| TabPFN / Causilo | 100 fitur numerik teratas (seleksi per fold berdasarkan KS, dengan filter greedy \|rho\| < 0,95) + 4 kategori. Causilo pakai pandas categorical (v8+) |
| RealMLP | median training-only + flag missing per fitur + vocabulary kategori training-only (rekomendasi Astra v7) |

## Verifikasi preprocessing yang sudah dilakukan
- `eda/12`: recompute independen dengan pandas pada 300 customer menghasilkan match 100%. Tes injeksi baris masa depan: fitur tidak bergeser.
- `eda/27` (v2), `eda/46` (v5), dan `eda/53` (v6): recompute 300 customer cocok 100%. Injeksi masa depan, termasuk item 1e9 di groceries yang bisa menggeser referensi, tidak menggeser satu pun dari 231 fitur.
- Di notebook: leak test 2 bagian berjalan di setiap run. Part 1: tidak ada fitur non-harga yang bergeser (230 dicek). Part 2: baris setelah cutoff Desember tidak menggeser fitur apa pun, termasuk pp.
- `eda/32`: nn_prep mempertahankan KS univariat.
- Astra (`eda/deep`): ECDF interpolated + Gaussian mengubah KS observed sebesar 0,000; robust scaling + asinh mengubah sampai 0,033, jadi **jangan dipakai**. Sampel kedua kelas dan kedua grup history disimpan di `eda/deep/out/`.

## Hal yang sengaja **tidak** dilakukan
- Augmentasi gaya CV (rotation, jitter, SMOTE): tidak ada dasar semantik untuk fitur keuangan.
- Menghapus item dengan konten kembar: lihat `leakage_dan_duplikasi.md`.
- Pembulatan float `pv_rate_delta` (usulan Astra): tidak diadopsi dan belum diuji di model.

## Konstruksi fitur 18–19 Okt (TIDAK ada yang diadopsi ke `features.py`; parquet di `eda/out/`, regenerable)
| Script | Kolom | Hasil terbaik |
|---|---|---|
| `eda/114_cross_timing.py` | A_/B_/C_/D_ (11) | null |
| `eda/115_repay_latent.py` | re_* (EB disiplin bayar) | LGB −0,0003, Causilo −0,0014 |
| `eda/125` thin-view (drop 113 kolom kosong thin + has_history) | model, bukan kolom | keluarga +0,0002 |
| `eda/135_prev_escalation.py` | E_* (9) | null |
| `eda/136_affordability.py` | F_* (8) | null |
| `eda/137_peer_relative.py` | x\|z\|grup, x\|r\|grup (3.150) | LGB_s +0,0023 (seleksi bocor), keluarga +0,0002 |
| `eda/138_month_normalized_account.py` | M_* (15, patuh cutoff) | lemah |
| `eda/142_snapshot_deltas.py --shifts` | d90_/d180_ (333), d30_/d365_ | keluarga −0,0002 / −0,0009 |
| `eda/144_misc_constructs.py` | G_* (15) | null |
| `eda/145` / `146` pasangan | pw_* | bersarang −0,0005 |
Snapshot fitur: `data_pq/features_shift{30,90,180,365}.parquet` (di-.gitignore, dibangun ulang oleh `eda/142`).
