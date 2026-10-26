# Kompetisi DSC LOGIKA UI 2026

## Tugas dan metrik
- Prediksi probabilitas pinjaman baru mencapai **end-of-day DPD ≥ 90 dalam 12 bulan kalender** sejak keputusan.
  Kamus data: "selected loans have 12-month terms". Data lending konsumen Indonesia **sintetis**.
- **Metrik: KS two-sample** `ks_2samp(pred[y==1], pred[y==0]).statistic` — hanya urutan (rank) yang berpengaruh.
  Helper identik: `eda/common.py::ks`, `eda/valid2.py::ks`.
- `decision_datetime` = cutoff informasi; fitur hanya boleh memakai data sebelum cutoff.

## Data (folder `data/`, 8 CSV + kamus `data/DATA_DICTIONARY.md`)
| Tabel | Baris | Catatan |
|---|---:|---|
| train_applications | 70.000 | target, 9.507 positif (13,58%) |
| test_applications | 30.000 | tanpa target; urutan berbeda dari sample_submission → **join by ID** |
| customer_profiles | 100.000 | umur, region (5), pekerjaan (3), tenure, pendapatan deklarasi, tanggungan, rumah |
| previous_applications | 201.984 | approved/rejected/cancelled, bunga grid 0,20/0,24/0,28/0,32 |
| repayment_history | 1.361.341 | per cicilan; DPD sudah snapshot per cutoff (audit 23 Okt) |
| account_behavior | 1.084.418 | bulanan, 70.000 customer × 8–23 bulan lengkap sebelum keputusan |
| bureau_observations | 797.563 | 3 provider, skor = laten + N(0,23) (reverse engineering teman) |
| ecommerce_transactions | 4.359.817 | item; sumber fitur price premium `pp_*` |
- Train/test: 1 aplikasi per customer, customer disjoint, interleaved Jan–Des 2025; PSI fitur maks 0,0016 →
  **StratifiedKFold 5** adalah skema CV yang tepat (adversarial AUC 0,4987).

## Penilaian dan aturan resmi
- Public LB 45% + Private LB 45% + Notebook reproducibility 10%. Kaggle memakai 30% public / 70% private dari test
  (9.000 / 21.000 baris; dikonfirmasi user 21 Okt).
- Notebook wajib reproducible, sesuai pendekatan & submisi tim, nama `ParticipantNumber_TeamName_Notebook`
  (contoh `DSC0001_RandomSeed_Notebook`). Deadline **30 Okt 2026 23:59 WIB**, telat −1 poin/menit.
- Submisi: CSV `application_id,default_probability`, 30.000 baris, ID unik lengkap, nilai finite di [0,1].
  **Maks 3 submisi/hari per tim, reset 07:00 WIB.**
- Data eksternal dilarang; leakage masa depan dilarang; manipulasi/probing leaderboard dilarang.
  **Semi-supervised/pseudo-label diizinkan panitia (20 Okt), hanya data panitia.** Model pretrained publik dipakai.
- Tim: **"lagi cape lomba asli"** — akun Kaggle athillazaidan, nicholaswisee, riantamaputra. Akun CLI: `riantamaputra`.
  Slug: `data-science-competition-logika-ui-2026`. **Nomor peserta (DSCxxxx) belum diketahui.**

## Leaderboard (snapshot 24 Okt ±15:00 WIB)
| # | Tim | Public |
|---|---|---:|
| 1 | Datadataan | 0,57788 |
| 2 | cupuu | 0,57768 |
| 3 | Magang Bareng Afiq | 0,57458 |
| 4 | ꧁༺₦Ї₦ℑ₳༻꧂ | 0,57351 |
| 5 | Antech2Async | 0,57232 |
| 6 | abis bantai filkom geser mipa | 0,57198 |
| 7 | depok sejuk dsj | 0,57189 |
| 8 | IRISIAN lagi mikir | 0,57168 |
| 9 | Hypomonē | 0,57141 |
| 10 | BCC mau ke kopdes depok | 0,57131 |
- Batas top-8 ≈ **0,5717**; tim kita 0,56470 setelah submisi 24 Okt malam (sebelumnya 0,56405). Selisih ke top-8 ≈
  +0,007 public ≈ +0,006 OOF (offset historis public − OOF ≈ +0,035…+0,039).
- Petunjuk eksternal (19 Okt, dari user): tim 0,57065 (#8 saat itu) memakai "Kumo ensemble 3 seed".

## Hardware
- Laptop CachyOS, **RTX 3070 Ti 8 GB** (≈7,69 GiB usable), **i7-12700H** (CPU 0–11 = 6 P-core HT), RAM 31 GB.
- Python utama `.venv/bin/python` (3.13, polars 2.0.0, sdm 0.2.1, torch, lightgbm 4.x). `.venv-zoo` untuk AutoGluon.
  Pipeline teman butuh environment Python 3.11 terpisah (`environments/teman/`).
- Kuota Kaggle kernel habis → **semua run lokal**; submit CSV lewat `kaggle competitions submit` (CLI 2.2.4).
