# Temuan EDA utama (angka)

KS univariat dihitung di train. NaN diperlakukan sebagai bin ekstrem tersendiri (`ks_nan`).

## Sinyal terkuat
| Fitur / keluarga | KS univariat | Catatan |
|---|---|---|
| `pp_mean` (price premium: rata-rata z log unit price per kategori) | **0,368–0,369** | ditemukan di v5 (`eda/41–46`). Varian: pp_trim 0,367, pp_done 0,366, pp_90 0,328, pp_30 0,315 |
| `rp_frac_dpd1` (fraksi cicilan telat >0 hari) | 0,318 | keluarga repayment terkuat. Bad rate 3,7% kalau selalu tepat waktu, 34,8% kalau pernah 90+ |
| `ec_accel_30d` (laju order 30 hari terakhir ÷ baseline) | 0,309 | 0,345 di thin-file. Bad mempercepat belanja sekitar 16 minggu sebelum keputusan, sampai 2,3× di minggu terakhir |
| `ec2_ewm28_exc`, `ec2_ewm56_rel` (burst halus) | sekitar 0,31 | v2 |
| `rp_maxdpd_mean`, `rp_frac_dpd30` | sekitar 0,31 | |
| `bc_score_mean` (skor bureau) | 0,25 | |
| `annual_interest_rate` | 0,23 | bad rate per tier 5,5% / 10,6% / 17,8% / 30,4% / 50,3%. Hampir redundan dengan skor (KS di dalam bin skor sekitar 0,06) |
| `ac_out_in_delta` (memburuknya cash flow) | 0,11 | |
| Demografi | < 0,03 | lemah |

- Scorecard tanpa model (`eda/13`): KS 0,41.
- E-commerce dan repayment tetap memberi KS sekitar 0,30 **di dalam desil skor bureau**, artinya ortogonal terhadap skor.

## Struktur sinyal (v9, korelasi di dalam kelas)
Di dalam kelas target=0 dan target=1 (jadi bukan efek label), korelasinya:
- pp_mean ~ ec_accel_30d: **+0,52**;
- pp_mean ~ rp_frac_dpd1: +0,37;
- pp_mean ~ declared income: +0,22 sampai +0,26;
- bc_score ~ rate: −0,72.

Interpretasinya: risiko dibentuk oleh beberapa **faktor laten** (stress / konsumsi berlebih, blok bureau/rate/utilisasi/penolakan, blok affordability), dan masing-masing diukur dengan noise lewat beberapa kanal. Ini menjelaskan kenapa TabM (NN) konsisten menjadi model tunggal terbaik, dan kenapa logit stacking sedikit lebih baik daripada rata-rata rank.

## Segmen
- Thin-file KS jauh lebih rendah daripada history. v1: 0,476 vs 0,517. v5 dan seterusnya: sekitar 0,500 vs 0,536. v9: 0,5009 vs 0,5357.
- Lompatan thin-file 0,477 → 0,500 datang dari fitur harga (v5), karena thin-file tetap punya e-commerce.
- KS per bulan (random OOF) berkisar 0,50–0,57 (sd sekitar 0,02). Ini variasi sampling, bukan drift.
- Tidak ada spesialisasi segmen: TabM terbaik baik di thin-file maupun history. Rekalibrasi per segmen selalu **menurunkan** KS pooled.

## Hal yang dicari tapi tidak ada
- Twin, ring, atau clone customer lewat nilai eksak yang sama (14 kunci) tidak ada (`eda/90`).
- Persona atau cluster perilaku tidak ada. Bad rate antar cluster hanya 10–15% (`eda/93`).
- Efek tanggal, makro, atau kalender tidak ada. Sd mingguan observed−expected 0,0101 vs binomial 0,0095 (`eda/64`).
- Pola urutan (bigram/trigram) tidak ada (`eda/72`).
- Kebocoran lewat ID atau urutan baris CSV tidak ada.
- Pola premium dari pemberi pinjaman (rate atau reject di luar skor) tidak ada (`eda/63`).
- Dari pola false negative terburuk: mereka terlihat **lebih sehat** daripada rata-rata good. Error ini lebih konsisten dengan noise Bernoulli daripada pola yang terlewat.
