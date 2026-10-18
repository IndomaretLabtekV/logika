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

⚠ Revisi 16 Okt: keunggulan TabM dan logit stacking itu terlihat di OOF yang bias (early stopping di fold validasi). Di protokol bersih TabM ~0,518–0,520 vs LGB ~0,520, dan stack LR kalah di public. Struktur faktor laten aditif tetap berlaku (EBM aditif ≈ LGB), tapi jangan jadikan alasan memberi bobot besar ke TabM.

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

## Tambahan 17–18 Okt (di atas OOF bersih validation_v2)
### Peta informasi per keluarga fitur (`eda/105`, LR rank-gauss cross-fit 5 fold; LR semua fitur 0,5212)
| Keluarga (n fitur) | KS sendiri | KS hilang bila dibuang |
|---|---|---|
| repayment rp/rq/sv (36) | 0,343 | **0,0266** |
| bureau (50) | 0,272 | 0,0139 |
| harga e-commerce pp (17) | 0,376 | 0,0114 |
| akun ac/cf (38) | 0,149 | 0,0051 |
| perilaku e-commerce ec/ec2 (40) | 0,364 | 0,0041 |
| previous pv/ob (25) | 0,213 | 0,0030 |
| cross (7) | 0,164 | 0,0017 |
| aplikasi + profil (19) | 0,232 | −0,0009 |

Korelasi skor keluarga di dalam kelas: blok {aplikasi, bureau, previous} ~0,65 ("grade kredit"), blok {perilaku,
harga e-commerce} 0,84 ("konsumsi"), repayment ~0,3 ke keduanya → **dua faktor laten utama + repayment**.
Repayment paling unik (hilang 0,027), padahal hanya ada untuk ~66% customer.

### Struktur risiko
- **Model linear ≈ TabM ≈ GBDT**: LR pada fitur rank-gauss (`lr_q`) OOF 0,52119 vs TabM bersih 0,52100 vs LGB 0,52316.
  Risiko hampir linear dalam fitur yang ditransformasi monoton → kapasitas nonlinear tidak menambah. Ini menjelaskan
  kenapa tuning/arsitektur tidak pernah membantu.
- **Learning curve datar** (`eda/102`, fold 0): lgb 0,5122/0,5169/0,5198/0,5197 di 40/60/80/100% baris train; cat_s
  0,5151/0,5167/0,5188/0,5194. Model di plafon informasi fitur; jarak ke top-1 bukan soal jumlah data.
- 10-fold vs 5-fold (`eda/104`): 0,52067 vs 0,52074 → +12,5% data latih tidak membantu.

### Segmen (OOF bersih)
- Thin-file KS ~0,496, history ~0,535 (`eda/103`). Per keluarga thin-file: trees/causilo/kumo/tabm
  0,4963/0,4963/0,4964/0,4945; history 0,5344/0,5349/0,5372/0,5323 (`eda/108`). Kumo unggul tipis di history.
- Causilo unggul di thin-file dan rate tier 24–28% (`eda/103`), tapi bobot per segmen tidak menambah (−0,0005).
- `fm_select` global memberi model ICL 44 dari 104 kolom yang 100% NaN untuk thin-file; memisah per segmen (v29) justru
  menurunkan (0,52409 vs 0,52583) → kolom NaN praktis diabaikan model ICL.

### Waktu dan drift
- Efek bulan Mei (O/E 0,888, z −3,9; `eda/101`), muncul juga sebagai chi2 tertinggi `dec_month` di residual, tapi koreksi
  bulan atau segmen × bulan dKS ≤ p95 null → tidak bisa dimanfaatkan.
- Maju-waktu (train ≤ bulan m, nilai m+1) vs acak: rata-rata −0,0024, sd per bulan ~0,02 → tidak ada drift (`eda/104`).

### Hal yang dicari dan tidak ada (17–18 Okt)
- **Utang eksternal** (`eda/100`): bureau outstanding **tidak** memuat pinjaman lender sendiri (korelasi 0,14), delinquency
  bureau juga tidak memuat delinquency lender sendiri (korelasi 0,01–0,02). Total DTI chi2/df 2,97, dKS_cf +0,0018 (null
  p95 ~0,0015) → lemah; Causilo + 14 fitur utang eksternal 0,5237 vs 0,5249 → null.
- **Harga item vs kovariat item** (qty, metode bayar, ukuran order, jam, status): 0,0000 varians dijelaskan; residual antar
  item dalam order independen → `pp_mean` sudah statistik cukup (laten + noise iid).
- **Harga relatif per subkelompok** (`eda/111`; metode bayar, cancelled, malam/siang, weekend, qty>1, basket, dispersi,
  tren, 10 item terakhir, share kartu kredit/COD; 30+ kandidat): maks LR chi2 5,4 (`z_cancel_minus_all`) < batas Bonferroni
  9,4. Kalibrasi: pp_* lama juga null. Daya uji: sinyal bernilai +0,01 KS akan memberi chi2 ratusan → habis.
- **Residual 228 fitur vs OOF bersih** (`eda/103`): 0 di atas null (chi2/df p95 1,93, dKS p95 +0,0018).
- **Neighbor gap** (`eda/110`): jarak 1/10/50-NN validasi→train vs test→train rasio 1,001/1,0005/1,0006 → tidak ada
  struktur "tetangga dalam train" yang menggelembungkan CV model ICL.
