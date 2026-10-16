# Cek lokal sesi 16 Okt (EDA dan aritmetika prediksi tersimpan)

Semua di laptop, tanpa melatih model: statistik, uji residual, atau logistic regression Newton numpy di atas logit
prediksi tersimpan (diperbolehkan, sama seperti stacking). Referensi: stack logit 9-model v8 (held-out 20×2 0,52811).
Hasilnya **semua null**. Script ad hoc tidak disimpan (dijalankan inline), kecuali bench di `eda/s01–s04`.

⚠ Referensinya OOF v8 yang bias (lihat `07_validasi_dan_leaderboard/bias_oof_early_stopping.md`). Untuk cek fitur
(apakah ada informasi baru) bias ini kecil pengaruhnya; untuk cek bobot blend pengaruhnya besar.

## Data dan leakage
| Cek | Hasil |
|---|---|
| Baris tabel bantu setelah cutoff | 0 (sudah dipotong panitia) |
| Urutan baris CSV tabel bantu | tidak membawa sinyal |
| Urutan baris test vs prediksi | acak |
| 1 aplikasi per customer | ya (customer_profiles 100k = 70k + 30k), jadi tidak ada label customer berulang |
| Semua kolom mentah sudah dipakai fitur? | ya; yang tidak dipakai hanya ID (`order_item_id`, `installment_id`) dan `opening_balance` (identitas eksak dengan closing bulan lalu) |

## Fitur harga (pp)
| Cek | Hasil |
|---|---|
| Beberapa skema pembobotan rata-rata premium per customer | plain mean tetap optimal, KS 0,3683 |
| Reliabilitas split-half pp_mean | 0,81 (KS hampir tidak bergantung jumlah item) |
| Tren / slope premium dari waktu ke waktu | null |
| Premium per metode bayar | null |
| Premium pada item dibatalkan | null |

## Repayment, akun, bureau, aplikasi
| Cek | Hasil |
|---|---|
| Persistensi Markov keterlambatan (p telat→telat) | chi2/df 3,4, dKS +0,0012 → marginal, tidak dipakai |
| Timing pembayaran (hari dalam bulan, jarak ke due) | null |
| Volatilitas dan tren income akun | null |
| Skala skor per provider bureau | simetris, tidak ada provider yang lebih informatif |
| Cicilan selected = anuitas dibulatkan ke atas; field waktu aplikasi | null |
| 9 fitur selected-history Astra di atas logit stack (LR + flag null) | −0,00068 (12/40) |

## Blending dan stacking
| Cek | Hasil |
|---|---|
| Bobot stack dioptimasi langsung untuk KS (hill climb di half A) | −0,00007 held-out |
| Interaksi logit × has_history | −0,00090 (6/40) |
| Stack terpisah per segmen has_history | −0,00072 (9/40) |
| TabICL17 bobot positif di blend rank v8 (5/10/15/20%) | −0,0005 / −0,0007 / −0,0011 / −0,0018 |
| Bootstrap baris gain stack +TabICL17 | +0,0003 ± 0,0007, P>0 0,69 |
| Rekonstruksi bobot blend v8 dari submission (NNLS di rank test) | cocok 0,99999999993; GBDT 46,7%, NN/FM 53,3% |

## Kesimpulan
Tidak ada sumber informasi baru. Yang ternyata penting bukan fitur, tapi **cara memvalidasi** (bias OOF).
