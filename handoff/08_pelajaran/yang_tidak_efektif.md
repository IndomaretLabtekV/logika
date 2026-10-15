# Yang tidak efektif (jangan diulang tanpa alasan baru)

## Fitur dan EDA
| Dicoba | Hasil | Kenapa |
|---|---|---|
| Lebih dari 1.200 rasio dan interaksi berpasangan (`eda/65`) | 0/1.138 | tree sudah menangkap interaksi; risiko aditif |
| DFS (buatan sendiri dan native TabPFN-Rel, depth 2/3) | tidak ada gain; TabPFN di DFS 0,49 | DFS generik tidak membuat konstruksi seperti premium harga |
| Hidden dimensions per baris (39), payday, jam, pembulatan | 0/39 | data sintetis tidak menyimpan sinyal di sana |
| Sequence bigram/trigram, matched filter burst | 0/12, noise | sinyal urutan sudah terwakili oleh EWM dan accel |
| Persona / cluster | bad rate 10–15% | tidak ada tipe customer diskret |
| Empirical Bayes shrinkage | KS tidak naik | model sudah belajar reliabilitas lewat fitur count |
| Residualisasi nuisance | univariat +0,007, tapi sudah dipakai model | model sudah melakukan "pengurangan" ini |
| Varian ukuran harga (median, shrinkage, order-equal, per kategori, deflasi) | ≤ noise | pp_mean sudah hampir optimal |
| Link antar customer / clone | tidak ada | tidak ada struktur ring |
| Event scan per jendela waktu di semua tabel | max \|z\| 3,81 = null | jendela-jendela yang ada sudah cukup |
| Thin-file khusus (`eda/31`) | di bawah noise | thin-file hanya punya bureau dan e-commerce |
| Spending 112–224 hari, semantic rate (v4) | +0,0028 di 2/5 fold, +0,0005 | tidak konsisten |
| Augmentasi pinjaman historis (v8) | ≈ 0, CI melintasi nol | domain shift, rekonstruksi tidak eksak, sinyal utama (pp, ec) tidak transfer |

## Model dan ensemble
| Dicoba | Hasil |
|---|---|
| Tambah seed (3 → 5, rata-rata dua run) | ≈ 0 (korelasi 0,999) |
| TabM 5 seed vs 3 | 0,5253 vs 0,5257, tidak naik |
| TabM periodic, RealMLP | setara TabM atau lebih rendah; tidak menambah di blend |
| TabPFN-3.5, Causilo | 0,519–0,524; bobot kecil |
| TabPFN-Rel native | 0,4917 |
| EXAONE, LimiX-2, TabICLv2 | tidak pernah selesai di T4 |
| Greedy dengan bobot dari OOF penuh | overfit, bobot tidak stabil |
| Rekalibrasi per segmen, reorder band cut KS | merugikan |
| Menumpuk banyak model berkorelasi ke stacking | bisa merugikan (v9 11 model < 9 model) |

## Proses
- Run panjang dengan banyak model baru sekaligus (v8). Satu model crash, semua yang di belakangnya ikut hilang, dan tidak ada pertanyaan yang terjawab tuntas.
- Mengubah beberapa hal sekaligus (v6) membuat perubahan public tidak bisa diatribusikan.
- Screening fitur terhadap OOF yang sudah dipilih memakai label yang sama berulang kali, jadi "konfirmasi" bukan holdout baru.
