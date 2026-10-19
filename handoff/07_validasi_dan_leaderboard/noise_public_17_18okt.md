# Noise public dan kalibrasi perkiraan (17–18 Okt)

## Metode perkiraan sebelum submit (`eda/107_public_noise.py`)
OOF cross-fit bersih dinilai di subset acak 30% (asumsi public = 9.000 dari 30.000 test; proporsi sebenarnya tidak
diketahui) → distribusi KS(kandidat) − KS(resep v8). Perkiraan public = 0,56145 (v8) + rata-rata selisih, dengan sd
ditambah karena public resep v8 bersih sendiri tidak pernah diukur.

| Pasangan | Selisih OOF penuh | Subset 30% |
|---|---|---|
| trees_causilo_kumo-20k − resep v8 | +0,00175 | +0,0016 ± 0,0022 (5–95% −0,0021..+0,0052), P(<0) 0,23 |
| trees_causilo_kumo ctx penuh − resep v8 (v31) | +0,00221 | +0,0024 ± 0,0022, P(<0) 0,14 |
| icl_nn_only − trees_only (analog bersih S5 vs S4) | +0,0007 | +0,0018 ± 0,0026; public S5−S4 −0,0027 ada di ekor bawah (~P 0,1) |

Satu submission tidak bisa membuktikan gain +0,002; hanya bisa menolak hipotesis kasar (misal "ICL rusak di public" kalau ≤ −0,004).

## Hasil vs perkiraan
| Submission | Perkiraan | Public | z |
|---|---|---|---|
| v31 | 0,5633 ± 0,0025 | 0,56405 | +0,3 |
| v31b (tanpa pohon) | 0,5642 ± 0,0017 | 0,55709 | **−4** |
| v31c (Kumo saja) | 0,5640 ± 0,0021 | 0,56056 | −1,6 |
| v32 | 0,5640 ± 0,0018 | 0,56376 | −0,1 |
| v34 | 0,5640 ± 0,0006 | 0,56271 | −2,2 (sd terlalu kecil) |
| v35 | 0,5638 ± 0,0015 | 0,56298 | −0,5 |

## Kalibrasi baru: "Spearman 0,9999 bukan berarti public sama"
v34 vs v31: Spearman 0,99985, selisih public −0,0013. KS adalah statistik maksimum di satu titik potong, jadi perturbasi
rank sekecil apa pun di sekitar titik itu menggeser KS public ~0,001. Konsekuensi:
- sd perkiraan untuk varian yang hampir identik **minimal ~0,001**, bukan dari sd bootstrap selisih.
- Klaster v31/v32/v34/v35 = 0,5627–0,5641 (mean 0,5634, sd 0,0006): **selisih < ~0,002 tidak bermakna**.
- Jangan memilih final dari selisih public kecil; pilih yang reproducible atau setara-CV.

## Gap ke leaderboard atas
- Kita 0,56405 (#24); top-1 0,57768 (+0,0136), #2 0,57530, #3 0,57290, tujuh tim > 0,570; 0,563–0,568 padat.
- 0,0136 ≈ 20× sd selisih antar varian kita → tidak bisa ditutup oleh keberuntungan satu submission atau pemilihan blend.
- Kontinuum skor konsisten dengan plafon ~0,565 + noise public + seleksi best-of-many (simulasi lama `eda/69`: tim teratas
  dari 20 tim setara dengan best-of-10 rata-rata +0,012 s.d. +0,028), **atau** tim atas punya informasi/metode yang tidak
  kita temukan. Tidak bisa dibedakan tanpa prediksi mereka.

## Hasil vs perkiraan, 19 Okt (probe keluarga)
| Submission | Perkiraan | Public | z |
|---|---|---|---|
| v39a pohon saja | 0,5611 ± 0,0026 | 0,56337 | +0,9 |
| v39b Causilo saja | 0,5614 ± 0,0024 | 0,55333 | **−3,4** |
| v39c LR saja | 0,5614 ± 0,0036 | 0,55835 | −0,9 |
- sd selisih subset 30% untuk pasangan berkorelasi 0,97–0,98 ternyata ±0,0031–0,0035 (bukan 0,002).
- Pola z semua probe ICL negatif (v31b −4, v31c −1,6, Causilo −3,4): **ICL di-overrate CV** secara konsisten di public.
- Pohon positif (+0,9) → offset pohon public−CV +0,039 paling tinggi di antara keluarga.
