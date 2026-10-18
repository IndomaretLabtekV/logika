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

## Tambahan 16 Okt (semua null; detail `04_riwayat_versi/v10.md` … `v21`)
| Dicoba | Hasil |
|---|---|
| Tuning LGB 12 konfigurasi nested (v10) | +0,0022, 3/5 fold; campuran −0,0013 |
| Tuning TabM optimizer (v11) dan kapasitas (v12) | +0,0024 (3/5) / +0,0005 |
| EBM aditif / pairwise (v13) | 0,517 / 0,518 |
| Set-encoder baris mentah semua tabel (v14) | 0,5187 hybrid, 0,5107 events-only; stack −0,0002 |
| LGB rank_xendcg, XGB rank:pairwise, L2 LGB stacker (v15) | ≈ 0 |
| TabICLv2 ctx 40k / penuh (v16/v17) | 0,508 / 0,513; gain stack hanya lewat bobot negatif; public −0,0037 |
| LGB linear_tree, monotone, ExtraTrees (v18) | ≈ 0 |
| Pseudo-label self-training (v19) | ≈ 0 |
| Fitur mature-history Astra, pruning fitur redundan (v21) | −0,0011 / −0,0023 OOF |
| Stack LR per segmen, bobot optimasi-KS, interaksi logit × history | −0,0001 sampai −0,0009 |
| Belasan cek EDA lokal (lihat `05_katalog_eksperimen/eda_lokal_16okt.md`) | null |
| Stack logit-LR dengan bobot negatif sebagai submission (v9, S3) | public −0,0023 / −0,0037 vs blend rank v8 |
| Blend yang condong ke TabM (S5) | public −0,0028 |

## Tambahan 17–18 Okt (validation_v2; detail di `04_riwayat_versi/` dan `05_katalog_eksperimen/`)
### Model
| Dicoba | Hasil |
|---|---|
| TabPFN-3.5, TabICLv2 (AutoGluon dan sdm), RealMLP-TD, TabSwift, AutoGluon LGB/XGB/Cat/NN/FastAI/Linear, RF/ET/KNN, WOE-LR | fold 0 semua ≤ resep v8 atau +20% di bawah P 0,7 |
| iLTM (v23), Mitra (v25), TabFM fp32 (v27) | 0,5216 / 0,4488 / 0,5173 |
| LimiX-2, TabDPT/Turbo, Nori, TabPFN-3 besar | OOM di 8 GB dan 15 GB, gagal load, atau gagal fit |
| Causilo e16 / f150 / f300 / f60 | ≤ titik default (4 est, 100 fitur) |
| Kumo small/medium, f154/198 fitur, e8, seed generator kedua (v33) | ≤ large e4 100 fitur ctx penuh dalam noise |
| Kumo dengan subsample context 20k (1 atau 2 draw) | −0,0018 OOF vs context penuh |
| seg_causilo (fit per segmen has_history, v29) | 0,52409 vs 0,52583 |
| TabM bersih | 0,52100, model terlemah; ≈ LR rank-gauss |
| Causilo + 14 fitur utang eksternal | 0,5237 vs 0,5249 |
| Fine-tune Kumo | belum valid (bug baseline 0,298) — **bukan null, belum diuji** |

### Blending
| Dicoba | Hasil |
|---|---|
| Blend tanpa pohon (Causilo+Kumo, Kumo saja) | public 0,55709 / 0,56056 vs 0,56405 |
| Bobot blend per segmen thin/history (`eda/108`) | −0,0005 (P 0,34), bobot tidak stabil |
| Greedy bagged vs bobot tetap per keluarga | +0,00164 (P 0,84) < +0,00221 (P 0,93) |
| Tambah TabM + LR ke campuran (v35) | CV setara, public 0,56298 (dalam klaster) |

### Fitur dan EDA
| Dicoba | Hasil |
|---|---|
| Utang/akun eksternal dari bureau (`eda/100`) | lemah (chi2/df 2,97, dKS +0,0018 ≈ null p95) |
| Shock makro per segmen × bulan (`eda/101`) | efek Mei ada, tidak bisa dimanfaatkan |
| Residual 228 fitur vs OOF bersih (`eda/103`) | 0 di atas null |
| Harga relatif per subkelompok, 30+ kandidat (`eda/111`) | maks chi2 5,4 < 9,4 |
| Kovariat harga item | 0,0000 varians dijelaskan |
| 10-fold vs 5-fold, refit dengan data lebih banyak | +0,0000 / datar |

### Proses
- Menjalankan full fold lokal karena salah baca status Kaggle → komputasi ganda (v26).
- Screening ICL dengan satu draw context → angka fold 0 terlalu optimis (0,5276 vs pooled 0,52484).
- Memperkirakan public dengan sd dari bootstrap selisih untuk varian hampir identik (v34 ± 0,0006) → terlalu sempit.
