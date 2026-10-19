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
| Fine-tune Kumo (18 Okt sore, bug sudah diperbaiki) | blok ICL lr 1e-5: inner-stop KS tidak naik dalam 150 langkah (null); semua parameter OOM di 8 GB |
| Stacked ICL (`stk_causilo`, `stk_kumo`: skor LGB OOF bersarang sebagai fitur) | fold 0 0,5224 vs 0,5249 / 0,5258 vs 0,5256; di v31 −0,0017 / −0,0011 |

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

## Tambahan 18 Okt sore
| Dicoba | Hasil |
|---|---|
| Timing lintas tabel (`eda/114`): ganti kerja vs riwayat, burst belanja sebelum jatuh tempo telat, low balance di bulan telat, belanja vs outflow | 11 kandidat, dKS maks +0,0012 < null p95 +0,0014 |
| Laten disiplin bayar per cicilan (`eda/115`, logistik kovariat + random intercept EB) | uji residual sedikit di atas null (re_sev_t1 χ²/df 3,5, dKS +0,0022) tapi **tidak transfer**: LGB −0,0003, LGB_s −0,0012, Causilo −0,0014 (P 0,21) |
| KumoRelational | tidak dijalankan: berbasis TabICLv2 (keluarga lebih lemah) dan meng-embed semua 4,36 juta item → tidak muat 8 GB tanpa subsampling yang merusak pp_mean |
| Feature bagging ICL (`fm_select(rand)`, Causilo 3 subset acak) | anggota 0,514–0,523; bag −0,0022 (P 0,09); di v31 −0,0006 |

## Tambahan 19 Okt (semua diuji di level keluarga pohon, referensi pohon saja)
| Dicoba | Hasil |
|---|---|
| Scan residual 228 fitur vs pohon saja (`eda/134`) | 1/228 (dec_month) |
| Eskalasi aplikasi sebelumnya (`eda/135`), keterjangkauan 3-arah (`eda/136`) | null |
| 3.150 fitur relatif-kelompok (`eda/137`) | LGB_s +0,0023 (P 0,98), keluarga +0,0002 |
| Seleksi konfigurasi LGB bersarang (`lgb_nested`) | −0,0002 |
| PCA rotation / LR stack untuk pohon | null |
| Pembersihan label bersarang (`eda/141`) | −0,001 |
| Snapshot fitur cutoff −90/−180 hari, 333 delta (`eda/142/143`) | LGB/XGB +0,0013–0,0022, keluarga −0,0002 |
| Validasi maju-waktu (`eda/140`) | tidak mereproduksi penalti ICL di public |
| TabR 3 seed (`results/v40`) | 0,51217; ditambah ke v31 −0,0008…−0,0026 |
| Distilasi ICL → LGB (`eda/118/119`) | fold 0 +0,004…+0,010, 5 fold ≤ v31 |
| Thin-view (`eda/125/126`) | thin +0,0026, pooled +0,0002 |
| Scan pita titik potong (`eda/127`), objective surrogate-KS (`eda/128`), spesialis pita (`eda/149`) | null / −0,004 / −0,0008…−0,0053 |
| 7.080 pasangan fitur (`eda/145/146`) | bersarang −0,0005 (yang bocor +0,0036) |
| Konstruksi lain (`eda/144`), persentil akun per bulan (`eda/138`) | null / lemah |
| Agregasi keluarga pohon prob/logit (`eda/147`) | ±0,0002 |
| Keluarga 9 pohon (lgb_x, xgb_d8, cat_d8) | −0,0001 |
| Seed outer tambahan untuk pohon (6 seed) | OOF 0,52324; test Spearman 0,99997 vs 3 seed |
| Snapshot −30/−365 (`eda/142 --shifts 30,365`) | keluarga −0,0009 |
| Riset web untuk kompetisi/generator | tidak ada informasi |
