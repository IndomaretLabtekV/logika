# Katalog semua script EDA (`eda/NN_*.py`, output di `eda/out/`)

Notasi:
- **chi2/df** adalah uji observed−expected per bin terhadap OOF referensi yang sudah dikalibrasi. Nilai sekitar 1 berarti tidak ada informasi baru.
- **dKS_cf** adalah gain KS cross-fit.
- Ambang noise biasanya chi2 p95 sekitar 1,2–2,2 dan dKS p95 sekitar +0,0015–0,002.
- ⚠ Script sebelum `eda/90` memakai bin kuantil yang bisa menyatukan nilai 0 dengan nilai positif langka (bug ditemukan Astra di v8). Hasil negatif untuk fitur bermassa-nol di script-script lama itu kurang kuat.

## Fase v1 (00–13): memahami data
| # | Pertanyaan | Hasil |
|---|---|---|
| 00 | Skema, null, kardinalitas | lihat `03_data_dan_fitur/dataset.md` |
| 01 | Kunci, target, waktu, cakupan, baris setelah cutoff | bad 13,58%, stabil per bulan, 0 baris setelah cutoff |
| 02 | Bangun matriks fitur | 1 baris per aplikasi, 176 fitur |
| 03–08 | KS per domain + plot desil | rate tier 0,23; rp_frac_dpd1 0,318; ec_accel 0,309; bc_score 0,25; ac_out_in_delta 0,11; demografi < 0,03 |
| 09 | Profil temporal bad vs good | bad mempercepat order sekitar 16 minggu sebelum keputusan (2,3× di minggu terakhir); cash flow memburuk 6 bulan |
| 10 | Duplikasi dan leakage | 2.388 item kembar itu sah; tidak ada twin; ID tidak membawa sinyal |
| 11 | Korelasi Spearman, redundansi > 0,95, KS kondisional | ec dan rp tetap sekitar 0,30 di dalam desil skor |
| 12 | Verifikasi preprocessing | recompute pandas 100% match; injeksi masa depan lulus |
| 13 | Scorecard tanpa model, simulasi gap | KS 0,41; \|pub−priv\| median sekitar 0,01, p95 sekitar 0,03 |

## Fase v2 (20–27): residual v1
| # | Hasil |
|---|---|
| 20 | Error OOF per segmen: thin 0,476 vs history 0,517; segmen lain rata |
| 21 | Laten bureau (tren, range, inquiry eksternal): kecil, lalu masuk sebagai fitur bz_ |
| 22 | E-commerce mendalam (EWM, cancel, malam, AOV): kecil, lalu masuk sebagai ec2_ |
| 23 | Struktur tersembunyi: anuitas eksak, identitas saldo eksak, partial payment bad 0,27 vs 0,089 |
| 24 | Konstruksi baru (cf_, ob_): kecil, masuk |
| 25 | Aditivitas: interaksi raw kuat **hilang** setelah dikondisikan ke v1, jadi risiko aditif di log-odds dan alasan menambah shallow trees |
| 26 | Keparahan telat: terkuantisasi 30/61/91; median bad 61 vs good 31, lalu masuk sebagai sv_ |
| 27 | Verifikasi v2: recompute dan injeksi lulus |

## Fase v3 (30–32)
| # | Hasil |
|---|---|
| 30 | Diagnosis FM: TabPFN dan Causilo −0,025 di semua segmen; residual vs GBDT setara noise; blend 10% FM cuma +0,0003 |
| 31 | Thin-file: 91/210 fitur terisi; kandidat (ec_orders_7d dll.) di bawah noise |
| 32 | Verifikasi nn_prep: rank-preserving, KS univariat tetap |

## Fase v5 (40–46): penemuan harga
| # | Hasil |
|---|---|
| 40 | DFS relasional (`dfs.py`, 301 kolom): hampir semuanya noise |
| 41 | Quantity: lemah; z mendekati ±2,7 |
| 42 | **Unit price premium: KS 0,368, chi2 28,8 vs v1, lalu menjadi fitur pp_** |
| 43 | Granularitas harga: grid 1.000, lognormal per kategori, pengali harga per customer |
| 44 | Deflasi harga per bulan: variannya tidak lebih baik |
| 45 | Level relatif lain (rl_*): semuanya chi2 ≤ 4,5, dKS ≤ 0,0015 |
| 46 | Verifikasi v5: recompute 300 match; injeksi masa depan 0/231 bergeser |

## Fase v6 (50–53)
| # | Hasil |
|---|---|
| 50 | Referensi harga expanding per bulan: KS 0,3689, sinyal dipertahankan |
| 51 | 39 hidden dimension per baris: 0 di atas noise |
| 52 | 301 DFS + interaksi pp vs v5: tidak ada gain |
| 53 | Verifikasi referensi harga v6 (injeksi 2 bagian): lulus |

## Fase setelah v6 (60–72): mencari kegagalan v6, semua negatif
| # | Hasil |
|---|---|
| 60 | Noise public: sd KS public 0,012 (30%) / 0,009 (50%); sd selisih v6−v5 0,0026; KS per bulan 0,50–0,57 |
| 61 | Naive-Bayes MIL per baris per tabel: 0/25 di atas noise (⚠ ini sebenarnya model supervised sederhana) |
| 62 | Kalibrasi blend per segmen dan stacking disagreement: tidak berguna; rekalibrasi per bulan malah merugikan |
| 63 | Penilaian lender (premium rate/reject/cut di luar skor): 0/6 |
| 64 | Makro dan tanggal: sd mingguan 0,0101 vs binomial 0,0095, tidak ada efek |
| 65 | 703 log-rasio + 435 grid interaksi: 0/1.138 lolos di kedua separuh |
| 66a/b | DFS native TabPFN-Rel (relarena_core, `.venv-rel`): depth 2 0/172 dan depth 3 0/297 di atas noise; KS terbaik 0,309 vs pp 0,369 |
| 67 | Band di sekitar cut KS (q = 0,673): AUC 0,51–0,54 untuk semua model; reorder −0,003 |
| 68 | Variance seed: run v5 vs v6 korelasi 0,999; rata-ratanya −0,0007 sampai +0,0006, jadi tambah seed tidak berguna |
| 69 | "Ceiling" dan simulasi LB: KS kalau p v6 adalah kebenaran 0,5277 ± 0,0046 vs observed 0,5283; 0,541 butuh spread log-odds sekitar 5% lebih lebar; top-of-20 tim setara (best of 10) +0,012 sampai +0,028 di atas tim setara (⚠ Astra: ini self-consistency, bukan ceiling) |
| 70 | Matched filter burst Poisson LLR: KS 0,26, minggu × harga 0,367, noise |
| 71 | Bundle test top-K dari 1.228 kandidat (additive, cross-fit): K=5 +0,0008, K=20 +0,0002, K=60 −0,0016; bundle noise +0,0012/+0,0009/−0,0002, jadi tidak ada gain |
| 72 | Sequence bigram/trigram MIL (repayment, account, order): 0/12 (⚠ supervised sederhana; representasi terbatas) |
| ad hoc | Unit price: efek level order adalah artefak (korelasi dalam order 0,105 = antar order 0,106); near-duplicate customer tidak ada; false negative terburuk terlihat sehat (noise Bernoulli) |

## Fase v8 (80–83): label historis
| # | Hasil |
|---|---|
| 80 | 81.702 pinjaman historis berlabel, bad 14,37%; jumlah historis bad → selected bad 8,8% / 34,3% / 43,6% |
| 81 | Snapshot fitur: identity 228/228; future invariance 0 bergeser; `data_pq/hist_feats.parquet` |
| 82 | Transfer: Spearman KS 0,79; pp dan ec tidak transfer (0,37 → 0,14; 0,31 → 0,06) |
| 83 | Counting +hist bobot 0,5: +0,0027 (5/6); +noec −0,0013; thin-file 0 |

## Fase v9 (90–96): `common9.py`, bin aman-nol, referensi = OOF blend v8 terkalibrasi
| # | Pertanyaan | Hasil |
|---|---|---|
| 90 | Link antar customer lewat 14 kunci nilai eksak (timestamp, saldo, order_total, profil, income) | \|z\| < 2, chi2 ≤ 2,35 (sel_decision_datetime, 1,98 z, multiple testing); tidak ada clone |
| 91 | Residual v8 vs count/sum event per (jenis event × jendela umur) di semua tabel | 1.142 uji, \|z\| maks 3,81; ekspektasi null sekitar 3,8, jadi null |
| 92 | Identitas aritmetika dan residualnya | semuanya eksak; residual tidak eksak chi2 ≤ 1,44 |
| 93 | Persona k-means (komposisi kategori, metode bayar, jam, hari, cancel, qty), k = 2…20 | bad rate cluster 10–15%; chi2 ≤ 0,96; PCA tanpa sumbu dominan |
| 94 | Empirical Bayes (beta-binomial keterlambatan, normal-normal harga, gamma-Poisson laju order, reliabilitas bureau) | KS univariat tidak naik; eb_k1/eb_k30 chi2 sekitar 2 (lemah, multiple testing) |
| 95 | pp_mean dll. di-residualisasi terhadap blok nuisance (income, umur, tenure, tanggungan, dummy) | pp_mean 0,369 → 0,376 univariat, tapi chi2 vs v8 1,18 (sudah dipakai model) |
| 96 | Rekalibrasi per segmen vs logit stacking | rekalibrasi −0,0005 sampai −0,0026; **logit stacking +0,0008 (5/6)**; stacking × history −0,0007 |
| cek cepat | urutan baris CSV, jendela aplikasi→keputusan, offset public, simetri provider bureau, korelasi di dalam kelas, spesialisasi segmen | lihat `03_data_dan_fitur/temuan_eda_utama.md` dan `07_validasi_dan_leaderboard/` |

## Fase validation_v2 (100–111, 17–18 Okt): referensi = OOF bersih cross-fit
| # | Pertanyaan | Hasil |
|---|---|---|
| 100 | Utang/akun eksternal = bureau − pinjaman lender sendiri? | bureau outstanding tidak memuat pinjaman sendiri (korelasi 0,14), delinquency juga tidak (0,01–0,02); total DTI chi2/df 2,97, dKS_cf +0,0018 (null p95 ~0,0015) → lemah; Causilo + 14 fitur 0,5237 vs 0,5249 → null |
| 101 | Shock makro per segmen × bulan | efek Mei O/E 0,888 (z −3,9), tapi koreksi bulan / segmen × bulan dKS ≤ null p95 → null |
| 102 | Learning curve fold 0 (40/60/80/100%) | lgb 0,5122/0,5169/0,5198/0,5197; cat_s 0,5151/0,5167/0,5188/0,5194 → datar; plafon informasi fitur |
| 103 | Residual 228 fitur vs OOF bersih (trees+causilo terkalibrasi) + KS per segmen | 0/228 di atas null (chi2/df p95 1,93, dKS p95 +0,0018); thin ~0,496 vs history ~0,535; Causilo unggul di thin-file & rate 24–28% |
| 104 | Skema CV: 5 vs 10 fold, maju-waktu, adversarial | 0,52074 vs 0,52067 (P 0,59); maju-waktu −0,0024 (sd bulan ~0,02) → tanpa drift; adversarial AUC 0,4987 |
| 105 | Peta informasi per keluarga fitur (LR rank-gauss) | repayment hilang 0,0266, bureau 0,0139, pp 0,0114, akun 0,0051, perilaku ec 0,0041, previous 0,0030; dua faktor laten + repayment |
| 106 | Kenapa `kumo_fit` 0,5182 vs sdm_screen 0,5276 | varians draw context 0,518–0,528 + estimator ±0,002; tidak ada bug |
| 107 | Noise public (subset 30% OOF cross-fit) | trees_causilo_kumo − v8: +0,0016 ± 0,0022 (20k), +0,0024 ± 0,0022 (ctx penuh); icl_nn_only − trees_only +0,0018 ± 0,0026 (S5−S4 public −0,0027 di ekor bawah ~P 0,1) |
| 108 | Bobot blend per segmen thin/history | −0,0005 (P 0,34) vs global; bobot grid tidak stabil; bobot tetap 1:1:1 lebih baik → null |
| 109 | (output saja, `out/109_seg_causilo.txt`) seg_causilo di blend | +0,00154 / +0,00162 (P 0,83 / 0,85) = noise |
| 110 | Neighbor gap validasi→train vs test→train | rasio 1,001 / 1,0005 / 1,0006 (1/10/50-NN) → null; menolak hipotesis CV ICL menggelembung |
| 111 | Harga relatif per subkelompok (30+ kandidat) | maks LR chi2 5,4 < Bonferroni 9,4; pp_* lama juga null → habis |
| ad hoc | Harga item vs kovariat item (qty, bayar, ukuran order, jam, status) | 0,0000 varians dijelaskan → pp_mean statistik cukup |
| ad hoc | account_behavior memuat bulan berjalan? | tidak (70.000/70.000 bulan terakhir = bulan keputusan − 1) |
| ad hoc | Lag keputusan previous apps | null (sudah di eda/51) |

Output: `eda/out/100_*` … `111_*`, blend `eda/out/blend2_v22.txt`, `blend2_v26.txt`, `blend2_v28.txt`, `blend2_v36.txt`,
riwayat submission `eda/out/107_submissions.csv`, noise public lama `107_public_noise.txt` (Kumo 20k) dan baru
`107_public_noise_v28.txt` (Kumo ctx penuh).

## Fase sesi 18 Okt sore – 19 Okt (112–152): referensi OOF bersih; sejak 19 Okt referensi utama = pohon saja
| # | Pertanyaan | Hasil |
|---|---|---|
| 112 | Kenapa `kumo_finetune.py` KS 0,298? (4 cara bangun tabel vs OOF v28m) | semua identik (Spearman 1,0); penyebab ternyata urutan kolom kelas per panggilan (diuji di scratchpad, dicatat di jurnal siklus 1) |
| 113 | Apa yang ICL tahu di luar pohon | residual Kumo LR χ² 255, Causilo ⊂ Kumo; penggerak cf_nonincome_share, pv_appr_max, tenure |
| 114 | Timing lintas tabel (11 kandidat) | null (dKS ≤ +0,0012 < p95 +0,0014) |
| 115 | Laten disiplin bayar per cicilan (EB) | residual lolos tipis; model null (LGB −0,0003, Causilo −0,0014) |
| 116 | Feature bagging ICL (eval) | bag −0,0022 |
| 117 | Pasangan bad/good mentah di pita yang sama | tidak ada pola yang terlihat mata |
| 118/119 | Distilasi Causilo bersarang → LGB | fold 0 +0,004…+0,010, 5 fold ≤ v31 |
| 120 | Swap keluarga v31 (Causilo/Kumo multi-seed) | margin jujur v31 +0,0017–0,0019 |
| 121 | Perkiraan probe pohon-saja | −0,0030 ± 0,0024 vs v31 → public teramati −0,0007 |
| 122 | Kurva averaging seed (LGB, Causilo) | keduanya naik +0,002–0,003; hipotesis "hanya pohon" ditolak |
| 123 | v31 dengan komponen di-average | ≈ v31 |
| 124 | Tambah komponen ke v31 (TabR) | −0,001…−0,0026 |
| 125/126 | Thin-view model | thin +0,0026, pooled +0,0002 |
| 127 | Scan residual di pita titik potong | 2/228 (multiple testing) |
| 128 | Objective boosting surrogate-KS | −0,004 |
| 129 | Kandidat v31 varian rendah | Spearman v31 0,99987 |
| 130 | Konsistensi prediksi test ICL | sehat (Spearman vs pohon OOF 0,984 / test 0,987) |
| 131 | Offset public−CV per keluarga | pohon +0,039, LR +0,037, Kumo +0,034, Causilo +0,029 |
| 132 | Gap ICL vs pohon per subset alami | tidak ada subset −0,01 di luar noise |
| 133 | Label tetangga (kNN) di luar pohon/v31 | χ² ≤ 3,5 → tidak ada sinyal tetangga |
| 134 | Residual 228 fitur vs pohon saja | 1/228 (dec_month) |
| 135 | Eskalasi aplikasi sebelumnya (9) | null |
| 136 | Keterjangkauan 3-arah (8) | null |
| 137 | 3.150 fitur relatif-kelompok | anggota dangkal +0,002 (seleksi bocor), keluarga null |
| 138 | Persentil akun per bulan populasi (patuh cutoff) | lemah |
| 139 | Audit cutoff train vs test | bersih |
| 140 | Validasi maju-waktu per keluarga | tidak mereproduksi penalti ICL |
| 141 | Pembersihan label bersarang | −0,001 |
| 142/143 | Snapshot fitur cutoff −90/−180 (333 delta) | anggota +0,0013–0,0022, keluarga −0,0002 |
| 144 | Konstruksi lain (volatilitas outflow, residual skor bureau, LTI lama, per-provider) | null |
| 145/146 | 7.080 pasangan fitur; versi bersarang | +0,0036 palsu → −0,0005 (winner's curse) |
| 147 | Agregasi keluarga pohon rank vs prob vs logit | ±0,0002 |
| 148 | Lantai noise label | tidak ada (2% teraman 0 bad) |
| 149 | Spesialis pita titik potong (bersarang) | −0,0008…−0,0053 |
| 150 (inline, `out/150_tree_variants.txt`) | Keluarga 9 pohon (lgb_x, xgb_d8, cat_d8) | −0,0001 |
| 151 (`142 --shifts 30,365`) | Snapshot −30/−365 | keluarga −0,0009 |
| 152 | VIP-COP untuk Kumo | **dibatalkan atas permintaan user** sebelum ada hasil (lihat `10_ide_belum_dicoba/`) |
Output: `eda/out/112_*` … `152_*` (txt/csv/png; parquet di-.gitignore).
