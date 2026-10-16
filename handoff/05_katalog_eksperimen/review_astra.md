# Review dari GPT-6 Astra (asisten lain yang dipakai user)

Setiap folder punya `README.md`, `RESULTS.md` (bahasa Indonesia), script, dan `out/`. Kalau perlu detail, baca `RESULTS.md` langsung.

## `eda/deep/` (13 Okt, setelah v3)
- Integritas: semua PK unik. `order_total` per item menggelembung 1,7396×. 2.388 item kembar itu sah.
- Simulasi public share 10/30/50%. Bobot 45/45 bukan proporsi baris.
- 289 fitur (210 lama + 79 kandidat temporal) diuji dengan contingency test bertingkat + BH. Satu-satunya nominee: **spending 112–224 hari** (discovery q 0,046, confirmation p 0,018). Dites di v4: +0,0028 tapi cuma 2/5 fold, jadi tidak diadopsi.
- Preprocessing: ECDF interpolated + Gaussian aman (Δ KS 0); robust + asinh merusak (0,033).
- Float `pv_rate_delta` (17 nilai float untuk 9 selisih ekonomi) secara tidak sengaja membedakan tier. Usulannya: bulatkan + kategori `rate_transition`. Dites di v4: +0,0005.
- Audit feature budget FM, ekspor relasional, dan catatan model (TabPFN-Rel, Causilo, EXAONE, Mitra-v2, RT-PluRel).

## `eda/v5_review/` (14 Okt)
Diterima dan diperbaiki di v6:
- referensi harga global;
- nn_prep di seluruh train;
- probe EXAONE (0 positif di query, seleksi fitur memakai label query, 104 kolom memicu FS);
- bobot greedy tidak stabil;
- dependensi tidak di-pin.

Hasil lain: 29 kandidat harga plus kontrol, 0 lolos replikasi; v5 − v3 +0,0132 [CI +0,0078, +0,018].

## `eda/v6_review/` (14 Okt)
- v6 tidak terbukti lebih buruk; penyebabnya margin kecil dan bobot sensitif.
- Referensi harga fold-local vs v6: ρ 0,99999838.
- Rasio DPD exposure-matang dan delta skor provider yang sama: 0/15 lolos.
- Sintetis: identitas saldo eksak, grid 1.000 pada item.
- Saran: isolasi satu perubahan per eksperimen dan EXAONE bounded pilot.

## `eda/v7_review/` (15 Okt)
Koreksi atas overclaim assistant (diterima):
- `eda/69` bukan ceiling;
- offset public−OOF bukan konverter;
- AUC ≠ KS (5 pasangan berlawanan);
- rank averaging tidak otomatis terbaik;
- mean fold KS > pooled;
- `eda/72` hanya menguji representasi tertentu;
- `calibrate` pada label yang sama lalu residual test bukan validasi independen.

Temuan teknis:
- akar crash EXAONE (pesan "cannot fit" tidak cocok);
- Causilo dan DFS kehilangan tipe kategori;
- input RealMLP (median + flag, 460 kolom);
- reliabilitas harga (korelasi split-half 0,50 untuk 1–15 order, 0,74 untuk >60);
- order-equal mean 0,375 vs 0,373 (tidak lolos BH).

Rencana 16 hari dari Astra:
1. EXAONE 8–12k;
2. RealMLP-TD;
3. Causilo kategori bertipe;
4. ablation checkpoint AUC vs KS;
5. TabICLv2 pilot;
6. DFS dengan lineage.

## `eda/v8_review/` (15 Okt)
- Bobot aktual CSV v8 direkonstruksi: 54,83% augmented. OOF 0,527025.
- Augmentasi tidak lolos (semua CI melintasi nol). Snapshot historis tidak identifikasi (ambiguitas ledger, profil saat ini, domain shift). Protokolnya transduktif.
- **Bug bin kuantil** di screening assistant (0 dan nilai positif langka digabung). Setelah diperbaiki, 0/30 kandidat lolos BH. Matched-provider bureau delta p 0,014/0,012 tapi q 0,22/0,36.
- 9 fitur selected-history (`out/03_selected_history_features.parquet`): diuji 16 Okt di v21, null.
- Saran: jalankan RealMLP dan EXAONE sebagai eksperimen terpisah; TabICL hanya setelah pilot dengan query yang di-chunk; selected-only sebagai kontrol; satu run untuk satu pertanyaan.

## Status rekomendasi Astra
| Rekomendasi | Status |
|---|---|
| Referensi harga aman-waktu, nn_prep per fold | ✅ v6 |
| Probe EXAONE benar, ≤ 100 kolom | ✅ v6 (jalan di 3k) |
| EXAONE OOF lengkap 8–12k | ❌ tidak pernah selesai (v7 crash 56k; v8 tidak sampai; v9 dibuang; 16 Okt tidak dicoba) |
| Causilo kategori bertipe | ✅ v8/v9 (0,5226 → 0,524) |
| RealMLP-TD | ✅ v9 (0,5223, tidak membantu) |
| TabICLv2 | ✅ 16 Okt v16/v17 dengan query di-chunk: 0,5079 / 0,5133, tidak menambah (S3 public −0,0037) |
| Ablation kebijakan checkpoint (AUC vs KS) | ⚠ tidak diuji langsung; ditemukan masalah lebih besar: checkpoint dipilih di fold validasi (bias OOF v1–v9) |
| DFS native dengan kategori dan lineage benar | ❌ belum (v7 versi kasar 0,4917) |
| Spending 112–224 / semantic rate | ✅ diuji di v4 (tidak konsisten) |
| Rasio DPD exposure-matang, delta provider sama | ❌ belum diuji di model (screen 0/15) |
| 9 fitur selected-history | ✅ 16 Okt v21: −0,0011 OOF LGB, null |
| Simpan per-fold/per-seed preds, bobot, hash, versi | ⚠ sebagian (oof_all.parquet dengan ID sejak v7; manifest di v10–v13; bobot v8 direkonstruksi 16 Okt) |
