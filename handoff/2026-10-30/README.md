# Serah terima 28–30 Oktober 2026: Markov, model per segmen dan MLP

Ditulis 30 Oktober 2026 siang untuk teman satu tim yang melanjutkan. Semua angka KS di bawah adalah pooled OOF KS
pada 70.000 baris train (protokol validation_v2: StratifiedKFold 5, inner 1/7 early stop, refit ×1,15), kecuali
yang ditandai public.

## 1. Posisi sekarang dan kesimpulan sementara

**Kesimpulan sementara (dipilih user): kandidat dengan OOF tertinggi di bawah aturan terkunci, yaitu recipe C.**

| | File | OOF KS | Public |
|---|---|---|---|
| OOF tertinggi (aturan terkunci) | `submissions/v75/v75_recipe_C.csv` | **0,52861** | 0,56374 |
| Public tertinggi | `submissions/v72/v72_six_markov_3seed.csv` | 0,52580 | **0,56822** |

Recipe C dibuat dari tiga bagian:

- **Bagian pohon (bobot 2/3):** rata-rata rank empat model pohon yang dilatih terpisah per segmen `has_history`.
  - `lgb_seg`: LightGBM v22;
  - `xgb_seg16`: XGBoost depth 3, 16 seed;
  - `lgb_s_seg`: LightGBM depth 3;
  - `cat_s_seg`: CatBoost depth 4.
- **Bagian mulus (bobot 1/3):** rata-rata rank MLP 10 seed dan MLP per segmen 5 seed.
- **Input:** semua fitur (`features.parquet` 237 kolom + blok Markov `f251`).

Hal yang perlu diketahui sebelum memakai C sebagai final:

- **Di public, urutan file justru terbalik dari OOF.** Setiap file yang bergerak menjauh dari v72 mendapat public
  lebih rendah, walaupun OOF-nya lebih tinggi (tabel di bagian 2).
- **Peluang hasil C terjadi karena kebetulan cuma sekitar 3%.** Kalau keunggulan OOF C (+0,0028 atas v72) nyata,
  peluang C kalah ≥ 0,0045 di 9.000 baris public kira-kira 3% (bootstrap 2.000 subset 9k dari OOF).
- **OOF tertinggi kemungkinan optimis.** Kita memilihnya dari puluhan varian, jadi ada efek pemenang kebetulan.
  - Contoh: dengan MLP 5 seed, varian "4 pohon segmen + MLP" dapat +0,0025 atas `eight_seg`.
  - Dengan MLP 10 seed (aturan terkunci), angkanya turun ke +0,0011.
- **Nilai akhir = 45% public + 45% private + 10% notebook.**
  - Public file yang dipilih ikut dihitung, jadi v72 unggul pasti ~0,002 poin di komponen public.
  - C baru menang total kalau unggul > 0,0045 di private.
- **C belum punya notebook Kaggle.**
  - Notebook yang sudah terbukti jalan end-to-end adalah `notebook/v72.ipynb` (kernel
    `athillazaidan/logika-v72-trees-markov`).
  - Anggota C dilatih lokal di Mac (pohon di CPU, MLP di MPS).

Deadline notebook: **30 Oktober 2026 23.59 WIB**, penalti 1 poin per menit telat. Format nama:
`NomorPeserta_TeamName_Notebook`.

## 2. Semua submission jalur pohon + Markov

| Tanggal | File | Isi | OOF | Public |
|---|---|---|---|---|
| 28 Okt | v71 | 6 GBDT + Markov, seed 7/2026 (lokal) | n/a | 0,56649 |
| 28 Okt | v71b | v71 50/50 dengan Kumo goal054 | n/a | 0,56503 |
| 28 Okt | v71c | v71 + offset logit per bulan (eda/264) | n/a | 0,56442 |
| 29 Okt | **v72** | 6 GBDT + Markov, seed 42/7/2026 (kernel Kaggle) | 0,52580 | **0,56822** |
| 29 Okt | v73 | v72 + 5 seed dari kernel v73, total 8 seed | 0,52577 | 0,56719 |
| 30 Okt | v74 `eight_seg` | 6 + lgb_x + xgb_d8 + lgb_seg + xgb_seg, 8 seed | 0,52706 | 0,56533 |
| 30 Okt | v74 `xgb_seg16` | single XGBoost per segmen, 16 seed | 0,52783 | 0,56225 |
| 30 Okt | **v75 recipe C** | lihat bagian 1 | **0,52861** | 0,56374 |

File ada di `submissions/` (lokal, tidak di git). Sebagai pembanding, catatan merge_R2DK teman (25 Okt) punya pooled
OOF 0,52886. Angka ini setara C, tetapi validasinya berbeda.

## 3. Cara mereproduksi recipe C

Semua dijalankan dari `eda/` dengan `../.venv/bin/python`. Output ke `results/v73/...` (lokal, tidak di git).

1. **Fitur.** `common.features()` membangun `data_pq/features.parquet` (cache), lalu `251_repayment_markov.py`
   membuat `data_pq/f251_repayment_markov.parquet`.
2. **Pohon per segmen** (`280_seg_members.py`, 8 seed: 42, 7, 2026, 1, 11, 101, 777, 31337, sekitar 30–75 menit
   per model di M5):
   - `lgb_seg 3`, `xgb_seg 3`, `lgb_s_seg 3`, `cat_s_seg 3`;
   - `xgb_seg 3 b`: 8 seed tambahan, sehingga `xgb_seg16` = gabungan kedua run.
3. **MLP** (`291_seq_model.py`, GPU MPS, sekitar 2–3 menit per seed):
   - `291_seq_model.py <seed> 30 mlp` untuk seed 42, 7, 2026, 1, 11, 101, 777, 31337, 3, 5;
   - `291_seq_model.py <seed> 30 mlpseg` untuk seed 42, 7, 2026, 1, 11;
   - butuh cache sekuens dari `290_seq_prep.py`, walau mode mlp tidak memakai tabel sekuensnya.
4. **Blend.** `295_final_recipe.py C` menulis `submissions/v75/v75_recipe_C.csv`.
   - Script ini juga membandingkan dengan `eight_seg`, jadi butuh OOF kernel v72/v73 dan `279_extras_local.py`.
   - OOF kernel diunduh dengan `kaggle kernels output athillazaidan/logika-v72-trees-markov -p results/v72k_out`
     dan `kaggle kernels output athillazaidan/logika-v73-trees-markov-seeds -p results/v73k_out`.

Untuk notebook Kaggle, pola embed kode ada di `notebook/build_v72.py` (varian v72/v73). Waktu di Kaggle
kira-kira 4× lebih lambat dari Mac untuk LightGBM, jadi jumlah seed mungkin perlu dikurangi.

## 4. Apa yang dipelajari tentang data (generator)

- **Satu risiko laten per customer, bergeser pelan.** Korelasi tingkat telat dua pinjaman customer yang sama turun
  dari 0,47 (jarak < 90 hari) ke 0,26 (540–730 hari) (eda/265).
- **Risiko itu ditulis ke beberapa kanal dengan noise independen:**
  - **Repayment:** telat bayar berbentuk rantai Markov. Peluang telat naik dengan urutan cicilan (0,20 → 0,37) dan
    nominal. Blok Markov f251 memberi +0,003 public.
  - **Harga belanja:** semua kategori bergeser sama (bobot faktor 1,00). `pp_mean` sudah hampir optimal
    (eda/268).
  - **Lonjakan belanja menjelang keputusan:** rasio order calon gagal bayar vs lancar 2,8× di hari terakhir,
    menurun ke 1× sekitar hari ke-90 (eda/285).
  - **Arus kas rekening:** ambruk di ~4 bulan terakhir untuk calon gagal bayar.
- **Bunga adalah rate card dari profil yang terlihat ditambah noise.** Residualnya KS 0,02 (eda/283).
- **Test adalah split acak:** AUC adversarial 0,501, bulan sama. Tidak ada lantai label noise: 0/591 bad di 1% skor
  terbawah.
- **Kelompok tanpa riwayat** adalah tepat 30.000 aplikasi (30% di train dan di test). Kelompok ini hanya punya
  aplikasi, biro dan e-commerce.
- **Tabel sudah dipotong panitia di `decision_datetime`.** Tidak ada baris setelah keputusan.

## 5. Yang sudah dicoba dan tidak menaikkan (jangan diulang tanpa alasan baru)

| Ide | Script | Hasil |
|---|---|---|
| Observed−expected telat bayar | 266 | single LGB −0,0014 |
| Screen blok lama di segmen tanpa riwayat | 267 | chi2 < 4 |
| Estimator harga shrink / faktor | 268 | sama dengan `pp_mean` |
| GAM aditif sebagai anggota | 269 | KS blend ≈ 0 |
| Analisis error: residual booster | 270 | berhenti di 1–2 pohon; kalibrasi segmen rata |
| Fitur spesifikasi 638 kolom teman (87 kolom yang belum ada) | 271 | residual +0,0000; single LGB −0,0017 |
| Turunkan bobot label mencurigakan | 272 | −0,002 s/d −0,0035 |
| Tuning LightGBM / XGBoost | 273, 274 | tidak ada konfigurasi di atas v22 |
| Blend logit / bobot dipilih per fold | 275 | tidak lebih baik dari rata-rata rank |
| LGB monoton, DART, RF, ET | 276 | netral / tidak selesai |
| Model bantu laten repayment + stacking tahap 2 | 277, 278 | gain sama untuk blok kontrol (artefak stacking) |
| Bobot fokus titik potong KS | 282 | −0,001 s/d −0,0046 |
| Residual keputusan lender | 283 | KS residual bunga 0,02 |
| kNN target mean / densitas | 284 | chi2 ≈ 0 |
| Pembobotan likelihood-ratio order | 286 | lebih lemah dari `ec_accel_30d` |
| Indeks memburuk lintas kanal | 287 | chi2 < 4 |
| Perubahan skor biro vs aplikasi lama | 288 | LR chi2 14, tetapi single LGB negatif di 3 seed |
| Blend per segmen dipilih nested | 289 | −0,00055 per fold |
| Model event mentah (sekuens) | 290–292 | 0,5127 sendiri; hibrida 0,5233; blend netral |
| LightGBM v10 nested tuning | 293 | 0,52368; menurunkan blend |
| Pseudo-label | riset (`reports/`) | literatur dan solusi Kaggle: ≈ 0 pada 70k label |

Yang positif kecil: anggota beragam (lgb_x 0,5258, xgb_d8 0,5260), model per segmen, dan pasangan pohon per segmen
+ MLP (+0,001 s/d +0,0016 OOF dengan aturan terkunci). Tidak satu pun terlihat di public.

## 6. Pelajaran metodologi

- **Single LGB satu seed punya noise ~0,003 per fold.** Butuh ≥ 3 seed sebelum percaya selisih < 0,003.
- **Chi2 tinggi di screen linear (256) belum tentu jadi gain di pohon (288).** Pakai sebagai filter, bukan putusan.
- **Stacking tahap 2 di atas OOF blend selalu memberi ~+0,0007 per fold,** bahkan untuk blok tidak berguna. Selalu
  jalankan blok kontrol.
- **Memilih juara dari puluhan varian OOF menghasilkan angka optimis.** Kunci aturan (commit) sebelum melihat hasil.
  Contoh: `research/autoresearch/` (protokol di-commit sebelum hasil).
- **Public 9k tidak bisa membedakan file yang selisih OOF-nya ≤ 0,002.** Dua file yang hampir identik (v72 vs v73)
  bergeser ±0,001.
- **Polars 2.0 `join` tidak menjaga urutan baris.** Pakai `maintain_order="left"`.

## 7. Langkah berikutnya yang disarankan

1. **Putuskan file final.** Pilihannya v72 (public tertinggi, notebook siap) atau C (OOF tertinggi, notebook belum
   ada). Hitungan nilai di bagian 1 memihak v72 kecuali tim yakin C unggul > 0,0045 di private.
2. **Kalau C:** bangun notebook Kaggle dari pola `build_v72.py` dan uji satu kali jalan sebelum 23.59 WIB.
3. **Kalau ada waktu riset,** satu-satunya uji yang masih bernilai informasi adalah lockbox:
   - sisihkan 20% train;
   - ulang pemilihan (6 model vs pohon segmen + MLP) di 80%;
   - ukur di lockbox.
   Ini memberi tahu apakah gain segmen + MLP nyata atau ilusi OOF.

Detail lengkap: `research/autoresearch/findings.md`, `research/autoresearch/research-log.md`, dan riset pseudo-label
`reports/Pseudo labeling untuk GBDT tabular.md`.
