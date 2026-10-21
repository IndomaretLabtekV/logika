# Reverse engineering pola sintetis dan jitter

Data sangat konsisten dengan generator sintetis, tetapi kode sumber generator belum tersedia. **Seed RNG asli belum ditemukan.** Ada aturan yang dapat direkonstruksi persis dari nilai observasi, dan ada parameter noise/laten yang hanya dapat diestimasi. Keduanya dibedakan di bawah.

## Aturan dan pola observasi

| Bagian | Temuan | Status |
|---|---|---|
| Usia | Integer 23–60; uji kesesuaian uniform p=0.7512 | Konsisten dengan `rng.integers(23,61)`; bukan bukti RNG tertentu |
| Region | Lima kategori dengan proporsi sekitar 20% | Konsisten dengan sampling uniform |
| Nominal pinjaman terpilih | Semua kelipatan Rp100.000 | Grid terkonfirmasi |
| Cicilan terpilih | `ceil(P*(r/12)/(1-(1+r/12)^(-term)))` | Cocok 100.0% dari 100.000 aplikasi |
| Harga per unit | Kelipatan Rp1.000, kemudian dikalikan quantity | Cocok 100.0% dari 4,359,817 baris |
| Harga minimum | Rp5,000; ada massa pada minimum | Konsisten dengan clipping/floor, bukan jitter yang bebas dipulihkan |
| Saldo awal bulan | Sama dengan saldo akhir bulan sebelumnya | Cocok 100.0% dari pasangan bulan |
| Saldo akhir bulan | opening + inflows − outflows | Cocok 100.0% |
| Skor biro | Integer; timestamp laporan memiliki detik nol | Grid terkonfirmasi |

Delay aplikasi-ke-keputusan cocok dengan campuran tiga uniform:

| Rentang jam | Proporsi observasi | Dugaan probabilitas generator |
|---|---:|---:|
| 0,2–3 | 0.44624 | 0,45 |
| 6–30 | 0.35268 | 0,35 |
| 36–120 | 0.20108 | 0,20 |

Tidak ada observasi di rentang 3–6 atau 30–36 jam. Rentang adalah fakta pada data; probabilitas 0,45/0,35/0,20 tetap hipotesis. Delay uniform dapat direkonstruksi hanya sampai interval pembulatan satu detik. Selisih cicilan di bawah Rp1 berasal dari **pembulatan ke atas**, bukan noise RNG independen.

## Estimasi jitter dan pola laten

Harga dimodelkan sebagai:

`log(unit_price_ij) ≈ intercept_kategori + preferensi_customer_i + efek_order + noise_item`.

Kategori dipisahkan dengan two-way fixed effects tanpa label target. Selisih antaritem dalam order yang sama membatalkan efek customer dan order. Estimasi SD noise item pada skala log adalah **0.6542**, sehingga parameter seperti `sigma=0.65` masuk akal sebagai hipotesis. SD efek bersama order sekitar 0.0607; SD preferensi customer sekitar 0.2241. Selisih item memiliki excess kurtosis 0.0114 dan jarak KS terhadap normal terfit 0.0022. Ini mendukung bentuk noise Gaussian dalam log harga, tetapi tidak mengidentifikasi algoritma RNG. Harga asli sebelum rounding biasanya berada dalam interval ±Rp500 dari unit price yang tersimpan, kecuali pada floor.

Parameter preferensi customer diestimasi dengan generalized least squares dan empirical Bayes; ketidakpastiannya juga disimpan. Korelasi estimasi dari dua kelompok order independen adalah 0.6687. Intercept kategori dan lokasi efek customer mempunyai kebebasan skala bersama: centering mean customer pada nol adalah konvensi, bukan pemulihan baseline harga asli secara unik.

Skor biro dibandingkan **pada customer dan aplikasi yang sama**. Dengan asumsi noise provider independen, `Var(A-B)=Var(error_A)+Var(error_B)`. Estimasi SD noise adalah:

```json
{
  "Bureau_A": 22.98196855073303,
  "Bureau_B": 23.03988187170871,
  "Bureau_C": 22.9697369073074
}
```

Nilainya sangat dekat **23 poin** dan offset provider kurang dari 0,1 poin. `score = round(latent_score + Normal(0,23))` adalah hipotesis yang sesuai, bukan kode sumber terkonfirmasi. Estimasi kemampuan laten memakai precision weighting dan prior antarcustomer; perubahan dari waktu ke waktu, clipping dan dependensi provider dapat mengganggu asumsi tersebut.

Untuk income-like credits, `Var(log income_t − log income_(t-1))/2` menghasilkan estimasi SD noise bulanan berikut:

```json
{
  "salaried": 0.2590129938849792,
  "self_employed": 0.34435424062467596,
  "variable_income": 0.41820045150346913
}
```

Residual income memiliki tail lebih tebal daripada Gaussian, terutama salaried. Angka di atas adalah **skala noise efektif**, yang dapat mencampur jitter biasa, shock, tren dan heterogenitas customer. Geometric mean dan selisih terhadap declared income dipakai sebagai sinyal laten; belum terbukti sama dengan pendapatan asli generator.

Frekuensi keterlambatan historis memakai prior beta-binomial yang dipelajari dari customer training **tanpa target default**. Kekuatan prior terestimasi sekitar 2,19 untuk DPD≥1, 2,19 untuk DPD≥7, dan 2,60 untuk DPD≥30. Ini lebih lemah daripada smoothing 6/24 yang dipakai sebelumnya. Dependensi pembayaran serta censoring membuat posterior hanya pendekatan, bukan nilai probabilitas asli generator.

![Distribusi noise dan efek laten](../figures/synthetic_noise_patterns.png)

## Pencarian RNG yang sudah diuji

PCG64 dan MT19937/RandomState diuji pada 10,006 seed per keluarga: 0–9.999 ditambah enam kandidat tanggal/kompetisi. Uji mencari 12 usia pertama pada 512 nilai stream, dengan offset dan stride 1–16. **Tidak ditemukan kecocokan.** Uji template hash ID sebanyak 6,000 evaluasi juga tidak menghasilkan kecocokan tiga baris. Sembilan hipotesis RNG per customer menggunakan potongan ID atau indeks profil, masing-masing 128 offset pada 1.000 customer, tidak menghasilkan pola yang mereproduksi seluruh usia. ID dipakai hanya untuk audit hipotesis seed, bukan fitur numerik model.

ID juga diperiksa sebagai **output RNG 64-bit atau delapan byte langsung** pada PCG64, PCG64DXSM, NumPy RandomState dan Python random. Selain prefix 512 draw pada 10.006 seed, 15 seed umum diperiksa hingga satu juta draw, termasuk Philox dan SFC64. Tidak ada kecocokan pertama yang persis. Uji state MT19937 tanpa mengetahui seed memakai 624 word untuk membentuk state dan 1.376 word berikutnya untuk verifikasi; empat interpretasi byte/word ID menghasilkan **nol kecocokan**. Positive control pada output MT dari Python dan NumPy berhasil. Uji ini memerlukan output mentah yang berurutan dan tidak menyingkirkan MT dengan hashing, shuffle, atau pemanggilan RNG yang diselingi. [Struktur state MT19937 dalam dokumentasi NumPy](https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html).

Hasil negatif ini terbatas pada hipotesis yang diuji. Generator dapat memakai seed lain, library lain, beberapa stream, row shuffle, atau pemanggilan acak yang diselingi jumlah riwayat berbeda. Reproduksi NumPy memerlukan BitGenerator, seed, urutan pemanggilan, argumen dan lingkungan yang sesuai. [Dokumentasi resmi NumPy](https://numpy.org/doc/stable/reference/random/compatibility.html). Rounding dan clipping juga menghapus sebagian informasi RNG. Distribusi Gaussian saja tidak membedakan PCG64 dari MT19937.

`synthetic_generator_prototype.py` menyediakan **generator pendekatan** untuk delay dan harga. Seed demo dipilih oleh kita; bukan seed asli dataset. Prototype mereproduksi beberapa pola marginal dan varians, tetapi tidak merekonstruksi seluruh dataset, jadwal RNG, ataupun label test.

## Dampak terhadap prediksi

Sebanyak 21 fitur laten/ketidakpastian dibuat; 20 bervariasi pada train. CatBoost dan spline diuji pada **tiga fold yang sama** dengan versi sebelumnya (seed fold 8247; seed model 2026). Tabel ini tidak dibandingkan langsung dengan KS sepuluh fold karena ukuran fold berbeda.

| Kandidat | Bobot model baru | KS OOF | Rata-rata KS fold |
|---|---:|---:|---:|
| version5 | 0.00 | 0.525285 | 0.527524 |
| reverse_cat5 | 0.25 | 0.524847 | 0.526337 |
| reverse_cat5 | 0.50 | 0.523806 | 0.525688 |
| reverse_cat5 | 1.00 | 0.520312 | 0.521919 |
| reverse_spline | 0.25 | 0.525505 | 0.527003 |
| reverse_spline | 0.50 | 0.524651 | 0.525474 |
| reverse_spline | 1.00 | 0.517505 | 0.519495 |

Gabungan spline 25% menaikkan KS gabungan hanya sekitar 0,00022 tetapi menurunkan rata-rata fold; gabungan CatBoost menurunkan keduanya. **Belum ada perbaikan konsisten**, sehingga submission versi 5 dan file default tidak diubah. Pemilihan fitur/model/bobot memakai training labels yang sama, sehingga estimasi CV mengandung optimisme seleksi. Label test tidak digunakan.

## Reproduksi

Notebook [07 — Reverse engineering](../../notebook/07_reverse_engineering_rng.ipynb) berisi EDA, probe seed, inferensi jitter, simulasi pendekatan dan review validasi dengan kode di sel. Paket analisis sama dengan requirements-optimization dan requirements-notebook.

```bash
python scripts/reverse_rng_probe.py --seed-limit 10000
python scripts/reverse_synthetic_patterns.py
# Memerlukan fitur/prediksi tiga fold versi 5:
python scripts/validate_reverse_features.py
python scripts/report_reverse_engineering.py
```

Audit rinci ada pada `pattern_audit.json`, `rng_probe.json` dan `validation_comparison.json`. Nilai per customer berada di `data/processed/application_features_reverse.parquet` dan dikecualikan dari Git.
