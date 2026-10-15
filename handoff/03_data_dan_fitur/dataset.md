# Dataset: struktur dan semantik

Skema lengkap ada di `data/DATA_DICTIONARY.md`. Bagian ini mencatat **semantik yang ditemukan lewat EDA**.

## Tabel
| Tabel | Baris | Kunci | Catatan |
|---|---|---|---|
| train_applications | 70.000 | application_id, customer_id | ada target; selected loan **selalu tenor 12 bulan** |
| test_applications | 30.000 | sama | tanpa target |
| customer_profiles | 100.000 | customer_id | snapshot **saat ini** (saat selected decision), tidak ada riwayat perubahan |
| previous_applications | 201.984 | application_id, loan_id | approved 159.323, rejected 24.269, cancelled 18.392; tenor 6/12/18 masing-masing sekitar 67k |
| repayment_history | 1.361.341 | installment_id, loan_id | **snapshot pada selected cutoff**; `last_payment_datetime` = alokasi terakhir sebelum cutoff |
| account_behavior | 1.084.418 | customer_id + month | hanya bulan yang lengkap; tersedia mulai awal bulan berikutnya |
| bureau_observations | 797.563 | bureau_observation_id | 3 provider (A, B, C); pakai `available_datetime` ≤ cutoff, bukan `report_datetime` |
| ecommerce_transactions | 4.359.817 item / 2.986.855 order | order_item_id | `order_total` diulang di setiap item |

## Cakupan
- 30% customer (21.000 train) **thin-file**: tidak punya previous applications, repayment, maupun account. Bad rate-nya 11,7%, sedangkan yang punya history 14,4%. Proporsinya sama di train dan test.
- Bureau dan e-commerce: cakupan 100%. Repayment sekitar 66%, previous dan account 70%.
- Semua tabel bantu **sudah dipotong di cutoff** (0 baris setelah cutoff). Kode tetap memfilter secara eksplisit.

## Identitas aritmetika (semua eksak; `eda/23`, `eda/92`, review v6)
- `closing = opening + inflows − outflows` (100%), `opening = closing bulan sebelumnya`, tidak ada gap bulan.
- `utilization = revolving_balance / revolving_limit`.
- `order_total = Σ item_total`. `item_total` dan unit price berada di grid 1.000 (100%). Unit price kontinu, lognormal per kategori (sd log sekitar 0,69), tanpa katalog harga.
- Cicilan = anuitas eksak (dibulatkan ke atas); cicilan terakhir menyesuaikan pembulatan. `due_date = decision + k bulan`.
- Hari pembayaran berupa bilangan bulat. Untuk baris lunas, max_dpd = selisih tanggal. Untuk baris belum lunas, dpd = floor(cutoff − due).
- `loan_amount` selalu kelipatan 100.000. Income hampir tidak terkuantisasi.
- Kesimpulan: **tidak ada laten tersembunyi di identitas-identitas ini**. Residual yang tidak eksak (path average balance, share partial payment, approved/requested) tidak membawa sinyal: chi2/df ≤ 1,44 terhadap v8.

## Semantik penting
- Keterlambatan terkuantisasi di sekitar 30/61/91 hari (bulanan). Pada bad, median hari telat (di antara cicilan yang telat dibayar) 61; pada good 31.
- Bureau: provider simetris. Skor antar provider berkorelasi 0,815. Inquiry dan delinquency antar provider hampir independen.
- Rate tier (16/20/24/28/32%) adalah fungsi kuat dari skor bureau: skor rata-rata turun sekitar 40 poin per tier, dan skor bin menjelaskan 55% varians rate.
- Aplikasi → keputusan: event di antara application dan decision punya bad rate 0,247 vs 0,13, tapi sinyalnya sudah tertangkap fitur yang ada.
- Unit price "premium": rata-rata z log unit price per kategori milik customer membentuk **pengali harga di level customer** (ICC sekitar 0,106; reliabilitas dengan sekitar 43 item sekitar 0,84). Customer yang membeli lebih mahal dari rata-rata kategori lebih berisiko. Ini sinyal terkuat (lihat `temuan_eda_utama.md`).
- Harga naik pelan dari waktu ke waktu (mean log unit price 11,75 di 2023-02 → 11,95 di 2025-12). Karena itu referensi harga harus *expanding per bulan keputusan*, tidak boleh global (lihat `fitur_dan_preprocessing.md`).
