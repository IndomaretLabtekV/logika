# Leakage dan duplikasi

## Leakage
| Cek | Hasil |
|---|---|
| Baris tabel bantu setelah cutoff | 0 di semua tabel (dataset sudah dipotong). Kode tetap memfilter |
| Customer overlap train/test | 0 |
| ID (application_id, customer_id) sebagai sinyal | tidak ada. ID opaque dan tidak dipakai sebagai fitur |
| Urutan baris CSV | tidak bocor (dicek v9) |
| Aplikasi terpilih ada di previous_applications | 0 |
| Previous apps yang diputuskan setelah cutoff | 0 |
| Referensi harga global (v5) | **masalah**: memakai transaksi customer lain setelah cutoff. Diperbaiki di v6 (expanding per bulan) |
| `nn_prep` fit di seluruh train (v5) | pengaruh distribusi validasi, bukan kebocoran label. Diperbaiki di v6 (fit per fold). Efek nilai: median 0,00125 |
| Seleksi fitur FM memakai label query probe EXAONE (v5) | diperbaiki di v6 |
| Augmentasi historis v8 | memasukkan baris historis milik customer validasi/test di setiap fold. Ini **transduktif** (bukan label target), tapi protokolnya tidak bersih. Fitur historis juga tidak bisa direkonstruksi eksak (lihat `04_riwayat_versi/v8.md`). Dibuang di v9 |
| Tes injeksi masa depan | lulus di `eda/12`, `46`, `53`, `81`, dan di setiap run notebook |

Prinsip yang dipegang: snapshot repayment adalah keadaan **pada selected cutoff**. Jangan di-backdate seolah-olah itu status pada due_date.

## Duplikasi
- **`order_total` diulang di setiap item.** Kalau dijumlah naif, total spending menggelembung **1,7397×** (1,28 T → 2,23 T IDR). Selalu agregasi per `order_id` dulu.
- **2.388 baris item** isinya identik kalau `order_item_id` dibuang. **Jangan dihapus.** Menghapusnya merusak 1.194 order, dan sebelum dihapus Σ item_total = order_total persis. Itu item sah dengan ID berbeda.
- Tidak ada duplikat untuk: installment_id, account customer-month, item_id, profil penuh tanpa ID, maupun baris fitur penuh tanpa ID.
- Near-duplicate atau twin customer lintas split: tidak ditemukan (`eda/10`, ad hoc v6, `eda/90`).
- Histori (v8): 81.702 pinjaman historis dari 50.510 customer (sampai 7 per customer). Ini observasi yang saling terkait, bukan duplikat. Tapi bobot per customer jadi tidak seimbang (bobot 0,5 × 7 = 3,5).

## Kesimpulan
Tidak ada leakage atau duplikasi yang bisa "dieksploitasi". Data bersih dan sintetis. Hal yang perlu dijaga hanya grain order dan aturan cutoff.

## Tambahan 17–18 Okt
| Cek | Hasil |
|---|---|
| Adversarial validation train vs test (`eda/104`, LightGBM 5-fold) | AUC 0,4987 → split acak, StratifiedKFold tepat |
| account_behavior memuat bulan berjalan? | bulan terakhir per customer selalu tepat 1 bulan sebelum bulan keputusan (70.000/70.000) → bersih |
| Bureau memuat pinjaman lender sendiri? | tidak (outstanding korelasi 0,14; delinquency 0,01–0,02) → tidak ada konstruksi "eksternal = bureau − internal" |
| Kebocoran lewat seleksi fitur FM | `fm_select` dihitung per train fold (validation_v2) |
| Pseudo-label / fitting distribusi test | tidak dipakai (dilarang); Kumo-70k memakai train saja sebagai context, test hanya query |
| Bobot blend | dipilih cross-fit (4 fold → fold ke-5), tidak pernah di baris yang dinilai; tidak disetel dari public |

Desain fitur dari EDA di seluruh label train tidak bisa dihindari; karena itu fitur baru wajib dikonfirmasi lewat model
cross-fit, bukan dari skor EDA.

## Tambahan 19 Okt
| Cek | Hasil |
|---|---|
| Baris setelah cutoff per tabel, **train vs test terpisah** (`eda/139`) | 0/0 di previous, repayment (due_date dan last_payment_datetime), bureau (available_datetime), e-commerce, bulan akun belum lengkap |
| Jendela ekstraksi train vs test | identik: median jarak catatan terakhir ke cutoff previous 177,2/178,1 hari, repayment 15,29/15,25, bureau 0,05/0,06, e-commerce 11,11/11,08 |
| `previous_applications` memuat aplikasi terpilih? | tidak (0 application_id sama, 0 keputusan dalam ±1 hari dari keputusan terpilih) |
| Semua loan di repayment ada di previous approved? | ya (0 loan tak terdaftar) |
| Kebocoran seleksi fitur (winner's curse) | **terjadi** di `eda/145` (30 dari 7.080 dipilih dengan residual label semua baris → +0,0036 palsu); versi bersarang `eda/146` −0,0005. Aturan: seleksi fitur dari banyak kandidat wajib bersarang per outer fold |
| Normalisasi lintas customer di kohort bulan keputusan | memakai data customer lain yang keputusannya lebih lambat di bulan yang sama → tidak patuh cutoff; versi patuh = persentil per bulan akun di seluruh populasi (`eda/138`) |
| Snapshot cutoff mundur (`eda/142`) | patuh by construction (hanya data lebih lama) |
