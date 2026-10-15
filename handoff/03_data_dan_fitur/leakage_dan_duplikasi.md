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
