# Kesalahan, bug, dan overclaim yang pernah terjadi

## Bug teknis di notebook
| Versi | Bug | Dampak | Perbaikan |
|---|---|---|---|
| v3 | LimiX-2 OOM, retry di query penuh; guard waktu baru dicek setelah fold selesai | hang lebih dari 3 jam, run sekitar 5 jam tidak selesai | LimiX dibuang; `budget_min` dan cap waktu per fold |
| v5 | EXAONE: FS internal memicu "No available kernel" (104 kolom > 100); patch `_sdpa_backend` privat tidak menutup jalur FS | EXAONE tidak jalan | ≤ 100 kolom + `mem_efficient` (v6) |
| v5 | Probe EXAONE: `stratified_subsample` menaruh positif di depan, sehingga query 0 positif; seleksi fitur memakai label query | probe tidak bermakna | split stratified disjoint dan acak (v6) |
| v6 | Budget EXAONE dari satu titik probe (mencampur overhead) + minimum context buatan 10k | EXAONE di-skip padahal jalan | probe 2 titik (v7) |
| v7 | `run_fm` hanya mengenali "out of memory"; EXAONE melempar "cannot fit" | EXAONE dilewati seluruhnya | `MEMORY_ERRORS = ('out of memory','cannot fit')` (v8) |
| v7 | `fm_matrix` mengubah kategori jadi `cat.codes` float; Causilo dan DFS kehilangan tipe kategori | Causilo sedikit lebih lemah | pandas categorical (v8) |
| v8 | TabICL `whole_query=True` mengirim sekitar 44k query sekaligus; mengecilkan context tidak membantu | kernel mati sekitar 4 jam, model sesudahnya hilang | TabICL dimatikan; safety-net submission |
| v8 | Notebook lokal ≠ notebook yang di-run (TabICL on/off, safety-net) | provenance CSV membingungkan | simpan notebook run di `results/vN/` |
| v1–v5 | Tie timestamp (170 order, 94 bureau, due_date paralel) membuat sort nondeterministik | fitur bisa berubah antar run | tiebreak ID |
| v5 | Referensi harga global (memakai transaksi customer lain di masa depan) | pelanggaran semangat cutoff | expanding per bulan (v6) |
| v5 | nn_prep fit di seluruh train | pengaruh distribusi validasi (kecil) | fit per fold (v6) |

## Kesalahan analisis (sebagian besar ditemukan Astra)
- **Bin kuantil menyatukan 0 dengan nilai positif langka** di `common.qbin` / screening lama. Akibatnya uji fitur bermassa-nol tidak valid. Diperbaiki di `common9.safe_bins` (v9). Script < 90 tidak dijalankan ulang.
- **Overclaim "ceiling"** di `eda/69`. Itu self-consistency di dunia asumsi model kita, bukan batas dataset.
- **Overclaim konversi offset**: "top-1 public ≈ OOF 0,541" tidak defensible.
- **"AUC monoton dengan KS"** ditulis di markdown v7. Salah: ada 5 pasangan model yang arahnya berlawanan.
- **`calibrate` lalu residual test di label yang sama** disebut "validasi". Itu bukan holdout independen.
- Hasil negatif `eda/72` (sequence) diklaim terlalu luas. Yang diuji hanya satu representasi.
- Augmentasi historis (v8) disebut "as-of reconstruction". Kenyataannya tidak eksak (ledger ambigu, profil saat ini), dan protokolnya transduktif.
- Prioritas EXAONE terlalu optimis di v6–v7 berdasarkan probe kecil yang tidak sebanding.

## Kesalahan komunikasi
- Terlalu sering bilang **"final" / "final picks"**. User marah: masih ada 15 hari. Bahas pemilihan akhir hanya kalau diminta.
- Kadang terkesan "menyerah" ("data sudah jenuh"). Lebih baik: sebutkan apa yang **belum** diuji dan tawarkan eksperimen konkret, sambil tetap jujur soal ekspektasi.
- Jawaban "submit / jangan" sering datang terlambat setelah analisis panjang. User butuh keputusan di kalimat pertama.

## Kesalahan proses
- Run v8 menumpuk terlalu banyak eksperimen baru dalam satu notebook panjang.
- Hyperparameter GBDT/TabM **tidak pernah di-tune** sepanjang 9 versi. Fokus terlalu lama di EDA fitur setelah fitur terbukti jenuh.
- Bobot blend, hash, dan versi paket tidak disimpan konsisten per run (saran Astra).
