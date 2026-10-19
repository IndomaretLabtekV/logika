# Ide lain yang belum diuji (ekspektasi rendah; urut dari yang paling masuk akal)

| Ide | Alasan | Kenapa belum / risiko |
|---|---|---|
| Fine-tune Kumo **semua parameter** dengan optimizer 8-bit / SGD, atau di GPU ≥ 16 GB | blok ICL saja null (lr 1e-5, 1e-4); row-embedding (10M param) belum pernah dilatih | OOM di 8 GB dengan AdamW; bitsandbytes tidak terpasang; kuota Kaggle habis |
| VIP-COP tingkat **baris** untuk Kumo | paper melaporkan gain terbesar dari seleksi sampel | 48k item butuh anggaran evaluasi ≫ 1.000 |
| Kumo dengan context per query (retrieval, LoCalPFN/TabDPT style) | paper menyebut sebagai alternatif | TabDPT (retrieval) gagal/lemah di zoo; batching mahal |
| Augmentasi historis (81.702 pinjaman lama berlabel sebagai baris latih) di protokol bersih | EDA `eda/83` +0,0027 (5/6) di protokol lama | v8 ≈ 0; domain shift (pp, ec tidak transfer); learning curve datar |
| Refit pohon di seluruh 70k untuk test | lebih banyak data | learning curve datar (≤ +0,0005) |
| KumoRelational di GPU besar | ICL relasional dari tabel mentah | v14 set-encoder null; 4,36 juta item |
| Model pohon dengan fitur yang dipilih bersarang dari pasangan/peer | anggota dangkal naik | bersarang −0,0005, keluarga null |
Sudah pasti null (jangan diulang tanpa alasan baru): lihat `08_pelajaran/yang_tidak_efektif.md`.
