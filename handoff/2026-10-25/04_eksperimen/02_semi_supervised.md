# Eksperimen 2 — Semi-supervised (diizinkan panitia 20 Okt)

Desain ditulis sebelum skor di `docs/integration/protokol_context_ssl_24okt.md` (beda dari v49 threshold-pseudo,
185 mixup, v57/v58 soft-label GBDT).

## 2.1 S3 — pseudo-label tersampel sebagai context Kumo (`eda/222`, inner f0)
- Teacher: Kumo nested_small (context 48k inner-fit) memprediksi 30k test. Pseudo label `y ~ Bernoulli(p_teacher)`
  untuk **semua** 30k test (bukan threshold), 2 draw independen dirata-rata rank; student = nested_small e8 dengan
  context fit + 30k pseudo. Kontrol: baseline tanpa SSL; permutasi label pseudo (jumlah positif sama).
- Hasil (KS inner-stop 8k): d0 0,49885, d1 0,49405, rata-rata 0,49822; baseline 0,50047; permutasi 0,49103.
  **pseudo − base −0,0022** (CI −0,0093…+0,0022, P>0 0,13); pseudo − perm +0,0072 (P 0,93).
- Kesimpulan: label semu membawa informasi (mengalahkan permutasi) tetapi menambah 30k baris pseudo **merugikan**
  dibanding context berlabel saja → null. Bersama v49, pseudo-context Kumo gagal dalam dua bentuk.
- Caveat: teacher memakai view nested_small (union) yang belakangan terbukti lemah.

## 2.2 CoLES — embedding urutan event self-supervised (`eda/240_coles_embeddings.py`)
- Paper: Babaev dkk., "CoLES: Contrastive Learning for Event Sequences with Self-Supervision", SIGMOD'22
  (arXiv 2002.08232). Di data bank riil: retail credit scoring 0,88 → baseline+CoLES 0,92 AUROC.
- Implementasi: GRU 128 di atas urutan event pra-keputusan per customer (ecom: kategori, metode bayar, log harga, harga
  relatif kategori, qty, ukuran order, cancel, gap, umur; cicilan: log amount, paid ratio, DPD, max DPD, delay, nomor
  cicilan, gap, umur), 5 potongan acak per customer, supervised-contrastive (τ 0,1), 12 epoch, **tanpa label**,
  100k customer (train+test). Embedding urutan penuh → PCA-32 → fitur LGB.
- 5-fold paired vs `lgb_base4` 0,52074: **ecom 0,51817 (−0,0026, P 0,13, kalah 5/5)**, **cicilan 0,52009 (−0,0007,
  P 0,43)**, gabungan 0,52002 (−0,0007, P 0,45) → null.

## Belum diuji (prior rendah, dicatat dari protokol SSL 23 Okt)
- Self-distillation cross-fitted (teacher OOF inner sebagai soft target student GBDT).
- R-CPL (density-regularized curriculum pseudo-labeling, Kim dkk. 2023) untuk GBDT, satu policy.
