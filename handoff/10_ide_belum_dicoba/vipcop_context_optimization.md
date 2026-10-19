# VIP-COP: Black-box Context Optimization for Tabular Foundation Models (belum dijalankan)

Paper: Yilong Chen, Xueying Ding, Leman Akoglu (Carnegie Mellon University), arXiv **2605.12904v2** (29 Sep 2026),
CC BY 4.0. Kode: https://anonymous.4open.science/r/VIP-COp. Diberikan user 19 Okt malam bersama info tim 0,57065.

## Ide
TFM (TabPFN, TabICLv2, OutFormer; secara prinsip juga Kumo) memprediksi dengan context = data latih. Context penuh tidak
selalu optimal (noise, sampel/fitur tak relevan, ukuran di luar distribusi pretraining). VIP-COP memilih **satu context
global** (n_C baris dan/atau d_C fitur) yang memaksimalkan performa validasi, black-box (tanpa gradien/internals),
lalu context itu dipakai untuk semua query (batching tetap efisien).

## Algoritma (Algoritma 1 paper)
1. Item = baris dan/atau fitur yang melebihi batas context (S = n + d atau salah satunya). φ⁽⁰⁾ = 1/S (seragam).
2. Beberapa rantai paralel dengan temperatur τ = η^(r − r_max), r_max = ⌊log_η R⌋, η default 2 (gaya Hyperband).
3. Per ronde t (R ronde): P(i) = softmax(φ/τ); ambil B context berukuran (n_C, d_C) dengan peluang itu; tiap context
   satu forward pass TFM pada D_val → AUROC p_b; indikator c_b ∈ {0,1}^S.
4. Update φ dengan SGD mini-batch pada regresi linear tak berbobot p ≈ Σ φ_i c_i (KernelSHAP untuk subset berukuran tetap).
5. Setelah R ronde: context = item dengan φ positif terbesar (top-k); estimasi skor = Σ φ; pilih rantai terbaik.
Implementasi asli ~1 detik per ronde (TabICL); R ≈ batas waktu dalam detik. Contoh setting OD: 100 ronde × 2.000
koalisi ukuran 500, top-3.000 titik.

## Hasil di paper
- TabPFN-v1, 38 dataset klasifikasi TALENT: +1,5% s.d. +14,7% vs baseline (acak, ensemble 20 context acak, context
  penuh "XLContext", KMeans, DT+TFM); TabICLv2: unggul atas kelima baseline (p < 0,01).
- Dengan augmentasi sampel/fitur: +1,9–5,8% (TabICLv2); di bawah noise fitur gain sangat besar.
- Outlier detection (OutFormer, 240 dataset) dan uji kontaminasi: VIP-COP lebih robust, membuang outlier yang ditanam.
Catatan: baseline "XLContext" (context penuh) = apa yang kita pakai; paper melaporkan VIP-COP unggul atasnya.

## Adaptasi untuk kita (rencana, belum disetujui user)
- Script: `eda/152_vipcop_kumo.py` (ditulis 19 Okt, dijalankan ~1 menit lalu **dibatalkan atas permintaan user**;
  satu-satunya output: AUROC fm_select pada query pencarian 0,83745).
- Bersih: D_val = **inner stop split** di dalam train fold (bukan outer fold); outer fold dinilai sekali.
- Item = 228 fitur numerik (4 kategori selalu ikut), d_C = 100. Tingkat baris (48k item) tidak terestimasi dengan
  anggaran ~1.000 evaluasi → ditunda / butuh anggaran jauh lebih besar atau inisialisasi pintar (coreset).
- Pencarian: context 20k baris inner-fit stratified, query 4.000 baris inner-stop tetap, e1, AUROC; 3 rantai τ
  ∈ {1; 0,5; 0,25} pada φ yang distandardisasi; 40 ronde × 8 koalisi; φ = ridge pada semua (c, p) yang terkumpul
  (pengganti SGD yang lebih stabil).
- Final: fitur terpilih, context seluruh train fold, e4, generator seed 0 → bandingkan paired dengan cache baseline
  `eda/out/kumo_finetune_f0_base.parquet` (KS fold 0 0,52865).
- Biaya terukur: forward Kumo 10k/20k context + 3.000 query = 3,0 / 4,2 s → ~1 jam pencarian + ~20 menit outer.
- Kalau fold 0 menjanjikan: 5 fold × 3 seed (seleksi fitur diulang per outer fold), lalu keluarga Kumo vs v31 dengan
  rata-rata seed, lalu probe public.
