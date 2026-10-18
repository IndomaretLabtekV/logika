# Status lama per 16 Oktober 2026 (malam) — arsip

Disimpan sebagai riwayat. Status terbaru ada di `00_mulai_di_sini/status_sekarang.md`.

## Posisi
| Item | Nilai |
|---|---|
| Public terbaik | **0,56145** (v5 dan v8) |
| Submission lain | v1 0,53139 · v2 0,54192 · v3 0,54545 · v6 0,56123 · v9 0,55917 · probe S3 0,55778 · S5 0,55866 · S4 0,56133 |
| Top-1 public | **0,57768** (tim "cupuu"); cutoff top-8 ~0,57005; kita ~#34 |
| Batas submit | 3 per hari (kuota 16 Okt sudah habis) |
| Kuota GPU Kaggle | 30 jam/minggu, **habis 16 Okt**; CPU tetap bisa |
| Deadline notebook | **30 Oktober 2026, 23:59 WIB** (telat −1 poin/menit) |
| Goal | `/goal` (>0,57768) sudah di-clear user |

## Apa yang terjadi hari ini (16 Okt)
1. 12 eksperimen Kaggle (v10–v21, sebagian GPU) dan belasan cek EDA lokal. **Semuanya null**: tuning LGB/TabM, EBM,
   set-encoder baris mentah, objective ranking, stacker L2, TabICL, linear/monotone/ExtraTrees, pseudo-label, fitur
   mature-history, pruning. Lihat `04_riwayat_versi/v10.md` … `v20_belum_jalan.md`.
2. User minta stop semua eksperimen dan membuat submission uji yang bisa dianalisis. Tiga probe disubmit satu per satu:
   S3 (stack LR + TabICL17) 0,55778, S5 (bagian NN/FM v8) 0,55866, S4 (bagian GBDT v8) 0,56133.
3. **Temuan utama:** semua model v1–v9 early-stop di fold validasi yang sama dengan baris OOF → OOF optimis, terutama
   TabM (~0,008 vs ~0,005 LGB). Ini menjelaskan kenapa setiap kandidat "OOF > v8" (v6, v9, S3, S5) turun di public,
   dan kenapa GBDT saja ≈ v8 di public (z 2,3). Detail: `07_validasi_dan_leaderboard/bias_oof_early_stopping.md`.

## Kenapa stuck
- **Informasi:** jenuh. Bahkan model yang membaca baris mentah semua tabel (v14) tidak menemukan apa-apa di luar fitur
  v5–v8. Lompatan terakhir adalah fitur harga v5.
- **Validasi:** bias, jadi selama ini kita mengoptimasi ke arah yang salah (bobot TabM terlalu besar, stack LR dengan
  bobot negatif). Ini bisa diperbaiki, tapi gain yang diharapkan kecil (bukan +0,016).
- Gap 0,016 ke top-1 terlalu besar untuk keberuntungan satu submission (sd 0,0026–0,0055) dan tidak terlihat jalannya
  dari validasi kita. Kemungkinan top tim punya sesuatu yang tidak kita temukan, atau gabungan seleksi + keberuntungan;
  tidak bisa dibuktikan.

## Langkah berikutnya
Lihat `09_langkah_berikutnya/rekomendasi.md`. Ringkasnya:
1. Rerun pipeline (GBDT 3 seed + TabM + TabPFN + Causilo) dengan **early stopping di inner split** begitu kuota GPU reset,
   lalu pilih blend dari OOF bersih (rank-average bobot positif, kemungkinan lebih berat ke GBDT).
2. Rakit notebook final yang benar-benar menghasilkan file yang disubmit, kumpulkan sebelum 30 Okt 23:59 WIB.
3. Jangan ulang eksperimen fitur/model yang sudah null.

## File penting
- Submission terbaik: `results/v8/submission.csv` (public 0,56145; dibuat lokal dari prediksi v8 setelah kernel crash).
- Probe: `submissions/probe/` (README.md, expectations.csv, S3/S4/S5).
- Bench lokal: `eda/s01_stack_bench.py` (base = OOF v8 yang bias; ganti setelah rerun bersih).
