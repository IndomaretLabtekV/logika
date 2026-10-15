# Status per 15 Oktober 2026 (malam WIB)

## Posisi

| Item | Nilai |
|---|---|
| Public terbaik | **0.56145** (v5, submit 14 Okt) |
| Public lain | v1 0.53139 · v2 0.54192 · v3 0.54545 · v6 0.56123 · v8 sekitar 0.561 (kata user "sama sama aja") |
| Belum diketahui public-nya | `results/v8/submission_v9_stack.csv` dan `results/v9/submission.csv` (tergantung apa yang sudah di-submit user) |
| OOF / held-out terbaik | Sekitar 0.527–0.529. Semua versi sejak v5 masih di dalam noise satu sama lain |
| Top-1 public (laporan user) | 0.57059 (13 Okt) → 0.57184 → 0.57458 (14 Okt) → **0.57493** (15 Okt) |
| Deadline notebook | **30 Oktober 2026, 23:59 WIB**. Telat dipotong 1 poin per menit |
| Sisa waktu | Sekitar 15 hari |

## Kenapa stuck

1. **Fitur.** Setelah v5, setiap kandidat fitur yang dites terhadap prediksi v5–v8 hasilnya setara noise. Yang dites: sekitar 1.200 rasio dan interaksi, DFS native TabPFN-Rel depth 2 dan 3, sequence bigram, event scan per jendela waktu, persona/cluster, empirical Bayes, residual nuisance, link antar customer, dan identitas aritmetika. Detailnya di `05_katalog_eksperimen/`.
2. **Model.** Semua model sangat berkorelasi (rank korelasi 0.97–0.997). TabM adalah model tunggal terbaik (sekitar 0.525). Foundation model (TabPFN-3.5, Causilo) mentok di 0.519–0.524. Model lain (EXAONE, LimiX-2, TabICLv2) tidak muat atau crash di T4.
3. **Blending.** Sudah jenuh. Logit stacking cuma menambah sekitar +0.001.
4. **Leaderboard.** Offset public − OOF stabil sekitar +0.03 di semua versi. Penyebabnya subset public yang kebetulan, jadi tidak bisa dimanfaatkan. Simulasi menunjukkan gap ke top-1 (sekitar 0.013) **mungkin** sebagian karena keberuntungan atau seleksi submission. Tapi ini tidak bisa dibuktikan: bisa juga mereka memang punya metode yang lebih baik.

## Yang terakhir dikerjakan

- **v9** sudah di-run di Kaggle dan selesai tanpa crash dalam sekitar 4 jam. Detailnya di `04_riwayat_versi/v9.md`.
  - Blend terpilih: logit stacking 11 model, held-out halves 0.5272.
  - Cek lokal: stacking 11 model v9 itu −0.0019 dibanding stacking 9 model v8 (`submission_v9_stack.csv`, held-out 0.5282). Pada 20×2 split, v9 cuma lebih baik di 1 dari 40. tabm_p dan RealMLP tidak menambah apa-apa.
  - Prediksi test v9 vs v8-stack: Spearman 0.9993. Jadi public-nya hampir pasti sama, sekitar 0.56.
- Saran ke user: submit v9 supaya file yang dihasilkan notebook reproducible ini bisa dipilih untuk private LB nanti. Notebook ini juga satu-satunya yang jalan utuh dari awal sampai akhir.
- **User minta berhenti bilang "final" terus.** Masih ada 15 hari, jadi jangan bahas pemilihan akhir kecuali diminta.

## Kalau mau langsung kerja

Baca `09_langkah_berikutnya/rekomendasi.md`. Ringkasnya:
- yang paling masuk akal adalah eksperimen terisolasi di Kaggle (satu perubahan per run, dibandingkan berpasangan dengan v9 sebagai kontrol);
- tuning hyperparameter GBDT/TabM, yang belum pernah dilakukan;
- fitur kandidat dari Astra yang belum pernah masuk model;
- objective ranking/pairwise.
