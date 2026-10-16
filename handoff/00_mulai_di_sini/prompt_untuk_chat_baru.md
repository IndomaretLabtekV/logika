# Prompt siap tempel untuk chat baru

Salin blok di bawah ke chat baru, lalu isi permintaan di akhir.

---

```
Gua lagi ikut lomba Kaggle DSC LOGIKA UI 2026 (credit risk, metrik KS, tabular binary classification,
slug data-science-competition-logika-ui-2026, akun Kaggle riantamaputra).
Konteks lomba ada di @docs/context.md, dataset di @data/ (lihat @data/DATA_DICTIONARY.md).
Seluruh progress dari chat sebelumnya ada di folder @handoff/. Baca dulu berurutan:
@handoff/README.md, @handoff/00_mulai_di_sini/status_sekarang.md,
@handoff/07_validasi_dan_leaderboard/bias_oof_early_stopping.md, @handoff/08_pelajaran/,
@handoff/09_langkah_berikutnya/rekomendasi.md.
Jangan ulang analisis/eksperimen yang sudah tercatat null di @handoff/04_riwayat_versi/,
@handoff/05_katalog_eksperimen/ dan @handoff/08_pelajaran/yang_tidak_efektif.md kecuali ada alasan baru yang jelas.

Aturan kerja:
- Jawab dalam Bahasa Indonesia.
- JANGAN RUN MODEL ATAUPUN NOTEBOOKNYA DI LOCAL! Lokal hanya EDA, preprocessing, dan aritmetika pada prediksi
  tersimpan (blending/stacking/bootstrap). Laptop: Intel Core Ultra 7 155H, iGPU Intel Arc (tanpa CUDA).
- Model dijalankan di Kaggle. Lu boleh push kernel sendiri lewat Kaggle CLI dan ambil outputnya
  (cara lengkap: @handoff/02_proyek_dan_tools/kaggle_cli_kernel_dan_submit.md). Kuota GPU 30 jam/minggu.
  Kalau CLI bilang "Authentication required", minta gua jalanin `kaggle auth login --force`.
- Validasi wajib bersih: early stopping / seleksi apa pun HANYA di inner split dari train fold, jangan di fold OOF.
- Submit: maksimal 3/hari, satu per satu, nama file selalu submission.csv, jelaskan isinya di description,
  analisis skornya dulu sebelum submit berikutnya. Setiap submit harus menjawab pertanyaan riset, bukan coba-coba.
  Jangan submit kandidat yang belum lebih bagus dari v8 di validasi bersih kecuali gua minta probe.
- Jujur soal hasil, jangan fabrikasi, jangan overclaim. Jangan LB probing label (rule 11).
- Jangan bilang "final" terus-terusan. Deadline notebook 30 Oktober 2026 23:59 WIB.
- Notebook final pakai skill @ref/notebook/ dan patuhi @docs/context.md; harus menghasilkan file yang disubmit.
- Jangan git push; gua push sendiri.

Permintaan gua sekarang: <isi di sini>
```
