# Prompt siap tempel untuk chat baru

Salin blok di bawah ke chat baru, lalu isi permintaan di akhir. Versi 19 Okt 2026 malam (menggantikan versi 18 Okt).

---

```
Gua lagi ikut lomba Kaggle DSC LOGIKA UI 2026 (credit risk, metrik KS, tabular binary classification,
slug data-science-competition-logika-ui-2026, akun Kaggle riantamaputra, tim Kaggle "lagi cape lomba asli").
Repo: /home/veinsan/Projects/logika. Konteks lomba di @docs/context.md, dataset di @data/ (@data/DATA_DICTIONARY.md).
Seluruh progress chat sebelumnya ada di @handoff/. Baca berurutan:
@handoff/README.md, @handoff/00_mulai_di_sini/status_sekarang.md, @handoff/01_kompetisi_dan_aturan/,
@handoff/07_validasi_dan_leaderboard/validation_v2.md, @handoff/08_pelajaran/,
@handoff/09_langkah_berikutnya/rekomendasi.md, @handoff/10_ide_belum_dicoba/.
Jangan ulang eksperimen yang tercatat null di @handoff/08_pelajaran/yang_tidak_efektif.md,
@handoff/04_riwayat_versi/ dan @handoff/05_katalog_eksperimen/ kecuali ada alasan baru yang jelas.

Posisi: public terbaik 0,56405 (v31), #33/118 (19 Okt). Top-1 0,57788; 9 tim > 0,570. Notebook submission
(notebook/build_final.py) reproducible di Kaggle dan lokal (= v34, public 0,56271). Temuan kunci: CV model ICL
(Kumo/Causilo) tidak transfer ke public, pohon transfer; pohon sudah jenuh. Tim teman (0,57065) pakai ensemble Kumo 3 seed.

Aturan kerja:
- Bahasa Indonesia, jujur, klaim disertai angka, jangan overclaim, jangan git push (gua push sendiri).
- Kalau gua BERTANYA ("udah dicoba belum?"), JAWAB saja (boleh usul rencana) — jangan langsung menjalankan apa pun.
- Jangan pernah menyebut "final" (notebook final, resep final) selama lomba berjalan; pakai "notebook submission".
- Lokal: CachyOS, i7-12700H 20 thread, RAM 32 GB, RTX 3070 Ti Laptop 8 GB. SEMUA run di lokal (kuota GPU Kaggle habis;
  jangan push kernel Kaggle kecuali gua minta saat itu). VRAM < 8 GB, satu job berat sekaligus, job berat pakai
  `taskset -c 0-11 nice -n 10` dan OMP_NUM_THREADS 4–8 (laptop pernah hang 2×).
- Kaggle CLI (submit/leaderboard): KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" kaggle ... ; kalau gagal,
  minta gua `! kaggle auth login --force`.
- Token HF/PriorLabs ada di ~/.config/logika/secrets.env dan ~/.cache/huggingface/token. JANGAN tulis token ke
  repo, notebook, commit, atau dataset Kaggle.
- Validasi wajib validation_v2 (eda/valid2.py): stopping hanya di inner split, bobot blend cross-fit, kandidat
  dibandingkan resep v8 bersih (0,52426) dan v31 di fold yang sama: paired bootstrap P>0 ≥ 0,9 dan menang mayoritas fold.
  Margin < ~0,002 butuh rata-rata beberapa seed outer (varians seed ±0,0015). Fitur yang dipilih dari banyak kandidat
  memakai label WAJIB divalidasi dengan seleksi bersarang per outer fold (winner's curse terbukti).
- Blend: rank-average bobot positif, GBDT minimal ~1/3; tanpa stack LR berbobot negatif. Model ICL di-overrate CV.
- Submit: maks 3/hari (reset 07:00 WIB), satu per satu, upload sebagai submission.csv, description berisi isi +
  pertanyaan riset, tulis perkiraan ± sd sebelum submit lalu analisis. Selisih public < ~0,002 = noise.
  Ikuti jendela waktu yang gua tetapkan persis (19 Okt gua larang submit analisis sebelum 21 Okt 06:00; dilanggar).
- Larangan: data eksternal, leakage masa depan, pseudo-label test, fitting distribusi test, LB probing (rule 11).
  Pretrained publik boleh.
- Notebook submission pakai @ref/notebook/SKILL.md, harus menghasilkan file yang disubmit, deadline 30 Okt 2026 23:59 WIB.

Permintaan gua sekarang: <isi di sini>
```
