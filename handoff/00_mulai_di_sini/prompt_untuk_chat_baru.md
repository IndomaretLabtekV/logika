# Prompt siap tempel untuk chat baru

Salin blok di bawah ke chat baru, lalu isi permintaan di akhir. Versi 18 Okt 2026 (menggantikan versi 16 Okt yang
masih melarang run model lokal; laptop sekarang punya GPU NVIDIA).

---

```
Gua lagi ikut lomba Kaggle DSC LOGIKA UI 2026 (credit risk, metrik KS, tabular binary classification,
slug data-science-competition-logika-ui-2026, akun Kaggle riantamaputra, tim "lagi mager lomba bentar").
Repo: /home/veinsan/Projects/logika. Konteks lomba di @docs/context.md, dataset di @data/ (@data/DATA_DICTIONARY.md).
Seluruh progress chat sebelumnya ada di @handoff/. Baca berurutan:
@handoff/README.md, @handoff/00_mulai_di_sini/status_sekarang.md, @handoff/01_kompetisi_dan_aturan/,
@handoff/07_validasi_dan_leaderboard/validation_v2.md, @handoff/08_pelajaran/,
@handoff/09_langkah_berikutnya/rekomendasi.md.
Jangan ulang eksperimen yang tercatat null di @handoff/08_pelajaran/yang_tidak_efektif.md,
@handoff/04_riwayat_versi/ dan @handoff/05_katalog_eksperimen/ kecuali ada alasan baru yang jelas.

Posisi: public terbaik 0,56405 (v31), #24; top-1 0,57768, tujuh tim > 0,570. Notebook final
(notebook/build_final.py) sudah reproducible di Kaggle dan menghasilkan v34 (public 0,56271).

Aturan kerja:
- Bahasa Indonesia, jujur, klaim disertai angka, jangan overclaim, jangan git push (gua push sendiri).
- Lokal: CachyOS, i7-12700H, RAM 32 GB, RTX 3070 Ti Laptop 8 GB. Model boleh di-screening lokal (1 fold),
  VRAM < 8 GB, satu job berat sekaligus, batasi OMP_NUM_THREADS 4–8 (laptop pernah hang).
- Full fold dijalankan di tempat yang lebih cepat: lokal 3070 ≈ 2,3–3× satu T4. Kaggle (T4×2, 30 jam GPU/minggu,
  maks 2 sesi GPU bersamaan, 12 jam/sesi) dipakai untuk paralel, model > 8 GB VRAM, dan notebook final.
- Kaggle CLI: kalau "Authentication required", pakai
  KAGGLE_API_TOKEN="$(kaggle auth print-access-token)" kaggle ...  ; kalau itu pun gagal, minta gua
  `! kaggle auth login --force`. Cek status kernel ≥ 2 kali sebelum menyimpulkan CANCEL.
- Token HF/PriorLabs ada di ~/.config/logika/secrets.env dan ~/.cache/huggingface/token. JANGAN tulis token ke
  repo, notebook, commit, atau dataset Kaggle.
- Validasi wajib validation_v2 (eda/valid2.py): stopping hanya di inner split, bobot blend cross-fit, kandidat
  dibandingkan dengan resep v8 bersih (paired bootstrap P>0 ≥ 0,9 dan menang mayoritas fold).
- Blend: rank-average bobot positif, GBDT minimal ~1/3 (blend tanpa pohon jatuh di public). Tanpa stack LR berbobot negatif.
- Submit: maks 3/hari (reset 07:00 WIB), satu per satu, nama file submission.csv, description berisi isi +
  pertanyaan riset, tulis perkiraan ± sd sebelum submit lalu analisis. Selisih public < ~0,002 = noise.
- Larangan: data eksternal, leakage masa depan, pseudo-label test, fitting distribusi test, LB probing (rule 11).
  Pretrained publik boleh.
- Notebook final pakai @ref/notebook/SKILL.md, harus menghasilkan file yang disubmit, deadline 30 Okt 2026 23:59 WIB.
- Jangan terus-terusan bilang "final"; bahas pilihan akhir kalau gua minta.

Permintaan gua sekarang: <isi di sini>
```
