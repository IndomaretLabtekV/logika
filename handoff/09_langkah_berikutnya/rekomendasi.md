# Langkah berikutnya (per 18 Okt 2026 siang; deadline notebook 30 Okt 23:59 WIB)

Rekomendasi versi 16 Okt sudah dikerjakan: Prioritas 1 (rerun bersih) = v22 + validation_v2, Prioritas 2 (notebook final
reproducible) = `notebook/build_final.py`, lulus run Kaggle. Statusnya ada di `status_rekomendasi_lama.md`.

## Inti
- Posisi 0,56405 (#24). Varian pendekatan sekarang mengumpul di 0,5627–0,5641. Target > 0,570 butuh ~+0,006 public,
  ≈ 3× gain terbesar yang pernah kita dapat sejak v5 (+0,0026 dari ICL).
- Fitur dan model zoo sudah habis diuji (lihat `08_pelajaran/yang_tidak_efektif.md`). Tuas yang **belum diuji** dan masih
  masuk akal hanya satu: **fine-tune Kumo-Tabular** pada data ini. Ekspektasi jujur: kecil (+0,001–0,003 kalau berhasil),
  bukan +0,006.
- Prioritaskan notebook final yang aman di atas eksperimen; 10% nilai bergantung padanya.

## Langkah 1: lacak bug `eda/kumo_finetune.py` (±1 jam, lokal)
Gejala: KS baseline (sebelum fine-tune, context = seluruh train fold 56k, e4) = **0,29776**, padahal Kumo context penuh
fold 0 di v28m = 0,5256. Yang sudah dipastikan **bukan** penyebab:
- pembacaan output: `out.numerical[..., 1]` identik dengan `out.to_pandas()["1"]`;
- cara ambil baris: `T[index_tensor]` dari satu TableTensor 70k vs tabel context+query terpisah ala `valid2.kumo_fit`
  memberi prediksi identik (korelasi 1,0) di context 16k dan 56k;
- cache fitur `data_pq/features.parquet` tidak berubah sejak 17 Okt 08:17; venv sama (`.venv`) dengan run v28.
Petunjuk aneh: dengan cara apa pun, context 4k (tr[:4000]) pada `va[:2000]` memberi KS 0,547, sedangkan context 16k/56k
pada `va[:3000]` memberi ~0,36.
Urutan diagnosa:
1. Jalankan ulang run yang dulu terbukti: `cd eda && ../.venv/bin/python run_valid2.py ../results/screen/kumo_recheck
   kumo_large_c20000 --folds 0 --seeds 42` (harus 0,518–0,528) dan `sdm_screen.py kumo_large --ctx 56000` (harus ~0,526).
   **Kalau tidak lagi reproduce → masalah environment** (driver NVIDIA/torch/sdm berubah, GPU bermasalah setelah laptop
   ~24 jam menyala). Cek `nvidia-smi`, `uv pip list` vs `results/v28_local/manifest.json`, restart laptop.
2. Kalau reproduce: simpan prediksi `kumo_fit` fold 0 per application_id dan bandingkan baris demi baris dengan
   `predict()` di `kumo_finetune.py` pada context/query yang sama (beda stypes yang diinfer di 70k baris vs context+query?
   beda dtype target `y.astype(str)`?).
3. Baru setelah baseline = ~0,526, jalankan screening fine-tune.

## Langkah 2: screening fine-tune Kumo (kalau langkah 1 beres)
- `python kumo_finetune.py --train all --lr 1e-5` dan `--train icl` (hanya blok ICL). Episode: context 2048 + query 512
  dari inner-fit; early stopping di inner-stop (context 16k, e1); outer fold 0 dinilai sekali sebelum dan sesudah.
- Kriteria lanjut: after − before di fold 0 dengan P>0 ≥ 0,9 **dan** +20% ke blend v31 di fold 0 positif.
- Kalau lolos: 5 fold (lokal ~1 jam/fold + fine-tune), blend 1:1:1 dengan pohon dan Causilo, kriteria validation_v2,
  pra-registrasi, submit. Untuk notebook final, fine-tune harus bisa dijalankan di dalam notebook (bobot hasil fine-tune
  adalah turunan data train; menyimpannya sebagai dataset Kaggle boleh, tapi notebook harus tetap bisa membuatnya ulang).
- Risiko: VRAM 8 GB untuk fine-tune 200M parameter + AdamW (~3,2 GB state + aktivasi) — kalau OOM pakai `--train icl`.

## Langkah 3: putuskan resep final
| Opsi | Public | Reproducible di notebook | Catatan |
|---|---|---|---|
| **v34** (pohon + Causilo + Kumo-70k) | 0,56271 | **ya, sudah terbukti** (Spearman 1,0) | rekomendasi default |
| v31 (pohon + Causilo + Kumo 5 fold) | 0,56405 | perlu Kumo 5 fold di notebook ≈ 8–16 jam T4 → tidak muat 1 sesi bersama pohon | selisih 0,0013 = noise public |
| v31/v34 + Kumo fine-tune | ? | tergantung langkah 2 | hanya kalau gain CV nyata |
Kalau Kaggle meminta memilih submission akhir: pilih yang dihasilkan notebook (v34) + satu lagi yang setara-CV (v31).
Bahas dengan user dulu; jangan terus menyebut "final".

## Langkah 4: rapikan notebook final (wajib, sebelum 28 Okt)
1. Isi banner `DSCxxxx - lagi mager lomba bentar` dan nama anggota — **tanya user**.
2. Pin versi paket di sel `%pip` (lightgbm 4.6.0, xgboost 3.4.1, catboost 1.2.10, structured-data-models 0.2.1,
   causilo 1.0.3, polars 2.0.0).
3. Rename `ParticipantNumber_TeamName_Notebook`, patuhi `ref/notebook/SKILL.md` dan `docs/context.md`.
4. Jalankan ulang penuh di Kaggle (±4,3 jam), bandingkan output dengan submission terpilih (Spearman ≈ 1), kumpulkan.
   Sisakan 2 hari untuk rerun kalau ada error.

## Ide lain (ekspektasi rendah; hanya kalau ada waktu dan kuota)
- KumoRelational (ICL relasional langsung di tabel mentah): ditunda karena sampler tetangga terbatas merusak agregat
  seperti pp_mean, dan v14 (set-encoder baris mentah) null. Biaya ~½ hari.
- Refit pohon di seluruh 70k untuk test: learning curve bilang ≤ +0,0005.
- Jangan habiskan submit untuk varian bobot di dalam klaster 0,5627–0,5641; selisihnya noise.

## Jangan dihabiskan waktunya lagi
Semua di `08_pelajaran/yang_tidak_efektif.md`. Khusus 17–18 Okt: zoo TabArena (TabPFN-3.5, TabICLv2, RealMLP, iLTM,
TabFM, Mitra, LimiX, TabDPT, Nori, AutoGluon default), Kumo small/medium/f154/e8/seed lain, subsample context, ICL per
segmen, bobot per segmen, blend tanpa pohon, utang eksternal bureau, shock makro, residual fitur, harga relatif per
subkelompok, 10-fold, menambah data latih.

## Kalender usulan
| Tanggal | Kegiatan |
|---|---|
| 19 Okt | Langkah 1 (bug fine-tune). Kalau masih buntu sampai sore, hentikan dan fokus notebook |
| 19–21 Okt | Langkah 2 (screening + 5 fold kalau lolos) |
| 21–24 Okt | Maks 1–2 submit/hari untuk kandidat yang lolos validation_v2; selain itu jangan submit |
| 24–26 Okt | Langkah 3 + 4: putuskan resep, update builder, pin versi, isi banner |
| 27–28 Okt | Run reproducibility final di Kaggle, cek output |
| 29–30 Okt | Cadangan rerun; kumpulkan sebelum 30 Okt 23:59 WIB |
