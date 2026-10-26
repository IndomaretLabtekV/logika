# Rencana merge pipeline teman + pipeline user (rekomendasi utama)

Tujuan: kandidat yang memadukan **Kumo V5 + spline + koreksi offset laten (teman, public 0,56470)** dengan
**keluarga pohon 6 GBDT + Causilo (user, rekor lama v31 0,56405)**, dievaluasi jujur di fold yang sama.

## Mengapa ini paling menjanjikan
- Pohon paling bagus transfer ke public; pipeline teman paling bagus di public; keduanya belum pernah digabung.
- Spearman test goal054 vs v31 0,98670 (vs antar-seed Kumo 0,999) → ada keragaman nyata.
- Koreksi offset laten terbukti transfer (+0,0008 → +0,0008).

## Aset tambahan dari sumber DSC0152 (diintegrasikan 25 Okt)
- **OOF + test asli per kaki** tersedia: `results/dsc0152/agas/v6x/output/preds_*.npz` (`train_ids`, `test_ids`,
  `oof_fac_full`, `oof_lgb_full`, `oof_lgb_lean`, `oof_lgbx_full`, `oof_cat_full`, `oof_xgb_full`, `oof_linear`, … dan
  `test_*`). Resep public terbaiknya `ext0.5|tm30|lgb_lean` (v66, 0,56424). Bisa langsung di-blend by ID dengan OOF
  pipeline utama (fold berbeda, tiap baris tetap OOF) — **jauh lebih murah** daripada mereproduksi pipeline teman.
- Kandidat blend tambahan (kunci bobot dulu): M4 = 1/3 pohon6 user + 1/3 DSC0152 v66-recipe + 1/3 Kumo (K64 atau
  Kumo-offset teman); M5 = v31-structure + DSC0152 sebagai keluarga pohon kedua.
- Catatan sumber DSC0152: OOF AUC/KS tertinggi mereka **tidak** menang di public (v70_06, v70_01) → nilai kandidat
  merge dengan bobot dikunci dan kemiripan ke submisi berskor, bukan hanya OOF.

## Langkah
### 1. Samakan fold (wajib sebelum blend OOF)
Pilihan A (disarankan): **jalankan resep teman di fold seed 42**.
- Fitur V5 per outer fold seed 42 sudah ada: `results/v56/teammate_features/outer_s42_f{k}/data/processed/
  application_features_v5.parquet` (1.012 kolom, ada kolom `split`; dibangun dengan fitting-customer fold k).
- Kolom Kumo teman: daftar `columns` di `artifacts/optimization/goal054/cv5_seed31027/kumo_offset_l20p003_unweighted_d0p5/definition.json`.
- Kumo: lebih mudah memakai engine user `eda/strict_kumo_engine.py::predict` (Kumo large v1.0.1 terverifikasi,
  context 50.000 stratified, e8, seed 2026+fold, chunk 128, fraction 0,7–0,85) daripada environment Python 3.11 teman.
  Estimasi ±10–12 menit/fold (44k query, ~90 kolom, context 50k) → ±1 jam 5 fold. Simpan OOF + test per fold atomik.
- Fitur laten `re_*` (12 kolom koreksi) **tidak ada di lokal** → bangun ulang dengan
  `scripts/reverse_synthetic_patterns.py` / `validate_reverse_features.py` (tanpa label; cek cutoff), atau mulai tanpa
  koreksi (Kumo V5 + spline saja = resep 0,56393) lalu tambahkan koreksi.
- Spline: `scripts/spline_model.py` (n_knots 3, degree 2) di fitur reverse/V5, per outer fold.
- Koreksi offset: logistic ridge l2 0,003, unweighted, kekuatan 0,5, difit di label outer-train **di luar context**
  (±6.000 baris), fold validasi tidak dipakai. 75% Kumo-offset + 25% spline.
- Cek reproduksi: OOF seed 42 seharusnya ~0,528–0,530 (seed 31027: 0,52930).

Pilihan B (murah, CPU): jalankan pohon user (v22 6 GBDT, `eda/valid2.py`) di fold **seed 31027** — tapi OOF teman
tetap tidak tersedia di lokal, jadi B hanya berguna bila OOF teman direproduksi juga. → Pilih A.

### 2. Blend dengan bobot dikunci sebelum skor (tulis di protocol JSON dulu)
- M1 = 1/3 pohon6 + 1/3 Causilo + 1/3 Kumo-offset-teman (struktur v31, Kumo diganti).
- M2 = 1/2 pohon6 + 1/2 Kumo-offset-teman (struktur v32-ish).
- M3 = 1/3 pohon6 + 1/3 Kumo-offset-teman + 1/3 K64 2-seed (dua Kumo berbeda fitur).
- Per-cohort rank (seperti `eda/221`), test = rata-rata fold. OOF pohon/Causilo yang ada (`results/v22/v22/gbdt/oof_*`,
  `results/v22/v22/fm/oof_causilo.parquet`, `results/v37/`, `results/v41/causilo_s12/`) adalah OOF rata-rata 3 seed
  outer (42/7/2026) — sah sebagai OOF; test pohon `results/v22/v22/gbdt/test_*`.
- Evaluasi: pooled KS/AUC + per fold, Δ vs REF (0,52625) dan vs Kumo-offset-teman sendiri, paired bootstrap 1.000.

### 3. Koreksi offset laten pada Kumo user
- Terapkan resep offset teman (12 kolom `re_*` + beban cicilan) ke logit K64 2-seed / main_small di fold seed 42.
  Bila naik dan transfer seperti di teman, ini kandidat Kumo baru.

### 4. Submisi
- Ambil kandidat dengan OOF tertinggi yang **jelas berbeda** dari submisi yang ada (cek Spearman vs goal054 0,56470,
  v31, K62). Perkiraan public: OOF + offset keluarga (blend dengan pohon ≈ +0,037; Kumo ≈ +0,035), sd 0,0015–0,0027
  tergantung kemiripan dengan submisi berskor.
- Tulis preflight (`eda/225_prepare_submission.py` atau JSON manual) sebelum upload; analisis sesudahnya.

### 5. Notebook
- Bila kandidat merge dipakai sebagai pegangan, perluas `notebook/build_final.py` (mode replay hash-checked + mode
  fresh) dan uji replay byte-identik.

## Risiko
- Hang laptop (4×): pecah per fold, cache atomik, guard aktif, jalankan saat dipantau.
- Optimisme seleksi resep teman (dipilih dari ±24 varian di seed 31027): reproduksi di seed 42 adalah uji yang jujur.
- `re_*` builder teman mungkin memakai reference populasi; pastikan tanpa label dan sebelum cutoff.
