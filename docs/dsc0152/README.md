# Sumber DSC0152 — ringkasan riset (dari PROGRESS.md dan agas/progress.md)

Menurut user (25 Okt) ini **tim yang sama**; nama tim Kaggle sekarang sementara dan akan diganti ke nama resmi menjelang
deadline. Dokumen sumber menyebut tim **"kali ini beneran deh" (DSC0152)**; anggota yang muncul: Agas (`agas/`), Dimaz
(`dimaz/`), dan pemilik laptop macOS `yutaaa` (skrip root/scheduler). Catatan skor sumber ini dulu tercatat sebagai entri
leaderboard terpisah (rank 18/83 pada 14 Okt) — bila memang dua tim Kaggle, perlu digabung di Kaggle sebelum deadline.

## File di folder ini
- `PROGRESS.md` — jurnal root tim (status 12–18 Okt, riwayat submit, CV per versi notebook, strategi, temuan, yang gagal).
- `next_submission.txt`, `DATA_DICTIONARY_dimaz.md` (versi kamus data lama, beda dari `data/DATA_DICTIONARY.md`),
  `gitignore_dsc0152.txt` (`.gitignore` asli).
- `inventaris_awal.csv` (608 file asal + SHA256), `pemetaan_file.csv` (path asal → path baru, status verifikasi).

## Skor public sumber ini (ledger)
| File | Public | Catatan |
|---|---:|---|
| `agas/v66/output/submission_v66_04_ext0.5_tm30_lgb_lean.csv` | **0,56424** | public terbaik sumber ini |
| `agas/v67/output/submission_v67_01_anchor.csv` | 0,56424 | kontrol |
| `agas/v70/output/submission_v70_07_ext0.5_tm30v70_lgb_lean.csv` | 0,56419 | ≈ anchor |
| `agas/v68/output/submission_v68_04_ext0.5_tm30_lgb_lean.csv` | 0,56411 | 10 fold seed |
| `agas/v61/output/submission_v61_alt_ext0.5_tm30_lean.csv` | 0,56312 | rekor sebelumnya |
| `v61_00_mix0.7_v61_p19_00` / `v61_04` | 0,56299 / 0,56253 | campuran root + Agas |
| `qc_00_ext1.5_v4_fac_full_only` (root) | 0,56171 | ekstrapolasi 1,5·v4 − fac |
| `v70_06` (OOF KS tertinggi) / `v70_01` (OOF AUC tertinggi) | 0,56040 / 0,55801 | OOF naik, public turun |
| root awal: lgb 5f 0,54417 → … → run_fac 0,55865 → LGB v4 0,55931 | | 10–13 Okt |
(Submisi CSV tidak ikut di upload; nama file merujuk struktur asli.)

## Pipeline dan CV
- Root (yutaaa): `features.py` + notebook 10 fold (fold seed 42/7/13), LightGBM + CatBoost + XGBoost + spline/linear,
  **factor model** (12 sumber risiko digabung jadi satu skor, loading per fold dari label train fold; LGB 0,84437 →
  0,84524). Terbaik CV: **bag v4 3 seed top_mix AUC 0,84774 / KS 0,52786**. Tim stack LR (64 komponen + Agas) AUC
  0,84806 / KS 0,52848.
- Agas: "kaki" (legs) `fac_full`, `lgb_full`, `lgb_lean`, `lgbx_full`, `cat_full`, `xgb_full`, `linear`, `tm30` per fold,
  blend + **ekstrapolasi** (mis. `ext0.5`: menjauh dari kaki lemah), 10–15 fold × beberapa fold seed
  (42,7,13,21,5,101,3,17,11,23). v69 factor features +0,0006 AUC; v70 blok lintas sumber; v71 panel historis, residual
  bunga, MLP. OOF AUC terbaik ±0,8484.
- Dimaz: model default awal (`train_default_model.py`).

## Temuan penting sumber ini
- Sinyal terkuat: harga satuan relatif median kategori (`ec_price_rel`) — sama dengan `pp_*` pipeline utama.
- Suku bunga = grade risiko 5 level (16–32%), default 5,5–50%.
- Train/test identik (adversarial AUC 0,503); **noise potongan public 0,0015–0,003** antar model; varian fold seed
  hampir identik di test (Spearman 0,99996).
- Batas teoretis bila risiko terukur sempurna ±AUC 0,869; celah berasal dari jumlah data per nasabah.
- Musim: default per bulan 12,0–14,7%; `dec_doy` +0,0002 AUC.
- **OOF AUC/KS tertinggi tidak menang di public** (v70_06, v70_01); resep `ext0.5|tm30|lgb_lean` (v66) tetap terbaik.

## Yang tidak membantu (sumber ini)
400+ fitur tambahan, MLP, GAM, stacking, pseudo label, seleksi fitur, denoising harga, normalisasi per kelompok, variasi
factor model, optimasi bobot blend, bobot kelas untuk KS (AUC turun ke 0,840), batasan monoton, tuning num_leaves/
min_child/feature_fraction, model khusus nasabah baru, 44 fitur ecommerce baru (`ec_deep.py`), 30 fitur tabel lain
(`other_deep.py`), seed ke-4 v4, 10 vs 6 fold seed.

## Relevansi untuk merge
- Tersedia **OOF + test asli per kaki** (`results/dsc0152/agas/v6x/output/preds_*.npz`, by ID) → bisa langsung di-blend
  dengan OOF pipeline utama (fold berbeda, tetapi tiap baris tetap OOF).
- Lihat `handoff/2026-10-25/07_langkah_berikutnya/rencana_merge.md` (diperbarui).
