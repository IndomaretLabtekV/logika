# Tool lokal untuk menilai prediksi tersimpan (era 16 Okt)

> ⚠ **Usang sejak 17 Okt.** Semua tool di bawah memakai OOF v8 yang bias sebagai base. Penggantinya: `eda/blend2.py`,
> `eda/write_candidate.py`, `eda/screen2.py`, `eda/107_public_noise.py` di atas OOF bersih validation_v2
> (lihat `validation_v2_dan_tool.md`). File ini disimpan sebagai riwayat.

Semua script dijalankan dari root repo dengan `.venv/bin/python`. Tidak ada yang melatih model.

| Script | Fungsi | Contoh |
|---|---|---|
| `eda/s01_stack_bench.py` | Tambahkan kandidat ke stack logit-LR 9-model v8 (lgb, lgb_s, xgb, xgb_s, cat, cat_s, tabm, tabpfn, causilo dari `results/v8/oof_*.npy`); 20 × 2-fold stratified halves; cetak held-out KS, Δ vs base, sd, wins/40 | `eda/s01_stack_bench.py t17=results/v17/v17/oof.parquet:tabicl` |
| `eda/s02_stack_submission.py` | Tulis submission stack (base + kandidat), cetak bobot, KS in-sample, Spearman vs `results/v8/submission_v9_stack.csv` | `eda/s02_stack_submission.py out.csv name=oof.parquet:col:test.parquet:col` |
| `eda/s03_mix_bench.py` | Campuran bobot tetap kandidat dengan stack v8 | |
| `eda/s04_probe_analysis.py` | Bandingkan skor public probe dengan perkiraan di `submissions/probe/expectations.csv` (z per submission) | `eda/s04_probe_analysis.py S3=0.55778 S4=0.56133 S5=0.55866` |
| `eda/review_remote_tuning.py` | Untuk run v10–v13: control vs tuned vs campuran 80% v8 + 20% kandidat, bootstrap, gambar ke `results/vNN/vNN/v8_review/` | `eda/review_remote_tuning.py results/v10` |

Fungsi yang bisa diimpor dari `eda/s01_stack_bench.py`: `load_base()` (OOF 9 model v8 + target, urut application_id),
`logit`, `fit_lr` (Newton, L2 = 1), `ks`, `bench`.

## Cara menilai yang dipakai
- **Held-out 20×2 halves**: fit stack di satu separuh, nilai di separuh lain, ulang 20 kali. Δ dan wins/40 adalah
  stabilitas fit stack, **bukan** ketidakpastian sampling.
- **Bootstrap baris** (P>0) untuk ketidakpastian sampling; ini yang menentukan apakah gain nyata. Contoh TabICL17: 36/40
  wins tapi P>0 hanya 0,69.
- **sd selisih public** = sd selisih KS dua prediksi di subsample 9.000 baris OOF (30% test). Selisih public < 2 sd
  tidak bisa dibedakan dari noise.

## Keterbatasan penting
Base bench = OOF v8 yang **bias** (early stopping di fold validasi). Kandidat dari run bersih (v10–v21) dibandingkan
terhadap base yang optimis, dan kandidat yang dipilih dengan bench ini condong ke model yang biasnya mirip TabM. Setelah
ada rerun bersih, ganti `load_base()` ke OOF bersih itu.
