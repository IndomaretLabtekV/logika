# Langkah berikutnya (15 hari tersisa, sampai 30 Okt 23:59 WIB)

Jujurnya: semua tuas di bawah punya **ekspektasi kecil** (sekitar +0,001–0,004 OOF). Tidak ada jaminan breakthrough ke 0,575. Tapi ini hal-hal yang **belum pernah diuji di model sungguhan**. Itu lebih bernilai daripada mengulang EDA residual yang sudah 40+ kali hasilnya null.

## Prinsip eksperimen (pelajaran v6 dan v8)
1. **Satu pertanyaan per eksperimen.** Kontrolnya v9 (fitur sama, fold sama, seed sama). Bandingkan berpasangan per fold dan dengan bootstrap. Promosikan hanya kalau naik konsisten (≥ 4/5 fold) dan Δ OOF > sekitar 0,002.
2. **Harness murah dulu.** LightGBM 1 seed × 5 fold memakan sekitar 100 detik di Kaggle. 20 konfigurasi jadi sekitar 35 menit. Kandidat yang lolos baru diulang 3 seed dan di model lain.
3. Simpan per-model, per-fold, per-seed OOF dengan ID, bobot blend, versi paket, dan hash fitur.
4. Model berisiko (FM baru) dijalankan di notebook atau run terpisah, jangan di depan model penting.

## Prioritas A: belum pernah diuji, murah
| # | Eksperimen | Kenapa | Cara |
|---|---|---|---|
| A1 | **Tuning hyperparameter GBDT** | parameter tidak berubah sejak v1, tidak pernah di-tune | grid kecil atau Optuna di **inner CV** (fit di train fold saja). Kandidat: lr 0,01, leaves 15/63, min_child 50–400, ff 0,3–0,7, l1/l2, `extra_trees`, `path_smooth`, CatBoost depth 5–8 + `border_count`, XGB `max_bin`. Bandingkan berpasangan dengan v9 |
| A2 | **Tuning TabM** | TabM model tunggal terbaik dan juga tidak pernah di-tune | k (ensemble size), n_blocks, d_block, dropout, lr/wd, epoch dan patience, `TABM_BINS`. Inner validation untuk early stopping |
| A3 | **9 fitur selected-history Astra** | KS observed 0,344; belum pernah masuk model | join `eda/v8_review/out/03_selected_history_features.parquet` (ada ID-nya), atau port kodenya ke `features.py`, lalu jalankan harness LGB |
| A4 | Rasio DPD exposure-matang dan delta skor provider sama (Astra v6) | screening 0/15, tapi screening lama kena bug bin | harness LGB |
| A5 | eb_k1 / eb_k30 (`eda/94`, chi2 sekitar 2) | lemah, murah | harness LGB |
| A6 | Seleksi fitur (buang fitur noise/redundan) | 232 fitur dengan banyak yang \|ρ\| > 0,95; belum pernah diuji | drop berdasarkan importance atau korelasi di dalam fold; terutama untuk TabM/FM |

## Prioritas B: objective dan ensemble
| # | Eksperimen | Catatan |
|---|---|---|
| B1 | Objective berbasis ranking: LightGBM `rank_xendcg` / `lambdarank` (satu grup), XGB `rank:pairwise`, TabM dengan loss pairwise / AUC surrogate | KS adalah metrik ranking. Belum pernah dicoba. Hasilnya tidak pasti |
| B2 | Kebijakan checkpoint: early stop pada AUC vs logloss vs KS (inner validation) | saran Astra. AUC dan KS kadang berlawanan |
| B3 | Stacking yang lebih rapi: L2 di-tune, bobot nonnegatif, seleksi kolom nested | sekarang 9 model > 11 model. Gain kecil, tapi bisa langsung ditanam di notebook |
| B4 | Pseudo-labeling test (soft label dari blend, hanya baris yang yakin) | hanya memakai data panitia (transduktif). **Cek dulu kesesuaian dengan rules.** Ekspektasi kecil |

## Prioritas C: model mahal, run terpisah
| # | Eksperimen | Catatan |
|---|---|---|
| C1 | EXAONE context 8–12k dengan **query di-chunk**, satu GPU dulu, sampai OOF lengkap | belum pernah punya OOF lengkap. Probe 4k 0,5437 di baris yang sama dengan LGB 0,5467 |
| C2 | TabICLv2 dengan query di-chunk + pilot kecil (RSS/VRAM dicatat) | crash di v8 karena query 44k |
| C3 | Xiaomi-TabLDM / TabFM | belum dicek kelayakannya di T4 |

## Yang **jangan** dihabiskan waktunya lagi
EDA residual univariat, rasio dan interaksi, DFS, sequence counting, persona, augmentasi historis sebagai baris training, menambah seed, model berkorelasi tinggi ke blend, rekalibrasi segmen, dan LB probing. Detailnya di `08_pelajaran/yang_tidak_efektif.md`.

## Rencana kalender (fleksibel)
| Tanggal | Kegiatan |
|---|---|
| 16–18 Okt | Notebook **v10 ablation harness** (LGB cepat): A1, A3, A4, A5, A6, dan B1 untuk LGB. Satu run Kaggle sekitar 1–2 jam |
| 19–21 Okt | Pemenang dari harness diulang 3 seed + CatBoost/XGB. Tuning TabM (A2), B2 |
| 22–24 Okt | Opsional: C1 / C2 di run terpisah. Integrasi kandidat yang lolos ke pipeline utama (v11) |
| 25–27 Okt | Run penuh v11 dan analisis berpasangan vs v9 / v8-stack. Submit hanya kalau held-out naik berarti |
| 28–30 Okt | Rerun reproducibility notebook yang dipakai, rename `ParticipantNumber_TeamName_Notebook`, isi banner tim, pastikan notebook menghasilkan submission yang dipilih, lalu kumpulkan **sebelum 30 Okt 23:59 WIB** |

## Catatan reproducibility
- Notebook yang dikumpulkan harus "sesuai dengan submission tim". Jadi submission yang dipilih di Kaggle sebaiknya file yang benar-benar dihasilkan notebook itu, bukan blend lokal.
- Kalau v8-stack (blend lokal) ingin dipakai, ubah sel blend notebook supaya stacking memakai 9 model yang sama, lalu run ulang.
- Waktu run v9 sekitar 4 jam, jauh di bawah batas 12 jam. Masih ada ruang untuk tambahan model.
