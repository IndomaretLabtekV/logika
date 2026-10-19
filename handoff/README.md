# Handoff DSC LOGIKA UI 2026

Rangkuman semua chat: 12–16 Okt 2026 (v1–v21, probe S3–S5), sesi `/goal` 17–18 Okt (validation_v2, v22–v36, Kumo-Tabular,
6 submission, notebook submission), dan sesi `/goal` 18 Okt sore – 19 Okt (35 siklus riset, bug Kumo, 3 probe keluarga,
run lokal notebook, ide VIP-COP). Tujuannya supaya chat baru bisa langsung lanjut tanpa mengulang apa pun.

Notebook dan hasil tidak disalin ke sini; semuanya ada di `notebook/`, `results/`, `eda/`, `submissions/`. File di sini
menjelaskan isinya.

## Urutan baca
| Folder | Isi |
|---|---|
| `00_mulai_di_sini/` | **Status sekarang** (posisi, temuan penting, kenapa stuck, langkah berikutnya) dan prompt siap tempel untuk chat baru |
| `01_kompetisi_dan_aturan/` | Aturan lomba, metrik, leaderboard (19 Okt), batas submit, **aturan kerja user** (semua `/goal`, kutipan teguran, keamanan token) |
| `02_proyek_dan_tools/` | Struktur folder, **cara pakai validation_v2 dan semua tool/evaluator**, notebook submission (Kaggle + lokal), kecepatan lokal vs Kaggle, Kaggle CLI, jebakan lokal |
| `03_data_dan_fitur/` | Dataset, temuan EDA (peta informasi, struktur skor, titik potong KS, varians seed), fitur `features.py` + konstruksi yang diuji, leakage dan audit cutoff |
| `04_riwayat_versi/` | Tabel skor semua submission dan OOF, kronologi chat, satu file per versi v1–v36, `v37_v45_sesi_18_19okt.md`, jurnal mentah 17–18 Okt dan **18 Okt sore – 19 Okt (35 siklus)** |
| `05_katalog_eksperimen/` | Semua script EDA 00–152 dengan hasilnya, screening fold 0, cek lokal 16 Okt, review Astra |
| `06_model_dan_blending/` | Semua model (protokol lama/bersih), blending dan pelajaran public (probe keluarga), riset model eksternal |
| `07_validasi_dan_leaderboard/` | **validation_v2** + revisi (varians seed, winner's curse, CV ICL tidak transfer), noise public dan kalibrasi perkiraan, bias OOF early stopping, probe 16 Okt |
| `08_pelajaran/` | Yang berhasil, yang tidak efektif, kesalahan dan bug (termasuk pelanggaran arahan user) |
| `09_langkah_berikutnya/` | Rekomendasi + kalender sampai 30 Okt, rencana slot analisis, status rekomendasi lama |
| `10_ide_belum_dicoba/` | **Audit setup Kumo** (petunjuk tim 0,57065), **VIP-COP** (arXiv 2605.12904) beserta rencana implementasi, ide lain |

## Ringkasan
Credit risk (DPD ≥ 90 dalam 12 bulan), metrik KS. Public terbaik **0,56405** (v31), tim Kaggle "lagi cape lomba asli",
**#33 dari 118** per 19 Okt. Top-1 0,57788; sembilan tim > 0,570.

Lompatan nyata hanya tiga: fitur perilaku v2 (+0,011 public), **price premium `pp_*` v5** (+0,016), dan blend pohon + ICL
(Kumo context penuh + Causilo) v31 (+0,0026). validation_v2 (16–17 Okt) memperbaiki bias OOF.

Sesi 18–19 Okt: bug fine-tune Kumo ditemukan (urutan kolom kelas berubah per panggilan) tapi fine-tune null;
±30 eksperimen fitur/model lain null di level keluarga. Tiga probe keluarga di public menunjukkan **CV pohon dan LR
transfer, CV ICL tidak** (pohon saja 0,56337 ≈ v31; Causilo saja 0,55333, z −3,4). Pohon jenuh; plafon pendekatan
~0,564–0,565. Varians seed outer ±0,0015 dan winner's curse seleksi fitur memperketat validasi.

Notebook submission (`notebook/build_final.py`) mereproduksi v34 di Kaggle (18 Okt) **dan** di lokal (19 Okt, 2,8 jam).
Sisa: banner tim (data user), keputusan resep. Petunjuk terbaru: tim teman 0,57065 memakai **ensemble Kumo 3 seed** →
prioritas berikutnya memperbaiki setup Kumo (`10_ide_belum_dicoba/`). Deadline notebook **30 Okt 2026 23:59 WIB**.
