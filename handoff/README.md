# Handoff DSC LOGIKA UI 2026

Isi folder ini adalah rangkuman semua yang terjadi di chat sebelumnya: 12–15 Oktober 2026, versi v1 sampai v9. Tujuannya supaya chat baru bisa langsung melanjutkan tanpa mengulang analisis yang sudah dikerjakan.

Notebook dan file hasil sengaja tidak disalin ke sini. Semuanya masih ada di `notebook/`, `results/`, `eda/`, dan `eda/out/`. File di sini cuma menjelaskan isinya.

## Urutan baca

| Folder | Isi |
|---|---|
| `00_mulai_di_sini/` | Status sekarang (stuck di mana) dan prompt siap tempel untuk chat baru |
| `01_kompetisi_dan_aturan/` | Aturan lomba, metrik, deadline, dan aturan kerja dari user (termasuk larangan run model di lokal) |
| `02_proyek_dan_tools/` | Struktur folder, script, venv, cara dry run, builder notebook, setup Kaggle, dan jebakan teknis |
| `03_data_dan_fitur/` | Dataset, temuan EDA utama, fitur di `features.py`, verifikasi preprocessing, leakage, dan duplikasi |
| `04_riwayat_versi/` | Tabel skor, kronologi permintaan user, dan satu file per versi (v1–v9): apa yang diubah, hasil, dan pelajarannya |
| `05_katalog_eksperimen/` | Setiap script EDA (00–96), pertanyaannya, dan hasilnya, plus ringkasan review Astra (GPT-6) |
| `06_model_dan_blending/` | Semua model yang dicoba, hasil blending/stacking, dan riset model eksternal beserta sumbernya |
| `07_validasi_dan_leaderboard/` | Skema CV, noise KS, simulasi public/private, offset public−OOF, dan analisis leaderboard |
| `08_pelajaran/` | Yang berhasil, yang tidak efektif, serta kesalahan dan bug yang pernah terjadi |
| `09_langkah_berikutnya/` | Hal yang belum pernah diuji dan rencana untuk sisa hari sampai 30 Oktober |

## Ringkasan satu paragraf

Kompetisinya credit risk dengan metrik KS. Skor public terbaik kita **0.56145** (v5). v6, v7, v8, dan v9 secara statistik sama dengan v5: OOF sekitar 0.527–0.528. Satu-satunya lompatan besar datang dari fitur **price premium** (`pp_*`) di v5, sekitar +0.014 KS. Setelah v5, lebih dari 40 analisis lokal dan beberapa model baru tidak menemukan sinyal tambahan yang bisa dideteksi (resolusinya sekitar 0.002 KS). Top-1 public terakhir yang dilaporkan user adalah 0.57493. Masih ada beberapa kandidat yang belum pernah diuji di model sungguhan, terutama hyperparameter GBDT/TabM yang tidak pernah di-tune sejak v1, fitur-fitur usulan Astra, dan objective berbasis ranking. Daftarnya ada di `09_langkah_berikutnya/`.
