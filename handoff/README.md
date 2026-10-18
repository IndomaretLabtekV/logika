# Handoff DSC LOGIKA UI 2026

Rangkuman semua chat sebelumnya: 12–16 Oktober 2026 (v1–v21, probe S3–S5) dan sesi `/goal` 17–18 Oktober
(validation_v2, v22–v36, Kumo-Tabular, 6 submission, notebook final). Tujuannya supaya chat baru bisa langsung lanjut
tanpa mengulang apa pun.

Notebook dan hasil tidak disalin ke sini; semuanya ada di `notebook/`, `results/`, `eda/`, `submissions/`. File di sini
menjelaskan isinya.

## Urutan baca
| Folder | Isi |
|---|---|
| `00_mulai_di_sini/` | **Status sekarang** (posisi, apa yang sudah dikerjakan, kenapa stuck, langkah berikutnya) dan prompt siap tempel untuk chat baru |
| `01_kompetisi_dan_aturan/` | Aturan lomba, metrik, leaderboard, batas submit, aturan kerja user (termasuk `/goal` 17 Okt, keamanan token, kutipan arahan) |
| `02_proyek_dan_tools/` | Struktur folder, **cara pakai validation_v2 dan tool**, **notebook final**, kecepatan lokal vs Kaggle, Kaggle CLI, jebakan lokal |
| `03_data_dan_fitur/` | Dataset, temuan EDA (termasuk peta informasi per keluarga fitur), fitur `features.py`, leakage dan duplikasi |
| `04_riwayat_versi/` | Tabel skor semua submission dan OOF bersih, kronologi chat, satu file per versi v1–v36, jurnal mentah 17–18 Okt, status lama 16 Okt |
| `05_katalog_eksperimen/` | Semua script EDA 00–111, screening fold 0 (zoo, Kumo, Causilo), cek lokal 16 Okt, review Astra |
| `06_model_dan_blending/` | Semua model (protokol lama dan bersih), blending (cross-fit, pelajaran public), riset model eksternal |
| `07_validasi_dan_leaderboard/` | **validation_v2**, noise public dan kalibrasi perkiraan, bias OOF early stopping, probe 16 Okt |
| `08_pelajaran/` | Yang berhasil, yang tidak efektif, kesalahan dan bug |
| `09_langkah_berikutnya/` | Rekomendasi dan kalender sampai 30 Okt, status semua rekomendasi lama |

## Ringkasan
Credit risk (DPD ≥ 90 dalam 12 bulan), metrik KS. Public terbaik **0,56405** (v31, #24 per 18 Okt); top-1 0,57768,
tujuh tim > 0,570.

Lompatan nyata hanya tiga: fitur perilaku v2 (+0,011 public), **fitur price premium `pp_*` v5** (+0,016), dan
**blend pohon + model in-context learning (Kumo-Tabular large context penuh + Causilo)** v31 (+0,0026). Temuan 16 Okt
bahwa OOF v1–v9 bias (early stopping di fold OOF) diperbaiki dengan **validation_v2** (stop di inner split, refit budget
tetap, bobot blend cross-fit, paired bootstrap); di protokol bersih urutan model menjadi Kumo > Causilo > GBDT > TabM.

Setelah itu semua jalur lain null: ±25 model zoo TabArena, 13 analisis EDA baru (learning curve datar, model linear ≈ TabM
≈ GBDT, 0/228 fitur residual, harga relatif habis), varian Kumo, ICL per segmen. Enam submission 17–18 Okt menunjukkan
**blend tanpa GBDT jatuh di public** dan varian setara-CV mengumpul di 0,5627–0,5641 (noise ~0,001). Pendekatan sekarang
kemungkinan berplafon ~0,564; target > 0,570 belum tercapai.

Notebook final (`notebook/build_final.py`) sudah lulus run penuh di Kaggle dan mereproduksi v34 (0,56271). Sisa:
lacak bug fine-tune Kumo (satu-satunya tuas yang belum diuji), isi banner tim, pin versi paket, kumpulkan sebelum
**30 Okt 2026 23:59 WIB**.
