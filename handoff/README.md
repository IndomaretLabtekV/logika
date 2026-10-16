# Handoff DSC LOGIKA UI 2026

Rangkuman semua yang terjadi di chat-chat sebelumnya: 12–16 Oktober 2026, versi v1 sampai v21 dan 3 probe submission.
Tujuannya supaya chat baru bisa langsung lanjut tanpa mengulang apa pun.

Notebook dan hasil tidak disalin ke sini; semuanya ada di `notebook/`, `results/`, `eda/`, `submissions/`. File di sini
menjelaskan isinya.

## Urutan baca
| Folder | Isi |
|---|---|
| `00_mulai_di_sini/` | Status sekarang (posisi, kenapa stuck, langkah berikutnya) dan prompt siap tempel untuk chat baru |
| `01_kompetisi_dan_aturan/` | Aturan lomba, metrik, deadline, leaderboard, batas submit, dan aturan kerja dari user (termasuk preferensi submit dan `/goal`) |
| `02_proyek_dan_tools/` | Struktur folder, jebakan lokal, notebook v9 + setup Kaggle, **Kaggle CLI (push kernel, output, dataset, submit, kuota, auth)**, tool bench lokal |
| `03_data_dan_fitur/` | Dataset, temuan EDA utama, fitur `features.py`, leakage dan duplikasi |
| `04_riwayat_versi/` | Tabel skor semua submission, kronologi permintaan user, satu file per versi v1–v21 |
| `05_katalog_eksperimen/` | Semua script EDA 00–96, cek lokal 16 Okt, review Astra (GPT-6) dan statusnya |
| `06_model_dan_blending/` | Semua model (protokol lama dan bersih), blending/stacking, riset model eksternal |
| `07_validasi_dan_leaderboard/` | Skema CV dan noise KS, **temuan bias OOF karena early stopping**, analisis probe submission 16 Okt |
| `08_pelajaran/` | Yang berhasil, yang tidak efektif, kesalahan dan bug |
| `09_langkah_berikutnya/` | Rekomendasi dan kalender sampai 30 Okt, status semua rekomendasi lama |

## Ringkasan
Credit risk, metrik KS. Public terbaik **0,56145** (v5 dan v8); top-1 0,57768, cutoff top-8 ~0,57005. Lompatan nyata
hanya dua: fitur perilaku v2 dan **fitur price premium `pp_*` v5** (+0,016 public). Setelah itu 40+ analisis EDA dan
~20 model/eksperimen (termasuk tuning, EBM, TabICL, set-encoder baris mentah, pseudo-label) tidak menambah informasi.
Temuan paling penting datang dari 3 probe submission 16 Okt: **OOF v1–v9 bias** karena semua model early-stop di
fold validasi (TabM paling parah), sehingga setiap kandidat "lebih bagus dari v8" di OOF turun di public. Langkah
berikutnya adalah rerun dengan validasi bersih, memilih blend dari OOF bersih (kemungkinan lebih berat ke GBDT), dan
merakit notebook final yang reproducible sebelum 30 Okt 23:59 WIB. Ekspektasi realistis: perbaikan kecil, bukan +0,016.
