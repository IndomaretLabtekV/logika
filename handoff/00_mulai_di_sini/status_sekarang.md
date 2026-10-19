# Status per 19 Oktober 2026 malam

Ringkasan satu halaman. Log mentah: `04_riwayat_versi/jurnal_17_18okt.md` (17–18 Okt) dan
`04_riwayat_versi/jurnal_18okt_sore.md` (18 Okt sore – 19 Okt, 35 siklus). Kronologi semua chat:
`04_riwayat_versi/kronologi_chat.md`.

## Posisi
| Item | Nilai |
|---|---|
| Public terbaik | **0,56405** (v31: rank-avg 6 GBDT 1/3 + Causilo 1/3 + Kumo-Tabular large context penuh 1/3) |
| Tim di Kaggle | **"lagi cape lomba asli"** (athillazaidan, nicholaswisee, riantamaputra); dulu "lagi mager lomba bentar" |
| Peringkat (19 Okt ~20:00) | **#33 dari 118**, 22 submission. Top-1 **Datadataan 0,57788**, #2 cupuu 0,57768, #3 0,57458; **9 tim > 0,570**; 33 tim ≥ 0,564 (klaster padat) |
| Kurang ke target | 0,0138 ke top-1; ~0,006 ke batas top-8 (~0,5707) |
| Klaster public resep kita | v31 0,56405 · v32 0,56376 · trees_only 0,56337 · v35 0,56298 · v34 0,56271 (sd ~0,0006) |
| Validasi | validation_v2 (`eda/valid2.py`). Margin jujur v31 vs resep v8 bersih = **+0,0017 s.d. +0,0019** (bukan +0,0022; seed 42 ICL dan set seed pohon v22 sedikit beruntung) |
| Notebook submission | `notebook/build_final.py` → `notebook/final.ipynb`. Lulus run Kaggle 18 Okt (= v34) **dan** run lokal penuh 19 Okt (2,8 jam, Spearman 1,000000 vs v34). Banner tim masih placeholder |
| Kuota | Submit 3/hari reset 07:00 WIB; window 19 Okt **habis** (3 probe). **Kuota GPU Kaggle habis** → semua run lokal |
| Arahan submit aktif | Sampai **21 Okt 06:00 WIB**: submit hanya bila breakthrough. Setelah itu (tanpa breakthrough) slot boleh untuk analisis (rencana sudah ditetapkan di `09_langkah_berikutnya/rekomendasi.md`) |
| Deadline notebook | **30 Okt 2026 23:59 WIB** (telat −1 poin/menit) |

## Perjalanan singkat (detail `04_riwayat_versi/`)
1. 12–14 Okt (v1–v8): fitur perilaku v2 (+0,011 public), **price premium `pp_*` v5 (+0,016)** — dua lompatan terbesar.
2. 15–16 Okt (v9–v21, probe S3–S5): stacking/TabICL/tuning null; ditemukan **bias OOF** (early stopping di fold OOF).
3. 17–18 Okt (validation_v2, v22–v36): protokol bersih; zoo TabArena null kecuali **Kumo-Tabular**; v31 0,56405 (terbaik);
   blend tanpa pohon jatuh di public; notebook submission dirakit dan lulus di Kaggle.
4. 18 Okt sore – 19 Okt (sesi ini, 35 siklus): bug fine-tune Kumo ketemu; fine-tune, stacking, distilasi, feature
   bagging, ~20 keluarga fitur baru, seleksi hyperparameter bersarang — **semua null di level keluarga model**.
   3 probe keluarga di public membuktikan **CV ICL tidak transfer** (Causilo-only 0,55333) sedangkan **pohon transfer**
   (trees-only 0,56337). Notebook submission lulus run lokal.

## Temuan paling penting sesi ini
| Temuan | Bukti | File |
|---|---|---|
| Bug `kumo_finetune.py`: urutan kolom kelas `out.numerical` Kumo berubah antar panggilan | Spearman chunk ±0,998; setelah perbaikan baseline 0,52865 | `eda/112`, jurnal siklus 1 |
| **CV keluarga ICL tidak transfer ke public**; pohon dan LR transfer | offset public−CV: pohon +0,0389, LR +0,0372, Kumo +0,0339, **Causilo +0,0292 (z −3,3)** | `eda/131`, probe v39a–c |
| Prediksi test ICL sehat (bukan bug) | Spearman vs pohon OOF/test 0,984/0,987; per chunk konsisten | `eda/130` |
| Penyebab ICL tertinggal tidak terlihat di data | subset alami (`eda/132`), tetangga (`eda/110`, `133`), maju-waktu (`eda/140`), noise label (`eda/148`) semua menolak | jurnal siklus 23, 30 |
| Varians seed outer ±0,0015 di OOF pooled — sebesar ambang keputusan | Causilo 5 seed tunggal 0,5215–0,5258; set seed pohon v22 juga beruntung | `eda/122`, siklus 10–13, 31 |
| **Winner's curse** seleksi fitur dari banyak kandidat | 30 pasangan dari 7.080: +0,0036 (P 0,95) → bersarang −0,0005 | `eda/145`, `146` |
| KS pooled hanya peka di sekitar satu titik potong (rank ~0,69) | thin-view: segmen thin +0,0026 tapi pooled +0,0002; ekstrem terpisah sempurna | `eda/126`, `127`, `148`, `149` |
| Pohon jenuh | setiap gain per anggota (+0,0013…+0,0036) hilang di keluarga 6 GBDT | siklus 17, 20, 25, 26, 34, 35 |
| Data bersih | 0 baris setelah cutoff train/test; previous_applications tidak memuat aplikasi terpilih; tidak ada lantai noise label | `eda/139`, `148` |

## Kenapa stuck
- Fitur: semua kolom mentah 8 tabel sudah dipakai; scan residual vs pohon 1/228; ±20 keluarga konstruksi baru null.
- Model: pohon di plafon (hyperparameter datar, keragaman tambahan tidak menolong); ICL menambah di CV tapi tidak di public.
- Perkiraan plafon pendekatan ini di public **~0,564–0,565**. Gap ke top-1 0,0138 tidak terjelaskan oleh apa pun yang
  kita miliki.
- **Petunjuk eksternal baru (19 Okt malam, dari user):** tim teman di 0,57065 (#8, kemungkinan "BCC mau ke kopdes depok")
  memakai **ensemble Kumo 3 seed**, sementara Kumo-only kita di public 0,56056. Selisih +0,010 → setup Kumo kita
  (fitur `fm_select`, context, preprocessing) kemungkinan besar kurang optimal. Belum diuji (lihat `10_ide_belum_dicoba/`).

## Langkah berikutnya (detail `09_langkah_berikutnya/rekomendasi.md`)
1. **Diskusikan dulu dengan user** sebelum menjalankan apa pun (pertanyaan = jawab saja).
2. Kandidat terkuat yang belum diuji: perbaiki setup Kumo (VIP-COP feature/context optimisation, fitur & preprocessing
   untuk Kumo, ensemble 3 seed seperti tim teman) — `10_ide_belum_dicoba/`.
3. Banner tim (nomor peserta DSCxxxx + nama anggota) — butuh data user — lalu rename notebook
   `ParticipantNumber_TeamName_Notebook`, run ulang lokal, kumpulkan sebelum 30 Okt 23:59 WIB.
4. Setelah 21 Okt 06:00 (bila tanpa breakthrough): slot analisis sesuai rencana yang sudah ditetapkan.

## File penting
| File | Isi |
|---|---|
| `results/v31/submission.csv` | public terbaik 0,56405 |
| `results/v34/submission.csv` | resep notebook submission (public 0,56271) |
| `results/final/submission.csv` | output run lokal notebook submission 19 Okt (= v34, Spearman 1,000000) |
| `results/v22/v22/{gbdt,nn,fm}/` | OOF/test bersih resep v8 (dengan application_id, 3 seed) |
| `results/v28m/` | Kumo context penuh 5 fold (OOF/test, seed 42) |
| `results/v37`, `v38`, `screen/seeds/` | Causilo / Kumo / LGB multi-seed |
| `results/v42/` | kandidat probe yang sudah dipra-registrasi |
| `eda/valid2.py`, `eda/run_valid2.py`, `eda/blend2.py`, `eda/screen2.py`, `eda/120`, `eda/124` | protokol, driver, blend, evaluator |
