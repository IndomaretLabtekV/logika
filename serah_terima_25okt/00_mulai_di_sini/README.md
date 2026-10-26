# Serah terima chat 23–25 Oktober 2026 — mulai di sini

Folder ini merangkum **seluruh** chat sesi 24 Okt 00:30 WIB – 25 Okt 00:05 WIB (lomba DSC LOGIKA UI 2026),
ditambah ringkasan riwayat proyek sebelumnya, supaya chat baru bisa melanjutkan tanpa membaca ulang transkrip.
Notebook dan folder `results/` sengaja tidak disalin; yang ada di sini penjelasan, angka, path, dan keputusan.

## Cara membaca (urut)

| Folder | Isi |
|---|---|
| `00_mulai_di_sini/` | ringkasan 1 halaman (file ini) dan prompt siap-tempel untuk chat baru |
| `01_konteks_lomba_dan_aturan/` | kompetisi, metrik, kuota, leaderboard, **semua aturan user** (termasuk yang berubah selama chat) |
| `02_kronologi/` | timeline menit-ke-menit sesi ini + ringkasan riwayat 11–23 Okt |
| `03_posisi_sekarang/` | skor & semua submisi, kandidat OOF, status notebook, apa yang sedang berjalan |
| `04_eksperimen/` | setiap eksperimen sesi ini: tujuan, desain, angka, kesimpulan, path |
| `05_riset_dan_temuan/` | temuan utama, literatur, detail pipeline teman (goal054) dan pipeline DSC0152 |
| `06_kesalahan_dan_pelajaran/` | kesalahan, yang tidak efektif, insiden laptop hang |
| `07_langkah_berikutnya/` | rencana merge pipeline teman + pipeline user, daftar pekerjaan tersisa |
| `08_kode_dan_infrastruktur/` | peta script `eda/212–249`, cara menjalankan, folder hasil, catatan pembersihan repo |

## Tiga sumber pipeline dalam satu repo (tim yang sama, kata user)
| Sumber | Lokasi | Public terbaik | OOF terbaik |
|---|---|---:|---:|
| Pipeline utama (user) | `eda/`, `results/v2x–v69` | 0,56405 (v31) | K64 2-seed 0,52740 (seed 42) |
| Pipeline "teman" (Kumo V5 + spline + offset) | `scripts/*.py`, `artifacts/optimization/goal054/`, `results/teman/` | **0,56470** (goal054) | 0,52930 (seed 31027) |
| Pipeline **DSC0152** (root + Agas + Dimaz), diintegrasikan 25 Okt | `scripts/dsc0152/`, `notebook/dsc0152/`, `results/dsc0152/`, `docs/dsc0152/` | 0,56424 (Agas v66_04) | AUC ±0,848 (10–15 fold) |
Nama notebook deliverable sumber DSC0152: `DSC0152_kali ini beneran deh_Notebook.ipynb` → nomor peserta **kemungkinan
DSC0152** dan nama tim resmi "kali ini beneran deh" (konfirmasi ke user; nama Kaggle sekarang sementara).

## Ringkasan 1 halaman (per 25 Okt 00:05 WIB)

- **Deadline notebook: 30 Okt 2026 23:59 WIB.** Kuota submit 3/hari, reset 07:00 WIB. Window 24 Okt 07:00 → 25 Okt
  07:00: **2 terpakai, 1 tersisa** (tidak dipakai saat sesi berakhir).
- **Public terbaik tim sekarang: 0,56470** — kandidat teman `goal054 kumo_offset`
  (`results/v70/goal054_kumo_offset/submission.csv`, ref Kaggle 57600249). Sebelumnya v31 0,56405.
- **Val (pooled OOF KS) tertinggi:** teman goal054 kumo_offset **0,52930** (fold seed 31027); di protokol user
  (fold seed 42) **K64 2-seed 0,52740** (public 0,56168 — tidak naik). Target user 0,53–0,54 **belum tercapai**.
- Leaderboard 24 Okt 15:00: top-1 0,57788, batas top-8 ≈0,5713. Tim kita ≈ peringkat 30-an.
- Sesi ini menguji **>30 eksperimen** (resep acuan, Kumo context/lebar/estimator/seed, SSL, teknik juara AmEx/Home
  Credit, CoLES, GRU-NN, mekanisme generator, stacking, bobot kelas). **Tidak ada** yang menaikkan OOF di atas blend
  terbaik; Kumo-only strict mentok ±0,562 public.
- **Kesalahan terbesar sesi ini:** terlalu lama memoles pipeline user sendiri dan baru di akhir menyadari pipeline
  teman (Kumo V5 + spline + koreksi offset laten) adalah yang terbaik di public. **Pipeline teman dan user belum pernah
  digabung secara model** (repo sudah satu, tapi OOF/test per-fold teman tidak tersimpan dan fold seed berbeda).
- **Langkah berikutnya yang direkomendasikan:** merge — jalankan ulang resep teman di fold seed 42 (fitur V5 per fold
  sudah ada di `results/v56/teammate_features/outer_s42_f*/`), blend dengan keluarga pohon & Causilo user dengan bobot
  dikunci, terapkan koreksi offset laten ke Kumo user. Detail: `07_langkah_berikutnya/rencana_merge.md`.
- **Laptop hang 4× sejak 23 Okt malam** (1× 23 Okt 23:49, 3× pada 24 Okt; termasuk sekali dengan guard termal aktif dan suhu normal) — penyebab belum
  diketahui; job berat sebaiknya dijalankan saat user bisa memantau. Detail: `06_kesalahan_dan_pelajaran/insiden_hardware.md`.
- Menunggu dari user: **nomor peserta** (banner/nama file notebook), keputusan resep notebook (sekarang replay v59;
  public terbaik dari goal054 teman), izin menjalankan merge.

## Dokumen lama yang tetap relevan
- `handoff/` (folder lama, 12–24 Okt): `handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md` adalah jurnal mentah
  sesi ini (semua angka asli). `docs/integration/indeks_eksperimen.md` = indeks semua eksperimen proyek.
- `artifacts/optimization/continuation_23okt/laporan_riset.md` (laporan 23 Okt), `continuation_24okt/` (protocol &
  amendemen yang dikunci sebelum skor).
