# Lokal (RTX 3070 Ti Laptop) vs Kaggle (T4): mana yang lebih cepat

Jawaban singkat: **per job, lokal 2,3–3× lebih cepat** dari satu T4. Kaggle hanya menang kalau butuh paralel banyak,
VRAM > 8 GB, atau laptop harus bebas. Pola tercepat yang pernah dipakai: **bagi fold** (sebagian lokal, sebagian Kaggle,
jalan bersamaan), lalu gabung dengan `eda/merge_runs.py`.

## Waktu terukur (protokol validation_v2 yang sama)
| Job (per fold, termasuk prediksi 30k test) | Lokal 3070 Ti | Kaggle T4 | Rasio |
|---|---|---|---|
| Kumo large, 2 draw context 20k | ~1.700 s | ~4.100 s | 2,4× |
| Kumo large, context penuh 56k | 3.333–3.850 s | 8.230–8.490 s | 2,3× |
| Causilo (4 est, 100 fitur) | ~115–145 s | ~375 s | ~3× |
| seg_causilo | – | ~205 s | – |
| Kumo large, context 70k, 30k test sekali | – | 16.172 s (peak 6,5 GB) | – |
| TabFM fp32 ctx 8k / 4k (fold 0) | OOM | 6.848 s / 3.207 s | – |
| 6 GBDT × 1 seed, fold 0 (screening) | 4,4 menit | – | – |
| TabM 1 seed, fold 0 | 1,5 menit | – | – |
| v22 penuh: 6 GBDT × 3 seed × 5 fold | – | 78 menit (GPU0) | – |
| v22 penuh: TabM 3 seed / Causilo 1 seed × 5 fold | – | 48 / 31 menit (GPU1) | – |
| Notebook final penuh | – | 15.484 s (T4×2) | – |

GBDT penuh tidak pernah diukur lokal di sesi ini, jadi rasio untuk pohon belum ada.

## Kapan pakai mana
| Situasi | Pilihan |
|---|---|
| Screening 1 fold | lokal |
| Full fold 1 model ICL (Kumo/Causilo) | lokal; atau bagi fold lokal + Kaggle supaya selesai lebih cepat |
| Model butuh > 8 GB VRAM (TabFM, LimiX, TabPFN-3 besar, Mitra) | Kaggle (T4 15 GB) |
| Beberapa run independen sekaligus | Kaggle: 2 GPU per sesi × maks 2 sesi GPU bersamaan = 4 T4 ≈ 1,6× satu 3070 |
| Laptop perlu dipakai / dimatikan | Kaggle (output baru bisa diambil setelah kernel selesai) |
| Uji reproducibility notebook final | wajib Kaggle |

## Biaya tersembunyi Kaggle
- Antre dan setup tiap kernel (pip install + bangun fitur ±10 menit).
- Batas 12 jam per sesi, 30 jam GPU per minggu, maksimal 2 sesi GPU batch bersamaan (push ke-3 ditolak
  "Maximum batch GPU session count of 2 reached").
- Output hanya tersedia setelah kernel selesai; status sementara CANCEL_ACKNOWLEDGED bisa menyesatkan (cek ulang).
- Access token CLI kedaluwarsa; perlu wrapper (lihat `kaggle_cli_kernel_dan_submit.md`).
- Dua T4 bersamaan masih ≈ 0,87× kecepatan satu 3070 untuk Kumo (2 ÷ 2,3).

## Contoh nyata
- v33 (Kumo ctx penuh seed generator kedua): fold 0–2 lokal + fold 3–4 Kaggle bersamaan ≈ 3 jam. Lokal saja ≈ 5 jam;
  Kaggle saja (3 fold di satu GPU) ≈ 7 jam.
- v26 (Kumo 2 draw 20k): lokal dan Kaggle sama-sama menjalankan 5 fold (karena status Kaggle salah dibaca) → komputasi
  ganda, tapi jadi bukti reproducibility lintas mesin (selisih ≤ 0,0004 per fold, Spearman OOF 1,0000).

## Angka tambahan 18–19 Okt (lokal RTX 3070 Ti)
| Job | Waktu |
|---|---|
| Kumo large, context 56k, 14k query (fold, tanpa test) | 1.050–1.230 s |
| Kumo-70k (context 70k, 30k test) | 6.212–6.478 s, VRAM 5,0 GB |
| Kumo forward context 10k/20k + 3.000 query, e1 | 3,0 / 4,2 s |
| Causilo 5 fold (seed tunggal, tanpa test) | ~10 menit (114–133 s/fold) |
| LGB 5 fold (20 thread) | ~4 menit; 6 GBDT × 3 seed dengan test ~1,5–2 jam |
| TabR fold | 74 s |
| Notebook submission penuh (Kumo-70k + 6 GBDT × 3 seed + Causilo) | 10.114 s (2,8 jam) vs Kaggle T4×2 4,3 jam |
