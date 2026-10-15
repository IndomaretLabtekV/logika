# Kronologi chat (UTC, WIB = UTC+7)

| Waktu (UTC) | Permintaan user | Yang dikerjakan / diputuskan |
|---|---|---|
| 12 Okt 22:44 | Instruksi besar pertama: pahami lomba, DEEP EDA dalam bentuk .py, preprocessing terverifikasi, duplikasi/leakage, SOTA (TabPFN-3.5, Causilo, LimiX-2), notebook T4x2, larangan run lokal | `eda/00`–`13`, `features.py` v1, notebook v1 + dry run. LB probing ditolak. LimiX-2 dikecualikan |
| 13 Okt 04:07 | Cek v1, submit atau tidak | submit v1 |
| 13 Okt 04:10 | v1 public 0,53139, top-1 0,57059; DEEP EDA lagi | `eda/20`–`27`, notebook v2 |
| 13 Okt 07:13 | Cek v2 untuk second submission | submit |
| 13 Okt 07:19 | v2 0,54192; perbaiki pretrained / cari model lain | `eda/30`–`32`, v3 (TabPFN konteks penuh, LimiX-2, TabM) |
| 13 Okt 12:08 | "5 jam ga beres beres", cek `log/` | diagnosis: LimiX hang; guard waktu diperbaiki, LimiX dibuang |
| 14 Okt 02:58 | v4 dibuat dengan Astra (jelek), cek `eda/deep/`; DEEP EDA + TabPFN-Rel, EXAONE, Mitra-v2, RT-PluRel | `eda/40`–`46` → **penemuan price premium** → notebook v5 |
| 14 Okt 06:00 | Cek v5 untuk first submission | submit |
| 14 Okt 09:05 | v5 0,56145 (bagus); cek `eda/v5_review/`, EDA lanjut | `eda/50`–`53`, notebook v6 (perbaikan kepatuhan) |
| 14 Okt 11:33 | Cek v6 untuk second submission | submit sebagai cek kepatuhan (bukan untuk gain) |
| 14 Okt 12:35 | `/goal`: v6 0,56123, cek `eda/v6_review`, cari cara ke top-1 0,57184, notebook RTX5090 vast.ai, "jangan stop sampai local val 0,54–0,56" | `eda/60`–`72` (semua negatif), notebook v7 vast.ai. Stop hook terus menolak karena target tidak tercapai |
| 14 Okt 14:16 | Perlu tunggu bundle test sebelum run v7? | tidak perlu (bundle test negatif) |
| 14 Okt 16:17 | vast.ai lambat; buat v7 untuk Kaggle T4x2, tanpa batas waktu | v7 versi Kaggle |
| 14 Okt 22:16 | Cek v7 untuk last submission | **jangan submit** (sama dengan v6) |
| 14 Okt 23:10 | "Stuck di v5–v7, masa nyerah", top-1 0,57458, cek `eda/v7_review`, DEEP EDA + audit v1–v7 + model lebih bagus | koreksi Astra diterima; `eda/80`–`83` (label historis), notebook v8 |
| 15 Okt 06:24 | v8 error setelah 4 jam, perbaiki supaya ada submission (belum submit sama sekali hari itu) | diagnosis TabICL OOM; blend lokal → `results/v8/submission.csv`; notebook diperbaiki |
| 15 Okt 06:39 | v8 lebih bagus dari v1–v7? layak submit? | sedikit terbaik tapi di dalam noise; submit |
| 15 Okt 07:52 | "Tetep stuck 0,56145", cek `eda/v8_review`, cari breakthrough (top-1 0,57493), model lain termasuk model China | `eda/90`–`96` (null kecuali logit stacking), `submission_v9_stack.csv`, notebook v9, riset web |
| 15 Okt 13:34 | Cek v9 untuk second submission | v9 sedikit di bawah v8-stack (−0,0019, noise); tetap disarankan submit supaya bisa dipilih (dihasilkan notebook utuh) |
| 15 Okt 13:53 | Kesal karena kata "final" terus; minta handoff lengkap | folder `handoff/` ini |
