# Aturan dan preferensi user (kumulatif, termasuk perubahan selama chat 24 Okt)

## Goal sesi (dipasang via `/goal`, 24 Okt 00:32 WIB) — inti
- Lanjutkan lomba; **public setinggi mungkin** (best saat itu v31 0,56405). Target lokal **pooled OOF KS 0,53–0,54**.
- Tugas: (1) jalankan resep acuan 5-fold lokal → OOF pooled + CSV test = baseline baru (kalau OOM cari setting yang
  muat: chunk/allocator/context, catat penyimpangan); (2) context optimization ala VIP-COP (arXiv 2605.12904) hanya
  varian yang belum diuji, tulis bedanya dari v48/v53/217 dulu; (3) semi-supervised (diizinkan panitia), seluruh
  pipeline SSL di dalam tiap outer fold; (4) ranking 3 kandidat terbaik berdasarkan OOF pooled.
- **Resep acuan (TLDR user):** fitur sebelum decision (~880–896), Kumo-Tabular large, **median impute**, maks 500
  kolom, e8; 3 view (capped500, uncapped, subset kecil) × 2 seed, rank-average sama bobot, dikunci sebelum lihat
  hasil; stratified 5-fold, reference populasi difit ulang per fold, test di-rank-average per fold lalu per run.
- **Validasi:** seleksi/tuning hanya di inner; outer hanya dinilai. Preprocessing & reference difit ulang per
  partition, tanpa cache global. Screening fold 0 dulu, lalu 5 fold. Per kandidat: OOF KS/AUC pooled & per fold,
  Δ vs baseline (fold sama), paired bootstrap P>0.
- **Submission (aturan awal goal):** sebelum 07:00 WIB berikutnya submit 3 kandidat terbaik satu per satu, prioritas
  OOF ≥0,53; kandidat <0,53 hanya kalau **jelas berbeda** dari submisi sebelumnya (sebutkan alasan). Tulis perkiraan
  public ± sd sebelum submit, analisis sesudahnya. Upload `results/vN/<nama>/submission.csv`; description = isi +
  pertanyaan riset.
- **Eksekusi:** semua run lokal (RTX 3070 Ti 8 GB), tanpa push kernel; satu job berat sekaligus, VRAM <8 GB,
  `taskset -c 0-11 nice -n 10`, `OMP_NUM_THREADS` 4–8; pantau suhu, **jeda kalau GPU >85 °C atau CPU >95 °C**; cache
  atomik (tmp+fsync+rename) dan bisa di-resume; tiap analisis = satu `.py` di `eda/`, cetak angka, gambar ke
  `eda/out/`; catat bukti & keputusan di `handoff/04_riwayat_versi/` (jurnal baru) dan `indeks_eksperimen.md`;
  resep dipromosikan → update `notebook/build_final.py`, uji ulang replay.
- **Larangan:** data eksternal, leakage masa depan (cutoff decision_datetime), LB probing/tuning ke public, menulis
  token HF/PriorLabs ke file/notebook, **git push**.
- **Gaya:** Bahasa Indonesia, jujur, klaim disertai angka. **Kalau user bertanya, jawab dulu, jangan langsung
  jalankan.** Pakai skill relevan (mlcraft, ml-training-recipes, 0-autoresearch-skill). Jangan berhenti di rencana;
  terblokir → lapor jujur.

## Perubahan/penegasan user selama chat (urut waktu)
| Waktu (WIB) | Pesan user (ringkas) | Efek |
|---|---|---|
| 24 Okt 00:39 | "Laptop tadi sempat nge-hang… lanjutkan prosesnya dari yang terakhir" | resume dari state terakhir |
| 24 Okt ~11:05 | "Hari ini sudah reset submission, gunakan 3 submission tersebut **jika memang sudah breakthrough atau lokal val sudah 0,53–0,54, yang penting lebih bagus dari highest sekarang**" | **aturan submit diperketat** (menggantikan "submit 3 kandidat terbaik") |
| 24 Okt ~11:25 | "Pikirkan pendekatan lain… cara outstanding, lu bebas melakukan apapun" | eksplorasi bebas (tetap dalam larangan) |
| 24 Okt ~12:05 | "Masih 6 hari… kenapa harus stop? Pikirkan terus… **cari write-up atau paper yang bisa kita tiru**" | riset literatur + teknik juara |
| 24 Okt ~14:20 | "Lanjutkan dari pekerjaan terakhir. Laptop sebelumnya hang" | resume |
| 24 Okt ~23:15 | "Val KS tertinggi berapa? public-nya?" | dijawab (tanpa run) |
| 24 Okt ~23:30 | "Kenapa ga coba submit yang dapet 0,5292 dulu?" | dijawab; persiapan tanpa upload |
| 24 Okt ~23:40 | "**Coba submit dulu aja yang belum disubmit dan emang dapet val tinggi, terutama dari temen gua**" | otorisasi eksplisit → submit goal054 (0,56470) & K64 2-seed (0,56168) |
| 24 Okt ~23:55 | "Kenapa ga lanjutin dari yang temen gua aja? OOF-nya pakai apa?" | dijawab |
| 24 Okt ~23:58 | "Seednya berapa aja?" | dijawab (31027 dan 2026+fold) |
| 25 Okt 00:00 | "Kenapa ga merge kerjaan teman dengan gua?… sudah ga ada val tinggi lagi kan?" | dijawab; rencana merge |
| 25 Okt 00:02 | "Tapi di repo udah ke gabung kan?" | dijawab (file iya, model belum) |
| 25 Okt 00:03 | "Rapihin repository, hapus yang ga penting, buat satu folder serah-terima selengkap mungkin, bagi beberapa folder, tanpa notebook/results" | **tugas yang menghasilkan folder ini** |

## Aturan dari memory dan handoff lama yang masih berlaku
- **Pertanyaan = jawab saja**; jangan menjalankan apa pun karena pertanyaan (memory `question-means-answer-only`).
- **Jangan menyebut "final"** (user marah bila berulang menyebut "final picks/submission final").
- Kuota Kaggle kernel habis → semua lokal, jangan push kernel.
- SSL/pseudo-label boleh (data panitia), pipeline di dalam outer fold; transduktif harus dinyatakan dan punya kontrol.
- User menghargai keputusan tegas (submit/jangan) dengan angka; tidak suka kesan "menyerah" tetapi tetap jujur.
- Aturan lama "submit hanya breakthrough >0,003, 5-fold×3 seed, P≥0,9" (21 Okt) **diganti** oleh aturan goal
  24 Okt lalu aturan user 11:05 di atas.
