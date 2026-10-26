# Kesalahan sesi ini (jujur) dan pelajarannya

## Kesalahan strategi
1. **Mengabaikan pipeline teman terlalu lama.** Hampir seluruh hari dipakai memoles Kumo strict user (public mentok
   0,562), padahal pipeline teman (Kumo V5 + spline + offset laten) sudah 0,56393 public dan punya kandidat OOF 0,5293
   yang belum disubmit. Ditemukan baru 23:30 setelah user bertanya; disubmit → **0,56470 (rekor)**.
   Penyebab: terpaku pada kontrak strict 23 Okt yang membuang fitur V3/V4/V5 (reference harga lintas customer — padahal
   tanpa label, bukan kebocoran target), menganggap fold seed berbeda = "tidak bisa dibandingkan", dan menganggap env
   Python 3.11 + tidak adanya OOF teman sebagai hambatan.
   **Pelajaran:** inventaris semua kandidat tim (termasuk CSV yang belum disubmit) di awal sesi; gabungkan pipeline,
   jangan memilih satu.
2. **Menganggap repo "sudah digabung" = model sudah digabung.** Konsolidasi 21 Okt hanya file; model belum pernah
   di-blend.
3. **Terlalu cepat menyimpulkan plateau** (sekitar 11:00) sebelum mencoba literatur dan pipeline teman; user harus
   mendorong dua kali. Hook Stop juga berulang memaksa lanjut.
4. **Resep acuan dijalankan apa adanya** walau 2 dari 3 view sudah diragukan (context 8k, seleksi union); 1–2 jam GPU
   terbuang sebelum diagnosis. Pelajaran: preflight + diagnosis satu fold sebelum komit 5-fold untuk resep baru.
5. **Screening satu fold dipercaya sesaat** (series OOF cicilan +0,0055 f0 → −0,0013 5f). Pelajaran: kandidat murah
   langsung 5-fold.
6. **Perkiraan public pertama untuk goal054 (0,5647 ± 0,0027) terlalu optimis**; setelah memeriksa ρ 0,99965 dengan
   submisi 0,56393, dikoreksi menjadi 0,5643 ± 0,0015 (hasil 0,56470). Pelajaran: selalu cek kemiripan rank terhadap
   submisi lama sebelum memperkirakan.

## Kesalahan teknis (sudah diperbaiki)
| Kesalahan | Akibat | Perbaikan |
|---|---|---|
| `pgrep -f` / `pkill -f` ikut menangkap shell tool sendiri (command line memuat nama pola) | pkill membunuh shell; antrean baru memulai fit paralel ±30 dtk (2 job GPU); guard menghitung pids=2 | pola ter-anchor `'^[^ ]*python3? 21[2]_strict'` (trik `[2]`), kill dengan PID eksplisit |
| Mengedit skrip bash antrean yang sedang berjalan / PID ganda di heredoc | skrip rusak (`while kill -0 9240\n9326`) | buat skrip antrean baru, tunggu PID eksplisit |
| Manifest 222 dibandingkan dict bertuple vs JSON list | arm M_ctx8192 gagal "changed run" | normalisasi `json.loads(json.dumps(run))`, manifest lama disimpan `manifest_v1_before_compare_fix.json` |
| `PYTORCH_CUDA_ALLOC_CONF` diset setelah CUDA init dalam satu proses | setting tidak berlaku | satu arm uncached per proses |
| Polars `int_range`/`len` uint32 → `n − L` underflow (240) | IndexError pack sequence | cast Int64 |
| `sed` replace `'    paused = False\n'` juga mengenai baris berindentasi lebih dalam | guard IndentationError | edit manual |
| `np.save(open(t,'wb'))` tanpa close sebelum fsync (rencana awal 231) | potensi file kosong | tulis bytes via BytesIO |
| `valid2.run` menulis parquet non-atomik | OOF 249 0 byte setelah hang | (belum diperbaiki di valid2; hati-hati) |
| Guard termal awal pause tiap spike 97 °C | job hampir tak jalan | aturan 3-sampel, lalu diperketat 1-sampel setelah CPU >95 °C 73% waktu |
| Me-restart guard saat job ter-SIGSTOP | job bisa tertinggal berhenti | handler SIGTERM → SIGCONT semua target sebelum keluar |
| Monitor CPU 100 °C terlalu cerewet | alert palsu | alert hanya bila panas **dan** tidak di-pause 2 sampel |
| Hitungan hang di chat ("4× hari ini") | tidak akurat | benar: 4× sejak 23 Okt 23:49 (1 + 3 pada 24 Okt) |

## Penyimpangan protokol yang dicatat (bukan kesalahan tersembunyi)
- Resep acuan: uncapped 5-fold & seed 2027 resep tidak dijalankan; median impute tidak ada di protocol 23 Okt (diuji
  terpisah, null); wide view context 8.192 karena VRAM.
- Urutan antrean diubah setelah melihat skor cap500 (kandidat terkunci tidak berubah).
- 231 8×e1: tiap member subsample context sendiri (context bagging).
- 242 GRU-NN: model inner-stopped menilai outer tanpa refit.
- Screening inner fold 0 memakai inner yang sudah pernah terekspos (217, v49) → bukti development, bukan holdout.
