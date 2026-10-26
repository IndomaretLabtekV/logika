# Insiden laptop hang dan mitigasinya

## Kejadian
| # | Waktu (WIB) | Job yang berjalan | Kondisi | Akibat |
|---|---|---|---|---|
| 1 | 23 Okt ~23:49 | 217 VIP baris (Kumo GPU) | GPU 87 °C, CPU zona maks 103 °C, VRAM 5,0 GB | hard freeze + reboot; cache 0 byte dikarantina (`results/eda_23okt/row_vip/quarantine_crash_2026-10-23/`) |
| 2 | 24 Okt ~00:35 | 219 build pool (polars CPU, ~1,4 core, 0,8 GB RAM), **tanpa job GPU**, browser terbuka | journal berhenti 00:32 tanpa error kernel | force shutdown; `pool_outer_f2/manifest.json` 0 byte → `results/v60/strict_tfm/quarantine_crash_2026-10-24/` |
| 3 | 24 Okt ~12:20 | 237 series OOF (LightGBM 8 thread, 4,36 juta baris), **tanpa guard** | – (boot baru 13:29) | hasil 237 hilang, diulang |
| 4 | 24 Okt ~18:09 | 249 (LightGBM 4 thread) **dengan guard 1-sampel aktif** | telemetri terakhir 18:10:00: CPU 61 °C, GPU 55 °C (normal) | boot baru 20:25; `oof_lgb_tsel30.parquet` 0 byte |

## Observasi termal
- i7-12700H: satu P-core naik ke 97–104 °C dalam ±15 dtk setiap kali beban (bahkan satu thread boost); turun ke 60–80 °C
  dalam 5 dtk saat pause. Dengan aturan 3-sampel, CPU >95 °C selama 73% waktu (maks 103 °C).
- GPU umumnya 75–85 °C saat Kumo; peak VRAM run terbesar 6,10 GiB (ctx16k fraction 0,85).

## Kesimpulan saat ini
- Hang #2 terjadi tanpa GPU dan #4 terjadi dengan suhu normal → hipotesis "termal tanpa throttle" **melemah**.
  Penyebab belum diketahui (kandidat: driver NVIDIA open 615.78 / nvidia-powerd (log "JPAC is not created" berulang),
  kwin/Wayland ("Failed to allocate graphics buffer"), daya/VRM). Journal sering berhenti beberapa menit sebelum hang
  sehingga bukti kernel tidak ada.

## Mitigasi yang dipakai (pertahankan)
1. `eda/thermal_guard.py <telemetry.csv> <pola...>`: SIGSTOP bila GPU >85 °C atau CPU zona termal >95 °C (sekarang
   **satu sampel**), SIGCONT di bawah 78/85 °C; telemetri fsync tiap 5 dtk; SIGTERM → SIGCONT semua target.
   Pola wajib ter-anchor, mis. `'^[^ ]*python3? 22[4]_strict'`.
2. Job CPU 4 thread (`OMP_NUM_THREADS=4`, `n_jobs=4`), `taskset -c 0-11 nice -n 10`, satu job berat sekaligus.
3. Cache atomik tmp+fsync+rename di semua script baru (212, 222, 224, 231, 237, 240, 241, 246…); `os.sync()` setelah
   build pool. (`valid2.run` masih non-atomik.)
4. Job panjang dipecah per fold/member supaya hang hanya kehilangan satu unit.
5. Saran: jalankan job berat saat user bisa memantau; tutup browser/aplikasi GPU desktop; pertimbangkan pendinginan
   eksternal; bila hang terus, uji tanpa nvidia-powerd atau dengan power profile lebih rendah (keputusan user).
