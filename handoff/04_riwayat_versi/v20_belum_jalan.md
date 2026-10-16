# v20: Causilo/TabPFN 16 estimator (dibuat, TIDAK pernah jalan)

Kode `notebook/v20_experiment.py`, builder `notebook/build_v20.py`, notebook `notebook/v20.ipynb`, metadata kernel
`results/v20k/kernel-metadata.json` (slug `riantamaputra/logika-v20-fm16`, T4).

## Ide
v8 menjalankan TabPFN dan Causilo dengan konteks penuh (56k) tapi hanya 4 estimator dan 104 fitur (~5–6 menit/fold,
dari log `results/v8/v8_run.ipynb`). Rencana: 16 estimator, lalu Causilo 200 fitur, query di-chunk 5.000.
Urutan konfigurasi: causilo16, tabpfn16, causilo16_f200; masing-masing disimpan terpisah.

## Status
- Push ditolak: "Maximum weekly GPU quota of 30.00 hours reached" (16 Okt).
- Loop retry push dihentikan atas permintaan user ("batalin semua").
- TabPFN-3.5 gated: notebook membaca secret `TABPFN_TOKEN`/`HF_TOKEN` lewat `kaggle_secrets`. Kernel yang dipush lewat CLI
  kemungkinan **tidak** punya secret terpasang (v8 dijalankan dari UI). Kalau token tidak ada, hanya Causilo yang jalan.

## Prioritas sekarang
Rendah. Setelah temuan bias OOF, yang lebih penting adalah rerun pipeline dengan early stopping bersih
(`09_langkah_berikutnya/rekomendasi.md`). Kalau tetap mau dijalankan, nilai dengan OOF bersih, bukan OOF v8.
