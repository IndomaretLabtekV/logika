# v16 dan v17: TabICLv2 dengan query di-chunk (16 Okt, Kaggle T4×2)

Kernel: `riantamaputra/logika-v16-tabicl`, `riantamaputra/logika-v17-tabicl-full`. Kode `notebook/v16_experiment.py`,
`notebook/v17_experiment.py`. Hasil `results/v16/v16/`, `results/v17/v17/` (oof.parquet + test.parquet kolom `tabicl`).

## Latar
v8 mati di TabICL karena `whole_query=True` mengirim ~44k baris query sekaligus (OOM tidak tertolong dengan mengecilkan
context). Perbaikan: query di-chunk 2.000 baris, `kv_cache=False`, `batch_size=1`, 1 thread per GPU (fold round-robin
di 2 T4), satu kolom flag per pola missing yang unik, seleksi fitur di train fold (KS univariat + greedy |ρ| < 0,95).

## Hasil
| Run | Setup | Waktu | OOF KS | + stack v8 |
|---|---|---|---|---|
| v16 | ctx 40k (sampel stratified), 100 fitur, 4 estimator | ~16 menit/fold | 0,5079 | +0,0006 (30/40), tapi bobot negatif & in-sample stack turun |
| v17 | ctx = seluruh train fold (≤56k), 150 fitur, 8 estimator | ~1 jam/fold (total ~3 jam) | 0,5133 (fold .5205 .5213 .5183 .5189 .5017) | lihat bawah |

v17 di stack v8:
- held-out 20×2: +0,00062 (36/40); CV stack 10×5: +0,00072 (10/10);
- bootstrap baris (sampling): P(>0) **0,69** → tidak signifikan;
- bobot stack TabICL17 **negatif (−0,22)** dan bobot TabPFN naik 0,13 → 0,26: TabICL dipakai sebagai "kontras"
  terhadap TabPFN (Spearman 0,984), bukan sebagai sinyal sendiri;
- dengan bobot positif di blend rank v8, TabICL17 menurunkan OOF (5%: −0,0005; 20%: −0,0018);
- Spearman test vs v8-stack 0,99984.

## Disubmit sebagai probe S3 → public 0,55778 (−0,0037 vs v8)
Gain OOF via bobot negatif tidak terbawa ke public. Lihat `07_validasi_dan_leaderboard/probe_submission_16okt.md`.

## Pelajaran
- TabICL **bisa** jalan di T4 asal query di-chunk; konteks penuh menaikkan KS (0,508 → 0,513), tapi tetap di bawah
  GBDT/TabM dan sangat berkorelasi dengan TabPFN.
- Jangan percaya gain stack yang datang dari bobot negatif pada model yang sangat berkorelasi.
- Kuota GPU mingguan Kaggle (30 jam) habis setelah v14–v17.
