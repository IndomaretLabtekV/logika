# Yang berhasil

| Hal | Bukti | Kenapa berhasil |
|---|---|---|
| Fitur perilaku dari residual (v2) | ablation +0,0046, 5/5 fold; public +0,011 | menangkap keparahan telat, burst e-commerce yang halus, cash flow, dan inquiry eksternal |
| **Price premium `pp_*` (v5)** | ablation **+0,014**, 5/5 fold; public +0,016; thin-file +0,023 | sinyal baru yang hampir ortogonal (KS 0,368); berlaku juga untuk thin-file karena e-commerce cakupannya 100% |
| Referensi harga expanding per bulan (v6) | KS dipertahankan (0,3689) | membuat fitur patuh cutoff tanpa kehilangan sinyal |
| TabM dengan nn_prep | model tunggal terbaik sejak v3 | NN cocok dengan struktur faktor laten aditif |
| Varian shallow trees | lgb_s / xgb_s setara atau sedikit lebih baik | risiko aditif di log-odds |
| Causilo dengan pandas categorical | 0,4953 → 0,5226 → 0,524 | representasi yang benar untuk FM |
| TabPFN di fitur manual | 0,4912 (DFS) → 0,5186 | FM butuh fitur yang informatif, bukan DFS mentah |
| Logit stacking | +0,0008 sampai +0,0017 held-out | blend di ruang log-odds sesuai dengan struktur data |
| Disiplin verifikasi | recompute independen dan injeksi masa depan di setiap versi; leak test di notebook | tidak ada leakage yang lolos (kecuali referensi harga global v5, ditemukan Astra) |
| Safety-net submission di tengah notebook | v9 | run yang mati di tengah tetap menghasilkan file |
| Guard deadline di dalam fold, run FM paralel di 2 GPU | v7+ | menghindari hang seperti v3 |
| Simpan `oof_all.parquet` dengan ID | v7+ | memungkinkan blend, stacking, dan audit lokal |
