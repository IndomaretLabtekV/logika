# Run 18 Okt sore – 19 Okt (lokal; tidak ada versi Kaggle baru)

Semua run memakai validation_v2 (`eda/run_valid2.py`), seed outer 42/7/2026 kecuali disebut, OOF + test berisi
application_id. Detail angka dan alasan tiap run: `jurnal_18okt_sore.md` (nomor siklus di kolom terakhir).

| Folder | Isi | Hasil | Siklus |
|---|---|---|---|
| `results/v37` | Causilo seed 7 + 2026 (dengan test) | OOF 2 seed; 3 seed 0,52418 < seed 42 0,52583 | 10 |
| `results/v38` | Kumo ctx penuh seed 7 + 2026 (tanpa test) | 0,52582; 3 seed 0,52570 | 12 |
| `results/v39` | probe submission `trees_only`, `causilo3_only`, `lr_q_only` (+ subfolder `*/submission.csv` yang diunggah) | public 0,56337 / 0,55333 / 0,55835 | probe 19 Okt |
| `results/v40` | TabR 5 fold × 3 seed | 0,51217, null | 15 |
| `results/v41` | Kumo-70k draw estimator g1/g2 + Causilo seed 1/2 dengan test; `v31_lowvar.csv` | Spearman antar draw 0,9984; kandidat varian rendah | malam 18→19 |
| `results/v42` | kandidat probe dipra-registrasi: `cat_only`, `lgbxgb_only`, `trees_kumo_50`, `tabm_only`, `trees_deltas`, `trees6seed`, `kumo70k3_only` | belum disubmit | 19 Okt |
| `results/v43` | 6 GBDT + 333 delta snapshot, 3 seed, dengan test | keluarga 0,52398 (null) | 26 |
| `results/v44` | 6 GBDT seed 1/2/3 dengan test | 0,52400; 6 seed 0,52324 | 31 |
| `results/v45` | lgb_x, xgb_d8, cat_d8 (3 seed, dengan test) | keluarga 9 null | 34 |
| `results/final` | **run lokal notebook submission** (pipeline, run, `submission.csv`, `executed_notebook_local.ipynb`) | Spearman 1,000000 vs v34 | 19 Okt |
| `results/finalk` | metadata kernel `riantamaputra/logika-final` (push versi 2 dibatalkan) | – | – |
| `results/screen/stk_f0` | stacked Causilo/Kumo fold 0 | null | 4 |
| `results/screen/fbag` | Causilo feature bagging r1–r3 | null | 8 |
| `results/screen/re_base`, `re_x4` | LGB/LGB_s/Causilo ± re_* | null | 6 |
| `results/screen/seeds/s*` | LGB/LGB_s seed 42,7,2026,1–5; Causilo seed 7,2026,1,2 (tanpa test) | kurva averaging | 13 |
| `results/screen/tabr_f0` | TabR fold 0 | 0,51833 | 15 |
| `results/screen/gainsel` | Causilo fm_select gain | 0,52390 | 16 |
| `results/screen/thinview` | thin-view LGB/LGB_s/XGB/Cat/Causilo | null | 17 |
| `results/screen/peer6*`, `peer_m4`, `peer_hr2` | LGB/LGB_s + 6 fitur peer (seleksi bocor) | keluarga null | 20 |
| `results/screen/lgb_nested` | seleksi konfigurasi LGB bersarang | null | 21 |
| `results/screen/pca20`, `slr` | PCA / LR stack untuk pohon | null | 25 |
| `results/screen/snapd*`, `snap2_s*` | 6 GBDT / LGB+XGB + delta snapshot | null | 26, 35 |
| `results/screen/pw30` | LGB + 30 pasangan (seleksi bocor) | +0,0036 palsu | 28 |
Analisis tanpa folder results (output di `eda/out/`): 112, 113, 114, 117, 118/119 distilasi, 121, 122, 123, 125–128,
130–149, 152 (dibatalkan).
