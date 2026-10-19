# DSC LOGIKA UI 2026 — Credit Risk (KS)

Prediksi pinjaman yang mencapai DPD ≥ 90 dalam 12 bulan. Metrik: KS. Deadline notebook: 30 Oktober 2026, 23:59 WIB.
Status, riwayat versi v1–v36, temuan, dan langkah berikutnya ada di **[`handoff/`](handoff/README.md)**. Baca itu dulu.

Public terbaik 0,56405 (v31: rank blend 6 GBDT + Causilo + NVIDIA Kumo-Tabular), tim Kaggle "lagi cape lomba asli",
#33/118 per 19 Okt. Notebook submission: `notebook/build_final.py` → `notebook/final.ipynb` (reproducible di Kaggle dan
lokal; output lokal di `results/final/`). Ide berikutnya: `handoff/10_ide_belum_dicoba/`.

| Path | Isi |
|---|---|
| `handoff/` | Handoff lengkap: status, aturan, tools, data, fitur, riwayat versi, katalog eksperimen, model, validasi, pelajaran, rekomendasi |
| `docs/context.md` | Overview, dataset, dan rules kompetisi (salinan halaman Kaggle) |
| `docs/data_guide.md`, `docs/reports/` | Panduan dataset dan laporan EDA awal (repo teman) |
| `data/` | CSV kompetisi (tidak di-commit). Unduh dari Kaggle ke sini |
| `eda/` | `features.py` (sumber fitur tunggal), `valid2.py` + `run_valid2.py` (validation_v2), `blend2.py`, script analisis `00`–`152`, output di `eda/out/` |
| `eda/deep/`, `eda/v5_review/` … `eda/v8_review/` | Audit independen per versi |
| `eda/baseline_duckdb/`, `eda/risk_exploration/` | EDA SQL baseline awal (DuckDB) |
| `notebook/build_final.py`, `final.ipynb` | Notebook submission (Kaggle T4×2 atau lokal 1 GPU) |
| `notebook/build_vNN.py`, `vNN.ipynb` | Builder dan notebook eksperimen v1–v33 |
| `results/` | Prediksi OOF/test per model (dengan application_id), kandidat submission, log run |
| `submissions/probe/` | Probe submission 16 Okt |
| `ref/notebook/` | Skill dan konvensi penulisan notebook |
| `research/` | Indeks paper metode (PDF diunduh ulang dengan `download_papers.py`) |

## Setup lokal (CachyOS, RTX 3070 Ti 8 GB)
```bash
uv venv .venv && uv pip install --python .venv/bin/python polars==2.0.0 numpy scipy matplotlib scikit-learn \
    lightgbm xgboost catboost torch tabm==0.0.3 rtdl_num_embeddings==0.0.12 structured-data-models \
    pytabkit skorch nbclient ipykernel
uv pip install --python .venv/bin/python --no-deps causilo
cd eda && ../.venv/bin/python run_valid2.py ../results/screen/check lgb --folds 0 --seeds 42   # bangun cache + screening
```
Versi persis dan environment `.venv-zoo` (AutoGluon/TabArena): `handoff/02_proyek_dan_tools/validation_v2_dan_tool.md`.
