# DSC LOGIKA UI 2026 — Credit Risk (KS)

Prediksi pinjaman yang mencapai DPD ≥ 90 dalam 12 bulan. Metrik: KS. Deadline notebook: 30 Oktober 2026, 23:59 WIB.
Status, riwayat versi v1–v9, temuan, dan langkah berikutnya ada di **[`handoff/`](handoff/README.md)**. Baca itu dulu.

| Path | Isi |
|---|---|
| `handoff/` | Handoff lengkap: status, aturan, data, fitur, riwayat versi, katalog eksperimen, model, pelajaran, dan rekomendasi |
| `docs/context.md` | Overview, dataset, dan rules kompetisi (salinan halaman Kaggle) |
| `docs/data_guide.md` | Panduan dataset dan saran keluarga fitur |
| `docs/reports/` | Laporan EDA awal (`EDA_RESEARCH_REPORT.md`) dan risk exploration (`RISK_EXPLORATION_REPORT.md`) |
| `data/` | CSV kompetisi (tidak di-commit). Taruh file dari Kaggle di sini |
| `eda/` | Pipeline utama: `features.py` (sumber fitur tunggal), `nn_prep.py`, `common.py`, script analisis `00`–`96`, output di `eda/out/` |
| `eda/deep/`, `eda/v5_review/` … `eda/v8_review/` | Audit independen per versi |
| `eda/baseline_duckdb/` | EDA dan fitur SQL baseline awal (DuckDB, 63 fitur) |
| `eda/risk_exploration/` | Risk exploration lanjutan (DuckDB SQL), output di `outputs/04_risk_exploration/` |
| `notebook/build_notebook.py` | Builder notebook Kaggle aktif (v9). `dryrun_stub.py` untuk dry run lokal tanpa model |
| `notebook/v1.ipynb` … `v9.ipynb` | Notebook hasil run Kaggle T4x2 |
| `notebook/0-initial-iteration/` | Baseline awal: tiga notebook GBDT CPU (pooled OOF KS sekitar 0,48) |
| `results/v1` … `v9/` | Output Kaggle: OOF/test prediksi per model, submission, gambar |
| `ref/notebook/` | Skill dan konvensi penulisan notebook |
| `research/` | Indeks paper metode (PDF diunduh ulang dengan `download_papers.py`) |

## Setup lokal
```bash
uv venv .venv && uv pip install --python .venv/bin/python polars==2.0.0 numpy scipy matplotlib
cd eda && ../.venv/bin/python 02_build_features.py                       # bangun cache data_pq/
uv pip install --python .venv/bin/python -r eda/baseline_duckdb/requirements.txt   # untuk eda/baseline_duckdb dan eda/risk_exploration
```
Model **tidak** dilatih di lokal. Semua training dilakukan di Kaggle (GPU T4 x2). Panduannya ada di `handoff/02_proyek_dan_tools/`.
