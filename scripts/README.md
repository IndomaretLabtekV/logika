# Pipeline teman dan DSC0152

| Lokasi | Peran |
|---|---|
| `build_features*.py`, `model_v5.py`, `spline_model.py` | Fitur V3–V5 dan model pipeline teman |
| `kumo_tabular.py`, `research_goal054.py`, `run_research054.py` | Kumo dan eksperimen goal054 |
| `dsc0152/` | Implementasi sumber DSC0152: fitur, eksperimen, Agas dan Dimaz |
| `run_dsc_s42.sh` | Runner DSC0152 v66 pada lima fold seed 42, inner stopping dan refit |
| `download_data.py` | Pengunduh dataset kompetisi menggunakan akses Kaggle milik penerima |

Nomor versi sumber teman, DSC0152 dan pipeline utama memiliki riwayat masing-masing. Lihat [peta integrasi](../docs/integration/perbandingan_repository.md), [dokumentasi DSC0152](../docs/dsc0152/README.md) dan [environment](../environments/README.md).

Runner merge `eda/260_merge_pipeline.py` memanggil `bash ../scripts/run_dsc_s42.sh` dari `eda/`. Runner shell menentukan root dari lokasi file, memakai `.venv/bin/python`, dan menyimpan cache/prediksi di lokasi lokal yang sama seperti sebelumnya. `LOGIKA_PYTHON` dan `LOGIKA_CPU_AFFINITY` dapat diatur untuk mesin lain. Runner ini menjalankan pekerjaan model berat; cleanup hanya memeriksa sintaksnya.

Symlink relatif di `dsc0152/` menghubungkan data dan output ke `data/` serta `results/` pada root. Targetnya perlu disiapkan sendiri pada clone baru. Symlink tidak memasukkan isi dataset atau prediksi ke Git.
