# Pipeline utama

Mulai dari root repository untuk menyiapkan environment dan dataset sesuai [README](../README.md). Jalankan eksperimen dari `eda/` bila skrip menggunakan import lokal.

| File | Peran |
|---|---|
| `common.py`, `features.py`, `nn_prep.py` | Path data, fitur dan preprocessing utama |
| `valid2.py`, `run_valid2.py` | Validasi outer fold, pemilihan stopping di inner split dan refit |
| `blend2.py` | Ensemble dan evaluasi |
| `166_teammate_nested_features.py` | Rebuild fitur teman per fitting partition |
| `219_build_outer_pools.py`, `224_strict_tfm_variant.py` | Pool fitur strict dan runner varian Kumo |
| `251_teammate_recipe_s42.py`, `254_kumo_dsc_features.py` | Famili teman dan Kumo pada fitur DSC0152 |
| `260_merge_pipeline.py` | Entry point merge: plan, build, blend dan verify |
| `265_fresh_check.py` | Pemeriksaan reproduksi subset pipeline di folder terisolasi |

Skrip bernomor mempertahankan nomor eksperimen dan lokasi aslinya supaya import, protokol, serta catatan riset tetap cocok. Nomor yang lebih tinggi tidak otomatis berarti resep terbaik.

`baseline_duckdb/`, `deep/`, `risk_exploration/`, dan `v5_review/`–`v8_review/` berisi jalur EDA/review khusus. Folder `out/` adalah keluaran runtime; laporan kecil yang sudah tercatat di Git dipertahankan sebagai bukti, sedangkan output baru diabaikan. Cache, model, OOF dan dataset tetap lokal.

Lihat [jurnal merge](../handoff/04_riwayat_versi/jurnal_25okt_merge.md) untuk hasil tiap famili dan [indeks eksperimen](../docs/integration/indeks_eksperimen.md) untuk versi sebelumnya. Membuka atau memeriksa kode tidak memerlukan training ulang.
