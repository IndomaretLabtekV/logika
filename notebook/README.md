# Notebook dan builder

| Entry point | Resep / penggunaan |
|---|---|
| `final.ipynb` | Resep v59; hasil dari `build_final.py` mode default |
| `final_merge.ipynb` | Merge tiga pipeline, resep `merge_R2DK` |
| `build_final.py` | Builder kedua notebook final; tidak menjalankan sel model |
| `README_teman.md` | Penjelasan notebook sumber teman |
| `dsc0152/` | Notebook deliverable dan run historis DSC0152 |
| `0-initial-iteration/` | Baseline awal dengan environment tersendiri |
| `archive/production_v34_21okt/` | Salinan produksi historis dengan checksum asli |

Notebook `v*.ipynb`, notebook bernomor dan builder per versi dipertahankan untuk reproduksi dan rujukan jurnal. Output eksekusi serta execution count notebook kerja dikosongkan sebelum commit. Snapshot audit di `research/snapshots/` dan `archive/` mempertahankan byte historisnya.

Dari root repository:

```bash
python3 notebook/build_final.py
LOGIKA_NOTEBOOK_RECIPE=merge python3 notebook/build_final.py
```

Builder merge membaca manifest lokal di output_dir dalam [recipe_merge.json](../artifacts/optimization/continuation_25okt/recipe_merge.json). Tanpa manifest itu, builder merge belum dapat mengisi bukti hasil. Notebook merge yang sudah dibangun tersedia di Git.

`replay` membutuhkan OOF/prediksi dan manifest lokal, yang tidak dibawa clone. `fresh` menjalankan fitur dan model, membutuhkan data panitia, bobot pretrained dan environment sesuai. Pengemasan/cleanup tidak mengulang training penuh. Banner `DSCxxxx` masih placeholder; sesuaikan dengan identitas peserta sebelum mengirim notebook lomba.
