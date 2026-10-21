# Environment penelitian

Root proyek memakai `.venv` lokal yang sudah tersedia. Snapshot versi runtime
yang dipakai saat integrasi ada pada `utama/requirements-runtime.txt` dan
`utama/runtime.json`. Jangan mengganti environment untuk menjalankan resep lama.

Sumber teman membawa tiga keluarga dependency yang berbeda:

- `teman/requirements.txt`, `requirements-optimization.txt`: Python3.10,
  numpy1.24.1,pandas1.5.3,sklearn1.2.1.
- `teman/requirements-v5.txt`: menambahkan torch1.13.1,TabM0.0.3,EBM0.6.10.
- `teman/requirements-kumo.txt`: Python3.11,torch2.8.0,SDM0.2.1,
  numpy2.2.6,pandas2.3.3,sklearn1.7.2.
- `teman/requirements-goal054.txt`: mewarisi Kumo dan menambahkan TabPFN9.2,
  EBM0.7.8 serta dependency lain sesuai metadata asal.

File `-r requirements...` tetap relatif terhadap folder file requirement.
Gunakan environment terpisah, misalnya `.venv-kumo-teman`, bila ingin mereproduksi
runtime asal. Tidak menginstal paket atau mengganti runtime aktif saat integrasi.
Contoh dari root:

```bash
python3.11 -m venv .venv-kumo-teman
.venv-kumo-teman/bin/python -m pip install -r environments/teman/requirements-kumo.txt
```

Kompatibilitas syntax/import pada runtime sekarang diuji terpisah dari klaim
reproduksi numerik lama. Semua run model berikut tetap tunduk batas lokal GPU8GB,
satu job berat,taskset0–11,nice10,OMP4–8. SSLtest kini diizinkan oleh pembaruan user
21 Okt; data panitia saja dan pipeline nested.
