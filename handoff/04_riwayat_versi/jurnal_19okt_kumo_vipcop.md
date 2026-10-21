# Jurnal Kumo context/input, mulai 19 Oktober 2026 malam

## Batasan aktif
- Target tetap public ≥ 0,570, idealnya > 0,57788. Belum tercapai di sesi ini.
- Kumo-Tabular large, 3 seed; semua komputasi lokal RTX 3070 Ti 8 GB, satu job GPU sekaligus.
- `taskset -c 0-11 nice -n 10`, OMP/BLAS/Polars 4 thread; tidak push kernel, tidak submit tanpa breakthrough.
- Tidak memakai izin probe historis setelah 21 Oktober: instruksi terbaru hanya kandidat breakthrough.
- Handoff wajib telah dibaca. Null lama bukan kandidat baru kecuali ada alasan mekanistis yang dicatat.

## Siklus 1: audit VIP-COp sebelum menjalankan pencarian

Sumber: https://arxiv.org/html/2605.12904v2 dan `eda/152_vipcop_kumo.py` versi awal.

Temuan kode:
1. `fm_select(X,y,tr,100)` pada baseline pencarian memasukkan label inner-stop dalam seleksi fitur. Outer tetap tidak bocor, tetapi baseline inner tidak independen. Diperbaiki menjadi `fi`.
2. Search dan pemilihan rantai sebelumnya memakai query inner-stop yang sama. Sekarang inner-stop dibagi stratified menjadi search 4.000 dan confirmation 4.000; confirmation tidak dipakai surrogate.
3. Urutan fitur acak koalisi mencampurkan efek feature selection dan efek urutan. Sekarang koalisi memakai urutan kolom sumber tetap; pembanding legacy dan canonical dikonfirmasi terpisah.
4. Search e1/context20k belum menjamin transfer e4/context48k–56k. Konfirmasi memakai seluruh inner-fit/e4; outer memakai seluruh outer-train/e4, pilihan disimpan sebelum outer dinilai.
5. Ini **adaptasi terinspirasi VIP-COp**, bukan reproduksi algoritma paper: ridge seluruh koalisi mengganti SGD; z-score softmax, tiga temperatur tetap, eksekusi serial. Tidak mengklaim angka paper berlaku untuk Kumo/KS kompetisi.

Pra-registrasi awal:
- Raw numeric input seperti baseline, 228 kandidat numerik, 100 dipilih + 4 kategori kode numerik.
- 40 ronde × 8 koalisi × temperatur 1/0,5/0,25; AUROC untuk surrogate (sesuai paper), KS confirmation untuk pilihan.
- Baseline inner-fit `fm_select` selalu boleh menang; tidak memaksakan fitur VIP bila kalah confirmation.
- Screening seed42 fold0 terlebih dahulu. Outer hanya pelaporan; tidak dipakai memilih rantai, stopping, ataupun fitur.
- Kandidat yang layak harus diuji 5 fold × seed42/7/2026, seleksi diulang bersarang; dibandingkan dengan Kumo dan v31 pada baris yang sama, paired bootstrap P>0≥0,9, mayoritas fold, gain >0,003.
- Belum ada perkiraan public maupun keputusan submission.

Artefak rancangan: `eda/152_vipcop_kumo.py`; hasil smoke disimpan terpisah dari hasil inferensial di `results/v46/smoke_raw/`. Parameter smoke bukan bukti kualitas model.
