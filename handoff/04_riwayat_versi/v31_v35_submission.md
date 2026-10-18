# Submission 17–18 Okt: v31, v31b, v31c, v32, v34, v35

Semua kandidat dipilih dari OOF bersih validation_v2, ditulis dengan `eda/write_candidate.py`, perkiraan public dihitung
sebelum submit (`eda/107_public_noise.py`, subset acak 30% OOF cross-fit = asumsi ukuran public 9k), satu per satu,
dengan description berisi isi + pertanyaan riset + perkiraan. Teks description persis ada di
`kaggle competitions submissions data-science-competition-logika-ui-2026`.

## Window 17 Okt 07:00 → 18 Okt 07:00 (dipakai 04:22–04:30 WIB)
### v31 — `results/v31/submission.csv` (dipromosikan)
- Isi: rank-average 6 GBDT (1/18 masing-masing, total 1/3) + Causilo 1/3 + Kumo large context penuh 1/3. Spearman vs v8 0,99615.
- CV: +0,00221 vs resep v8 bersih (P>0 0,93, 4/5 fold) → kandidat pertama yang memenuhi kriteria goal.
- Pra-registrasi: subset 30% +0,0024 ± 0,0022 (P<0 0,14); perkiraan 0,56145 + 0,0019 ± 0,0025 = **0,5633 ± 0,0025**.
  Pertanyaan: apakah gain ICL bersih transfer ke public, atau pola "ICL/NN kalah di public" (S3/S5/v16) berulang?
  Hasil ≤ 0,5590 = pola itu nyata.
- **Hasil 0,56405** (+0,0026 vs v8, +0,3 sd dari perkiraan; jurnal mentah menulis +0,7, salah hitung) → **public terbaik**. Gain ICL bersih transfer. Pola lama
  kemungkinan karena tabm_aug / stack bobot negatif / OOF bias, bukan ICL.

### v31b — `results/v31/kumo_causilo50.csv`
- Isi: Causilo 1/2 + Kumo 1/2, tanpa pohon. CV +0,00252 (P 0,91, 4/5). Vs v31: OOF +0,0009, row-bootstrap +0,0003 ± 0,0012.
- Perkiraan 0,5642 ± 0,0017. Pertanyaan: apakah 1/3 pohon membantu di public?
- **Hasil 0,55709** = −0,0070 vs v31 (−4 sd). Konsisten dengan S5 (NN/FM saja −0,0028) dan v16 (TabICL −0,0037).
- Diagnostik: prediksi test ICL normal (rank corr test ≥ OOF, mean/kuantil sama) → bukan bug sisi test; `eda/110` neighbor
  gap null. Kurva CV porsi pohon datar 0–0,4 (0,5270 → 0,5259), turun setelah 0,5 → v31 (1/3) tetap pilihan CV-konsisten.
  Penjelasan tersisa: noise ekor atau hubungan label berbeda di subset public. Jangan menyetel bobot dari public.

### v31c — `results/v31/kumo_only.csv`
- Isi: Kumo context penuh saja (model tunggal terbaik). CV vs v31 −0,0001 ± 0,0016.
- Perkiraan 0,5640 ± 0,0021 kalau CV transfer; ≈ 0,557 kalau "ICL sendirian gagal" berlaku umum.
- **Hasil 0,56056** (−0,0035 vs v31, −1,6 sd). Urutan: v31 0,56405 > Kumo 0,56056 > Causilo+Kumo 0,55709.
  Menambah Causilo ke Kumo −0,0035 di public (CV +0,0004) → bukti lemah-sedang (~2 sd) bahwa Causilo tidak transfer
  sendirian; pohon tetap penting.

## Window 18 Okt 07:00 → 19 Okt 07:00 (dipakai 07:01–07:22 WIB; habis)
### v32 — `results/v32/trees_kumo_1to2.csv`
- Isi: 6 GBDT 1/3 + Kumo 2/3, tanpa Causilo. CV +0,00212 (P 0,91, 4/5), vs v31 −0,0001 ± 0,0010.
- Perkiraan 0,5640 ± 0,0018. Pertanyaan: apakah Causilo menyumbang di dalam blend berpohon?
- **Hasil 0,56376** (−0,0003 vs v31, sesuai perkiraan). Causilo netral di dalam blend berpohon.

### v34 — `results/v34/submission.csv` (resep notebook final)
- Isi: 6 GBDT 1/3 + Causilo 1/3 + **Kumo context 70k fit sekali** 1/3. Spearman vs v31 0,99985.
- Perkiraan v31 ± 0,0005 = 0,5640 ± 0,0006. Pertanyaan: apakah resep yang bisa dijalankan ulang di Kaggle mempertahankan v31?
- **Hasil 0,56271** (−0,0013 vs v31; perkiraan meleset). → kalibrasi noise public: perturbasi rank sekecil apa pun menggeser
  KS public ~0,001 karena KS adalah statistik maksimum di satu titik potong.

### v35 — `results/v35/five_family.csv`
- Isi: 6 GBDT 0,3 + Kumo 0,3 + Causilo 0,2 + TabM 0,1 + lr_q 0,1. CV +0,00209 (P 0,95, **5/5 fold**), vs v31 −0,0001 ± 0,0009.
- Perkiraan 0,5638 ± 0,0015. Pertanyaan: apakah keragaman tambahan (NN + linear) dihargai public?
- **Hasil 0,56298.**

## Kesimpulan gabungan
1. Blend pohon + ICL dengan pohon ≥ 1/3 stabil di **0,5627–0,5641** (v31/v32/v34/v35: mean 0,5634, sd 0,0006).
   Semua varian setara-CV juga setara di public → ini plafon pendekatan sekarang.
2. Blend **tanpa pohon** jatuh jauh (v31b 0,5571, v31c 0,5606, S5 0,5587) walau CV setara → **GBDT wajib ≥ ~1/3**.
3. Selisih public < ~0,002 antar varian tidak bermakna. Jangan memilih final dari selisih kecil; pilih yang reproducible
   atau setara-CV.
