# Pilihan semi-supervised berikutnya (23 Oktober 2026)

Dokumen ini rencana, bukan hasil. Tidak ada model SSL dijalankan untuk dokumen ini.

## Bukti yang membatasi pilihan

- Label tersedia 70.000 baris, unlabeled test 30.000 baris; distribusinya sangat
  mirip (PSI marginal maksimum 0,001599, `results/eda_23okt/deeper_audit/summary.json`).
  SSL biasanya paling membantu saat label sedikit atau ada shift yang bisa
  dimanfaatkan. Di sini, keduanya tidak terlihat. Prior gain SSL rendah.
- Sudah null/gagal: hard pseudo-context Kumo dengan threshold kelas (v49, 12 policy,
  terbaik −0,00074 di inner-search), soft pseudo-label prior-preserving pada GBDT
  (v57/v58 di bawah main pada inner), retrieval (v53), dan same-class mixup (v54).
  Jangan mengulang grid threshold/cap atau soft-label prior-preserving.
- Positive tail teacher sangat tipis (8 baris pada p≥0,95 di audit163). Pseudo-positive
  hampir tidak tersedia tanpa menurunkan threshold ke daerah noise.

## Kandidat yang masih punya alasan baru (urut prioritas)

1. **Self-distillation cross-fitted (tanpa unlabeled).** Teacher OOF di dalam
   outer-train (inner 5-fold) memberi soft target untuk student GBDT pada baris
   berlabel yang sama. Ini bukan pseudo-labeling test; menguji apakah smoothing
   label memperbaiki ranking. Kontrol: student label asli dengan budget sama.
2. **Density-regularized curriculum pseudo-labeling (R-CPL,
   [Kim dkk. 2023](https://arxiv.org/abs/2302.14013))**, untuk GBDT, dengan kuota
   persentil per kelas, bukan threshold tetap. Perbedaan dari v57/v58: skor
   confidence dikalikan likelihood kelas di bawah naive density (fit-only), sehingga
   pseudo-label di daerah densitas rendah ditolak. Satu policy dipraregistrasi
   (α=1, curriculum 10%→20% satu putaran); tidak ada sweep.
3. **Unlabeled hanya untuk preprocessing representasi** (misalnya quantile
   binning dari train+test covariate). Label-free, tetapi transduktif dan gain yang
   diharapkan kecil karena shift sudah ~0. Prioritas terendah; harus dicatat
   sebagai transduktif, tidak boleh dicampur dengan pipeline strict.

## Kontrol wajib

- Teacher per fitting partition; outer teacher dibangun ulang dari outer-train setelah policy terkunci.
- Baseline tanpa SSL dengan fitur, preprocessing strict, seed, dan budget sama.
- Kontrol ukuran: duplikasi baris berlabel asli dengan komposisi kelas yang sama.
- Kontrol label: pseudo rows yang sama dengan label dipermutasi (count tetap).
- Satu challenger; search → confirmation inner. Inner fold0 sudah terekspos, jadi
  hasilnya bukti development, bukan holdout bersih. Outer hanya untuk laporan akhir.
- Test inference: query test tidak boleh menjadi pseudo-context bagi dirinya sendiri
  (cross-block). Untuk evaluasi OOF, test pseudo boleh dipakai karena labelnya tidak ada,
  tetapi outer-validation rows tidak pernah dijadikan pseudo.
- Klaim target hanya dari pooled OOF 70.000 baris lengkap.
