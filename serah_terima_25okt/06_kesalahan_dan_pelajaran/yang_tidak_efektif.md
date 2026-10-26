# Yang tidak efektif — jangan diulang tanpa alasan baru

Daftar lama (11–23 Okt) tetap berlaku: `handoff/08_pelajaran/yang_tidak_efektif.md` dan
`handoff/10_ide_belum_dicoba/*` (VIP-COP fitur v48, VIP baris 217, SSL v49, retrieval v53, mixup 185, fine-tune Kumo,
TabICL/TabPFN/zoo, 7.080 pasangan fitur, 3.150 fitur relatif-kelompok, distilasi, stacking, tuning LGB, seg_causilo,
shock makro eda/101, augmentasi pinjaman historis v8, …).

## Tambahan sesi 24 Okt (semua dengan angka di `04_eksperimen/`)
| Ide | Hasil | Kenapa (dugaan) |
|---|---|---|
| Resep acuan apa adanya (cap500@8k + nested_small union) | K 0,51960 < REF 0,52625 | context 8k memotong sinyal; seleksi KS univariat union mengisi slot dengan varian teman yang redundan |
| Seleksi fitur union 896 lewat KS univariat untuk Kumo | −0,017…−0,023/fold | redundansi `t__*` |
| Median impute untuk Kumo | ±0 / −0,0037 | Kumo menangani NaN sendiri |
| Pool 777 (dedup + affine) untuk cap300 | −0,0052 | duplikat tidak merugikan subset acak |
| Seed tambahan Kumo (2→3 seed) | 0,52685 < 0,52740 | noise seed |
| Estimator lebih banyak (24 sekuensial; e16 OOM) | AUC datar | plateau |
| Context penuh pool lebar (cap128@48k via 8×e1) | −0,0031 | trade kolom/context sudah optimal di ~300@28k |
| Anchored union (104 tetap + 196 acak) | −0,0086 | – |
| Budget main 150 | ≈ main100 | – |
| SSL pseudo-label tersampel sebagai context | −0,0022 | pseudo menambah noise ke context berlabel |
| CoLES embedding (ecom/cicilan) | −0,0026 / −0,0007 | fitur buatan tangan sudah menyerap |
| LGB DART ff 0,05 | −0,0062 (f0) | cocok untuk 1.000+ fitur AmEx |
| Series OOF per baris (ecom/cicilan) | −0,0003 / −0,0013 | sinyal per event sudah teragregasi |
| KNN target mean (top-5 univariat) | −0,0040 (f0) | fitur dasar redundan (varian pp) |
| Offset kelompok × bulan (cross-fit) | bulan +0,0007, lainnya negatif | efek kecil |
| Stacking LR cross-fit 11 member | −0,0009 | – |
| GRU-NN di atas blend terbaik | −0,0001 / −0,0005 | keragaman sudah dari Kumo/Causilo (tapi +0,0019 di atas LGB tunggal) |
| Simulasi ruin arus kas | −0,0004 | arus kas bukan mekanisme utama |
| 42 agregat baru di pita KS | −0,0024 | dilusi |
| Target analog pinjaman lama | +0,0006 | sudah tertangkap rp_* |
| Bobot kelas / label smoothing | +0,0001 / −0,0026 / −0,0003 | ranking hampir tak berubah |
| Seleksi nested fitur teman untuk pohon (top-30 gain) | ≈ +0,0002 (4 fold) | – |
| Kumo-only sebagai submisi (K62, K64) | public 0,56203 / 0,56168 | ICL-only transfer lebih lemah dari blend dengan pohon |

## Yang sudah dicek "sudah ada" (jangan dibangun ulang)
volatilitas income, keterjangkauan saldo vs cicilan, premium bunga lender, delay keputusan, payment-method share,
cancel rate, saving rate, DTI total, existing installments.
