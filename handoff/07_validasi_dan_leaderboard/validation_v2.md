# validation_v2 (dibangun 17 Okt; dipakai untuk semua kandidat sejak v22)

Kode: `eda/valid2.py` (protokol), `eda/run_valid2.py` (driver), `eda/blend2.py` (blend). Cara pakai:
`02_proyek_dan_tools/validation_v2_dan_tool.md`.

## Audit skema sebelum validation_v2
| Sumber bias | Status di v10–v21 | Tindakan di validation_v2 |
|---|---|---|
| Early stopping di fold OOF (v1–v9) | sudah diperbaiki di v10–v21 (inner 1/7) | tetap inner 1/7, lalu **refit di seluruh train fold dengan budget tetap** (GBDT best × 1,15; TabM best epoch) |
| Bobot blend dipilih di OOF yang sama dengan penilaian | masih (greedy/stack di seluruh OOF) | bobot blend **cross-fit per outer fold** (dipilih di 4 fold, dinilai di fold ke-5) |
| Kandidat bersih dibandingkan dengan base v8 yang bias | masih (bench s01 pakai OOF v8) | resep v8 dijalankan ulang di protokol sama (v22) → semua pembanding paired di fold yang sama |
| Banyak kandidat → winner's curse | sebagian (wins/40 = stabilitas fit) | **paired bootstrap baris (P>0 ≥ 0,9) + menang di mayoritas fold** |
| Seleksi fitur FM / nn_prep | sudah per fold | tetap |
| Desain fitur dari EDA di seluruh label train | tidak bisa dihindari | konfirmasi fitur baru wajib lewat model cross-fit, bukan dari skor EDA |

## Protokol
1. Outer: StratifiedKFold(5, shuffle) per seed 42/7/2026 (ICL biasanya 1 seed karena mahal). Outer fold **hanya dinilai**.
2. Inner: split stratified 1/7 dari train fold (seed `1000 × seed + fold`) untuk early stopping/seleksi epoch.
3. Refit di seluruh train fold dengan budget tetap. ICL (Causilo, Kumo, TabPFN): tanpa stopping, context = train fold.
4. Seleksi fitur FM (`fm_select`) dan `nn_prep` dihitung per train fold.
5. Simpan OOF (dengan application_id) + test (rata-rata model fold) per model; manifest versi paket + config.
6. Blend: rank-average bobot non-negatif, bobot dipilih cross-fit; pembanding resep v8 bersih (0,52426).
7. Kriteria promosi kandidat: **Δ vs resep v8 dengan paired bootstrap P>0 ≥ 0,9 dan menang di ≥ 3/5 fold**, lalu
   perkiraan public ± sd ditulis sebelum submit.
8. Jangan memilih skema karena skornya lebih tinggi. Angka utama selalu **pooled OOF KS** (mean fold KS ~+0,003 lebih tinggi).

## Validasi skema itu sendiri (`eda/104`)
- 5-fold vs 10-fold: 0,52074 vs 0,52067 (Δ +0,0004, P 0,59) → 5-fold cukup.
- Maju-waktu vs acak: −0,0024, sd per bulan ~0,02 → tidak ada drift, split acak tidak menyembunyikan apa pun.
- Adversarial AUC 0,4987 → train/test interleaved acak; StratifiedKFold tepat.

## Batas resolusi
- sd selisih KS dua blend berkorelasi tinggi di 70k baris ≈ 0,0015 → untuk P>0 ≥ 0,9 butuh gain ≥ ~0,002.
- Seed outer tambahan tidak mengecilkan noise ini (baris yang sama).
- Satu fold sd ~0,008: screening fold 0 hanya untuk perbandingan paired, bukan angka absolut.

## Apa yang terbukti dengan protokol ini
- v22: resep v8 bersih 0,52426; Causilo > LGB > TabM (kebalikan OOF bias).
- v31 lolos (+0,00221, P 0,93, 4/5) dan gain transfer ke public (+0,0026).
- Tapi CV **tidak** memprediksi kegagalan blend tanpa pohon di public (v31b −4 sd) → tambahan aturan praktis: GBDT ≥ 1/3.

## Revisi 18 Okt sore: varians seed outer untuk model ICL
- Causilo seed 42 (dipakai v22/v31) 0,52583; rata-rata 3 seed outer (42/7/2026) 0,52418 → seed 42 kebetulan
  beruntung ~+0,0016. sd antar seed outer untuk ICL satu seed ≈ 0,0015 di OOF pooled — sebesar ambang keputusan.
- Bootstrap baris (P>0) **tidak** menangkap varians ini (barisnya sama, assignment fold berbeda). Untuk kandidat ICL,
  bandingkan pada beberapa seed outer bila biayanya terjangkau, atau perlakukan selisih < ~0,003 sebagai noise.
- v31 dengan Causilo 3 seed vs v8 tetap +0,00194 (P 0,90, 4/5); Kumo 3 seed 0,52570 vs seed 42 0,52665. v31 dengan kedua ICL 3 seed vs v8: **+0,00170** (P 0,88, 5/5) — margin yang
  jujur untuk v31 (bukan +0,00214).

## Revisi 19 Okt: aturan tambahan
1. **Winner's curse seleksi fitur** (`eda/145` vs `146`): memilih 30 dari 7.080 kandidat dengan residual label semua baris
   lalu menilai di baris yang sama memberi +0,0036 (P 0,95) palsu; bersarang −0,0005. → seleksi apa pun yang memakai label
   dari banyak kandidat wajib dilakukan di dalam train fold per outer fold.
2. **Nilai di level keluarga**: gain per anggota pohon tidak berarti; pembanding = keluarga rank-mean 6 GBDT v22.
3. **Rata-rata seed** untuk selisih < ~0,002 (set seed v22 juga beruntung ~+0,001).
4. **CV tidak memvalidasi keluarga ICL untuk public** (offset ICL lebih rendah 0,005–0,010 dari pohon). Gain CV yang
   datang dari ICL harus dianggap tidak pasti sampai dikonfirmasi public.
5. Tidak ada skema validasi alternatif (maju-waktu, subset alami, label tetangga) yang mereproduksi penalti ICL.
