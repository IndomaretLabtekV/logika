# validation_v2 (dibangun 17 Okt; dipakai untuk semua kandidat sejak v22)

## Audit lanjutan 23 Oktober: estimand dan batas bukti

Tetap gunakan lima outer fold stratified sebagai estimasi utama kompetisi:
train/test sama-sama mencakup Januari–Desember2025 dan tidak beririsan customer.
Audit lokal `eda/215_distribution_cv_audit.py` pada pool strict menemukan
PSI marginal maksimum0,001599 dan KS distribusi observed maksimum0,016740
(48k fitting versus30k test). Ini mendukung kesamaan distribusi terukur,
bukan bukti exchangeability, tidak adanya hidden groups, atau kestabilan label.
GroupKFold berdasarkan customer unik tidak menambah proteksi substantif.

Temporal CV adalah stress test retrospektif terpisah, bukan pengganti primary CV
untuk test yang interleaved. Target horizon12bulan memerlukan tanggal maturity
label agar forward validation dapat disebut simulasi deployment; tanggal itu
tidak tersedia dalam schema aplikasi. Jangan menyebut split waktu lama sebagai
simulasi deployment point-in-time yang sudah lengkap.

Bootstrap tidak menghapus bias dari penggunaan ulang hasil outer/confirmation
untuk memilih eksperimen. Inner8k/fold0 sudah dilihat dalam EDA dan eksperimen;
pembagian ulang menjadi search/confirmation bukan holdout baru yang pristine.
Seed baru juga bukan data berlabel baru. Catat exposure dan bekukan resep
sebelum menilai outer, lalu laporkan seluruh70k OOF, bukan fold terbaik.

Bobot blend cross-fit pada OOF biasa belum membuktikan isolasi nested end-to-end:
base model pembentuk meta-training OOF dapat memakai label meta-validation.
Untuk klaim ketat, gunakan bobot tetap yang dipraregistrasikan atau bangun
inner-OOF seluruh base model di dalam masing-masing outer-train.
Pipeline lama juga memakai vocabulary kategori seluruh train; pipeline strict
baru memfit vocabulary/reference pada fitting partition. Jangan campur cache.

Gerbang submission historis di bawah bukan instruksi aktif: user menggantinya
dengan pooled OOF≥0,53000 plus evaluasi lengkap/audit bersih. Scope terakhir
mengizinkan EDA/preprocessing dan inferensi TFM beku; tidak melatih bobot.


**Koreksi21Okt:** user mengonfirmasi public9k/private21k (30%/70%test),rubrik45/45.
Analisis131 lama memakai21k sebagai proxypublic;167 menghitung ulang ukuran9k:
ZKumo−pohon−1,16,Causilo−pohon−2,10,bukan−1,7/−3,3. Bukti penaltiICL lebihlemah
dari narasilama dan bukan pembuktian penyebab. Lihat
[diagnosis167](../../docs/integration/diagnosis_public_private.md). Angka historis
di bawah tetap dicatat sebagai riwayat; jangan pakaiZlama untuk melarang semuaICL.

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
- 5-fold vs10-fold pooled:0,52074 vs0,52067 (selisih observed−0,00007). Mean bootstrap delta+0,00044,P>0=0,59 bukan selisih observed; tidak ada bukti keuntungan10-fold.
- Maju-waktu vsacak:mean delta−0,0024,sd perbulan sekitar0,02. Tidak terlihat degradasi rata-rata besar pada pemeriksaan ini; bukan bukti tidak ada drift.
- Adversarial AUC0,4987 konsisten dengan distribusi train/test serupa pada model pembeda yang diuji. Bersama overlap kalender/customer unik, ini mendukung StratifiedKFold untuk kompetisi, bukan membuktikan mekanisme split panitia.

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
