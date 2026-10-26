# Context-optimization dan SSL lanjutan (24 Oktober 2026, ditulis sebelum skor)

Pembanding: member resep acuan yang sama (view, seed model, e8) pada split yang sama.
Screening di outer seed42/fold0 **inner** (48.000 fit / 8.000 stop, pool `pool_inner_f0`);
outer hanya dinilai setelah resep dikunci. Inner fold0 sudah pernah terekspos (217, v49),
jadi screening adalah bukti development, bukan holdout bersih.

## Yang sudah diuji dan TIDAK diulang

| Eksperimen | Apa yang dipilih | Hasil |
|---|---|---|
| v48 VIP fitur | subset 100 dari 228 fitur | outer −0,0061 |
| 217 VIP baris | 8 dari 16 blok baris (24k dari 48k) | −0,0030 vs semua 48k |
| v53/183 retrieval | context tetangga per kelompok query | +0,0025, < gerbang |
| teman balanced/full56k/e16 | rebalancing kelas, context penuh view sempit | null |
| v49 pseudo-context | baris test confident, threshold per kelas, 12 policy | semua kalah inner |
| 185 same-class mixup | baris sintetis | −0,004 / −0,011 |

Jadi "context per-query/kNN" (=v53) dan "kuota kelas positif" (=balanced) **tidak** diulang:
tidak ada alasan baru.

## Tugas 2 — C2 "context maksimum yang muat" untuk view lebar

Alasan baru: 217 menunjukkan random 24k → semua 48k menaikkan KS confirmation
0,4788 → 0,4889 (+0,010) pada view 104 kolom. View lebar resep (cap500/uncapped)
dipaksa memakai 8.192 baris karena VRAM, sehingga kemungkinan kehilangan sinyal paling
besar ada di situ. Ini **bukan seleksi** (beda dari v48/217) dan bukan rebalancing (beda
dari balanced): hanya memperbesar context acak-stratified ke batas memori.

- Varian: `pool_cap500`, s2026, e8, context = 24.576 (memory fraction 0,85 +
  expandable segments); bila OOM, 16.384. Dicatat sebagai penyimpangan engineering.
- Pembanding paired: `pool_cap500` context 8.192, seed dan e sama, query stop 8k sama.
- Lolos screening bila ΔKS stop > 0 dan AUC tidak turun; lalu 5-fold outer sebagai
  pengganti member cap500 (dan uncapped bila muat), dinilai sebagai kandidat baru.

## Tugas 3 — S3 pseudo-label tersampel (soft-label lewat sampling) sebagai context

Beda dari v49: v49 hanya memasukkan baris test **confident** dengan label keras di atas
threshold per kelas (12 policy), sehingga komposisi kelas dan kalibrasi berubah. Di sini
**semua** 30.000 baris test masuk, label disampel `y ~ Bernoulli(p_teacher)`, dua draw
independen dirata-rata rank (ekspektasinya = soft label; prior kelas teacher terjaga).
Satu policy, tanpa grid threshold.

- Teacher: Kumo `nested_small` (fitur 104 dipilih di fitting) dengan context fitting saja,
  memprediksi test. Untuk outer: teacher dari 56k outer-train. Baris outer-validation tidak
  pernah menjadi pseudo row; target outer tidak masuk teacher/seleksi/stopping.
- Student: `nested_small`, e8, context = fitting + 30k pseudo.
- Kontrol wajib: (a) baseline fitting saja; (b) kontrol ukuran = fitting + 30k test dengan
  label tersampel lalu **dipermutasi** (jumlah positif sama).
- Lolos screening bila student > baseline dan student > kontrol permutasi.

## Keputusan setelah screening

Hanya varian yang lolos screening diteruskan ke 5-fold outer; ranking kandidat dari pooled
OOF 70k; Δ vs baseline resep pada fold yang sama, paired bootstrap stratified 1.000.

## Amendemen 02:10 (sebelum skor screening apa pun)

- C2 max context diubah 24.576 → **16.384**: dari preflight (8.192 baris → peak 3,98 GiB, buffer
  baris 2,1 GB) buffer 24.576×504×256×2 B ≈ 6,3 GB + overhead ~1,9 GB > 7,7 GB GPU.
- **Arm M (median impute)**: TLDR resep acuan user menyebut median impute, tetapi protocol.json
  23 Okt dan 212 mengirim NaN mentah. Arm `M_ctx8192`, `M_ctx16384` (view cap500) dan `M_small`
  (nested_small, context penuh) mengisi NaN numerik dengan median baris fitting berlabel saja.
  Pembanding paired: arm yang sama tanpa impute.
- Konteks: hasil outer yang sudah terlihat (cap500 s2026 pooled 0,51885; nested_small f0–f2
  0,5088/0,5146/0,5001 vs legacy 0,5256/0,5380/0,5197) memotivasi arm ini; screening tetap di inner.
