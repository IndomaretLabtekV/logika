# Kelanjutan riset 23 Oktober 2026

## Arahan terbaru: EDA dan preprocessing saja

User kemudian meminta tidak melatih model dan memfokuskan pekerjaan pada EDA
serta preprocessing. Sesudah perubahan arahan ini tidak ada training, inferensi
TFM, atau submission. Target KS0,53–0,54 tetap arah riset, bukan hasil terukur.
Bagian preflight di bawah terjadi sebelum perubahan arahan.

## Audit awal

- Environment lokal cocok dengan snapshot utama; GPU 8 GiB, tidak ada proses model saat pemeriksaan awal. Satu job berat sekaligus, CPU empat thread, affinity0–11, nice10, cgroup RAM high16/max20 GiB dan swap1 GiB.
- Train70.000/test30.000, positif9.507, tanpa duplikasi atau irisan application/customer ID. Urutan sample berbeda dari test; ekspor harus join ID.
- Sebelas fit v56 lolos audit hash, checkpoint, membership context/query, target, dan formula fusion. Empat fit seed2026/fold1–4 tidak lengkap. Pooled kandidat seed42=0,5278099569 dan seed7=0,5274366367. Tidak dilanjutkan: di bawah target, reference utama legacy tidak memenuhi kontrak fit-only sekarang, dan penyebab hang lama belum terbukti.
- Reference harga legacy memang mengubah fitur context ketika hanya harga customer query diubah. `v5_features(reference_customers=...)` memperbaiki populasi reference tanpa menghapus histori query sendiri. Default lama hanya dipertahankan untuk reproduksi artefak historis.
- Rekonstruksi current DPD dan maksimum DPD akhir hari cocok dengan seluruh1.361.341 cicilan. Untuk pelunasan, maksimum adalah `max(date(last_payment)-date(due)-1,0)`; untuk saldo outstanding, selisih hari cutoff-due. Nol timestamp pembayaran/due setelah cutoff. Ini bukti untuk cutoff keputusan terpilih, bukan rekonstruksi sembarang tanggal historis.

## Pool bersih yang benar-benar dibangun

`results/v60/strict_tfm/pool_inner_f0/manifest.json` mencatat100.000 baris,48.000 fitting dan52.000 target nonfit dihapus. Main232 ditambah664 fitur numerik customer-local dari prefix teman menjadi **896 prediktor**, bukan880. Fitur V3/V4/V5 teman yang memakai reference lintas customer tidak dipakai. Vocabulary pivot dibekukan dari kamus data; vocabulary kategori model dipasang pada actual context.

Audit raw: previous201.984, repayment1.361.341, account1.084.418, bureau797.563, ecommerce4.359.817; nol pelanggaran timestamp pada pemeriksaan masing-masing. Hash pool: `5fcc0534ae128069c12696c3f0c1c4d56cea784dc4b95d2bd04c3959aca5d47c`.

## Kandidat dan arti batas kolom

[Protokol terkunci](protocol.json): Kumo-Tabular large v1.0.1; tiga view x dua model seed2026/2027, e8 internal, rank-average sama bobot. Outer seed42 terpisah dari model seed. Pool capped500 dan uncapped memakai8192 labelled context secara eksplisit karena batas memori; subset kecil100numerik+4kategori memakai seluruh fitting context. Ini adaptasi resource-constrained, bukan klaim bahwa semua anggota memakai context penuh.

Batas500 adalah `SelectColumns` setelah preprocessing dan Latin column shuffle. `fm_select(n_num=500)` adalah selector supervised berbeda, bukan full-view. Dalam preflight aktual,896 kolom input menjadi500 kolom processed per estimator.

## Verifikasi yang sudah dijalankan

-11 regression checks strict pool lulus, termasuk isolasi covariate query, reference sebelum bulan query, erasure target, ID membership, dan rekonstruksi pembayaran.
- CPU preparation engine menolak vocabulary query-only dan tidak menginisialisasi CUDA. Checkpoint large cocok dengan SHA audit.
- Smoke GPU lokal:128 context/16query, uncapped, e1, dua chunk. Selesai dengan label positif berdasarkan nama`1`; peak allocated0,827 GiB. Bukan skor model.
- Preflight real capped view:8192 context,896 input,500 processed,e8,8query. Selesai126,16detik termasuk builder/selector; peak allocated3,975 GiB/reserved5,336 GiB. Tidak menghitung KS.

Preflight uncapped8192/e8 gagal OOM saat fitting:894 kolom processed, peak
allocated5,224 GiB/reserved5,342 GiB pada allocator0,7. Ini kegagalan engineering,
bukan null kualitas. Tidak diulang. Belum ada OOF atau skor kualitas model baru.

## Hasil EDA dan preprocessing tanpa model

Entry point: `eda/214_preprocessing_eda.py`. Artefak lokal:
`results/eda_23okt/preprocessing/`.

| Pemeriksaan | Hasil |
|---|---:|
| Pool awal |896 prediktor|
| Prediktor sesudah preprocessing |792|
| Duplikat nilai dan missing-mask persis pada fitting |102|
| Kolom konstan pada fitting |2|
| Fitur prefix tambahan dengan max abs Spearman ke main <0,95 |436 dari664|
| Selisih missing-rate fit/test maksimum |0,0055875|
| Kolom dengan selisih missing-rate fit/test >0,02 |0|
| Kolom numerik dengan >50% missing |41|

Preprocessing dipasang pada48.000 inner-fit, bukan seluruh train/test. Constant
dan duplikat dihapus dengan preferensi mempertahankan fitur utama. Fitur dengan
satu nilai observed tetapi missing-mask bervariasi tetap dipertahankan. NaN
tidak otomatis diisi nol atau median; median fitting disimpan sebagai opsi untuk
adaptor dense mendatang. Infinity dinormalisasi ke null, kategori query yang
tidak dikenal menjadi null. Tidak ada pemangkasan berbasis korelasi atau label.

Ekspor`features.parquet`:100.000 baris,792 prediktor+5 metadata,ID unik100.000,
52.000 target nonfit null,nol infinity. SHA256:
`b2ea5e6e20148f7233dcaba6ef3044e0a74a5ee583c15d47face117db2e0185b`.
`preprocessing.json` menyimpan fitting-ID digest, vocabulary, median, dan alasan
penghapusan. Smoke fixture lulus untuk isolasi query, duplikat, konstan,
missingness yang informatif, dan kategori yang belum dikenal.

Shortlist20 fitur prefix dipilih dari KS univariat inner-fit, lalu hanya
dideskripsikan pada8.000 inner-confirmation;14.000 outer tidak dievaluasi.
Contoh yang tidak identik dengan fitur utama: late-share365hari
(fitKS0,309334,confirmation0,295160,rho maksimum0,9021),meanDPD365hari
(0,301608/0,273895,rho0,8999),serta mean maksimumDPD perpinjaman
(0,300422/0,266833,rho0,8931). Ini kandidat informasi komplementer,
**bukan bukti gain model**. Beberapa fitur prefix dengan KS tertinggi justru
duplikat/rho1 terhadap main, sehingga menambah semuanya tanpa audit tidak tepat.

`feature_diagnostics.csv` memuat seluruh pemeriksaan,
`fit_selected_prefix_confirmation.csv` memuat shortlist terkunci, dan
`information_audit.png` memvisualisasikan redundansi versus KS univariat.
Missing-rate stabil tidak membuktikan seluruh distribusi stabil. Korelasi
dihitung pada8.000 sampel fitting; korelasi rendah bukan bukti sinyal tambahan
bersyarat pada model. Tidak ada klaim pooled OOF mencapai0,53–0,54.

Pemakaian berikutnya wajib memfit ulang reference dan resep preprocessing
di masing-masing fitting partition; file ini bukan preprocessing global
yang boleh langsung dipakai pada seluruh outer-fold. Tidak ada model baru
yang dijalankan untuk menguji gain sesuai arahan terbaru.

## Audit lanjutan: distribusi, CV, entitas, preprocessing affine

Scope user berikutnya: EDA/preprocessing lebih dalam, audit CV, rencana SSL, dan
VIP-COP ([arXiv2605.12904](https://arxiv.org/abs/2605.12904)); user mengizinkan
inferensi TFM beku, bukan training bobot.

| Pemeriksaan (`eda/215`,`eda/218`) | Hasil |
|---|---:|
| PSI marginal maksimum fit48k vs test30k (792 fitur) | 0,001599 |
| Fitur PSI>0,1 | 0 |
| KS distribusi observed maksimum fit vs test | 0,016740 |
| Rentang decision train / test | 2025-01-01–2025-12-28 keduanya |
| Customer duplikat train / test / irisan | 0 / 0 / 0 |
| Key lintas-customer (loan, installment, order, bureau, application) | 0 |
| Profil identik non-ID | 0 grup |
| Timestamp sumber ≥ cutoff (semua tabel; account pakai awal bulan berikut) | 0 |
| Keputusan train dengan horizon 12 bulan penuh sebelum batas bulan mana pun | 0 |

Prevalensi bulanan train 12,0–14,7% (Wilson CI saling tumpang tindih, misalnya Mei
0,112–0,129 vs Februari 0,139–0,157); proporsi test per bulan 8,0–8,7%.

**Keputusan CV:** stratified 5-fold tetap paling cocok untuk test kompetisi yang
interleaved: tidak ada entitas berulang untuk GroupKFold, dan forward CV tidak bisa
mensimulasikan deployment karena tidak ada label yang matang penuh sebelum batas
bulan dalam data. Risiko utama bukan skema split, melainkan penggunaan ulang
inner/outer untuk seleksi, blend cross-fit yang tidak nested penuh, dan cache
lintas preprocessing. `validation_v2.md` sudah dikoreksi.

Korelasi kondisional (rank, dalam desil fitur main terdekat, fitting saja) dari
20 fitur prefix komplementer hanya −0,009 sampai 0,044: sebagian besar informasi
marginal tumpang tindih dengan fitur main. Ini membatasi harapan gain dari sekadar
menambah kolom; bukan bukti nol gain kondisional pada model.

Preprocessing opsional `eda/216_affine_preprocessing.py`: 15 kolom yang persis
affine terhadap kolom yang lebih awal (mask missing sama, error relatif ≤1e-10 pada
fitting) dibuang, 792→777. Dicek pada 52.000 baris nonfit sesudah resep dibekukan:
0 kolom gagal rekonstruksi. Smoke fixture lulus (affine, nonlinear/missing
dipertahankan, isolasi query). Output SHA
`6a7a4b66c970efcb2a5e08e81d098f5a72bd505a199b947767fbc77c916b7202`.

Rencana SSL: [protokol_ssl_lanjutan_23okt.md](../../../docs/integration/protokol_ssl_lanjutan_23okt.md).

## VIP-COP baris (frozen Kumo-Tabular large) — null

`eda/217_row_vip_context.py`, manifest SHA
`ebdcf246cf0f7307972e8323b92e36d130a7247ea5d01fb846778cd147177778`. Inferensi beku,
tanpa training bobot. Outer seed42/fold0, inner-fit48k dibagi 16 blok bertingkat
(3.000 baris, 407–408 positif). Fitur tetap: `fm_select` 100 numerik + 4 kategori
dari view affine777, dipilih pada fitting saja. e4, model seed20261023.
Search4k/confirmation4k adalah inner stop yang dibagi dua; outer dan test tidak
diprediksi. Berbeda dari v48 (VIP fitur), v53 (retrieval), dan v49 (pseudo-context).

Search: 2 temperatur × 8 ronde × 2 koalisi (8 dari 16 blok, 24k baris), surrogate
aditif least squares, ditambah 32 draw uniform sebagai kontrol anggaran yang sama.
KS search rata-rata VIP 0,5139 (maks 0,5187) vs uniform 0,5144 (maks 0,5253).
Surrogate tidak menemukan blok bernilai: spread φ ≤0,0028, setara noise antar-koalisi.

| Context (confirmation 4k) | KS | AUC |
|---|---:|---:|
| Random tetap 24k | 0,47879 | 0,82688 |
| Terbaik dari 32 uniform 24k | 0,48686 | 0,82728 |
| VIP terkunci 24k | 0,48587 | 0,82815 |
| Semua baris 48k | **0,48886** | **0,82896** |

VIP − all48k = −0,00299 (bootstrap 95% −0,0138…+0,0083, P>0 = 0,32);
VIP − uniform-best = −0,00099 (P>0 = 0,58). **Kesimpulan:** pada data ini, memilih
subset context berlabel tidak lebih baik dari memakai seluruh context. Selisih
antar subset didominasi noise. Bersama hasil v48 (VIP fitur −0,0061 outer),
VIP-COP tidak layak diteruskan ke 5-fold. Ini satu fold dan inner yang sudah
pernah terekspos, jadi hasilnya bukti development, bukan holdout bersih; tidak ada
klaim KS0,53.

### Insiden freeze

Run pertama berhenti karena **hard freeze lalu reboot** (boot 23:57:37; journal
berhenti 23:49, cache masih ditulis sampai 23:54; tidak ada OOM/error kernel yang
tercatat). kwin mencatat "Failed to allocate a graphics buffer for a swapchain".
Satu pasangan cache berukuran 0 byte ditolak guard, lalu dikarantina ke
`results/eda_23okt/row_vip/quarantine_crash_2026-10-23/`; 45 entri valid lolos
verifikasi hash dan dipakai untuk resume. Telemetri resume (`telemetry.csv`, 181
sampel, fsync): GPU 87 °C, power 139,5 W, VRAM maks 5.045 MiB, zona termal CPU
maks 103 °C. Hipotesis utama: beban termal/daya laptop, bukan VRAM. Ini korelasi,
belum bukti penyebab. Untuk run berat berikutnya: pendinginan eksternal, tutup
aplikasi GPU desktop (browser), jeda antar-fit, dan cache ditulis atomik
(tmp+fsync+rename).
