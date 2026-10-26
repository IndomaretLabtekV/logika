> **Mulai dari `serah_terima_25okt/00_mulai_di_sini/README.md`** — serah terima lengkap sesi 24 Okt (posisi, eksperimen,
> kesalahan, rencana merge). Dokumen ini tetap sebagai status historis.

# Status sekarang — 24 Oktober 2026 (malam)

**Public terbaik baru: 0,56470** — goal054 kumo_offset teman (`results/v70/goal054_kumo_offset/`, ref 57600249),
mengalahkan v31 0,56405. K64 2-seed public 0,56168. Notebook masih me-replay v59 (perlu diputuskan user).


Detail: [jurnal 24 Okt](../04_riwayat_versi/jurnal_24okt_resep_lokal.md). Semua run lokal, satu job GPU,
`eda/thermal_guard.py` aktif (pause GPU>85 °C / CPU>95 °C 15 dtk).

- Resep acuan strict (protocol 23 Okt) 5-fold s2026: cap500@8k 0,51885, nested_small(union) 0,50936,
  K 0,51960 — di bawah REF v59-analog 0,52625. Penyebab: seleksi 104 kolom dari pool union (fitur teman)
  dan context 8k pada view lebar; reference harga strict terbukti identik dengan legacy.
- Context lebih besar membantu: cap500@16k 0,52426 (+0,0054, 5/5 fold). Kandidat v62: K62 0,52662,
  KT62 0,52654, KTC62 0,52634 (semua <0,53). **K62 disubmit: public 0,56203** (Kumo-only legacy 0,56056).
- Null hari ini: median impute (±0), SSL pseudo-label tersampel (−0,0022 vs tanpa SSL, inner).
- cap300@28k (context dari menukar kolom) 0,52549 s2026 / 0,52387 s2027; **K64 2-seed 0,52740** = OOF
  terbaik (tidak disubmit: Spearman 0,9993 vs K62). Null: dedup 777, main150, e16/cap128@48k OOM.
  Context penuh kini muat lewat 8×e1 berurutan (231), tapi AUC inner semua varian lebar 0,8407–0,8416:
  plateau. Anchored union (232) −0,0086. 24 Okt: 0 submit (gerbang user: breakthrough/OOF ≥0,53 belum).
  Siang: teknik write-up juara (AmEx/Home Credit) dan CoLES SSL diuji 5-fold → semua null (detail jurnal).
  Hang laptop ke-3 saat job CPU 8 thread tanpa guard → job CPU kini 4 thread + guard 1-sampel >95 °C.
  Best public tetap v31 0,56405; batas top-8 public ~0,5713 (24 Okt 15:00).

## Status 23 Oktober 2026

Arahan terbaru user: lanjutkan **EDA dan preprocessing saja**, tidak perlu
training model. Jangan menjalankan inferensi TFM atau submission sebagai
kelanjutan otomatis. Semua pekerjaan tetap lokal; tidak ada git push.

- Audit v56 memverifikasi11 checkpoint; tidak di-resume. Dua seed lengkap masih
  pooledKS0,527810/0,527437 dan memakai reference utama legacy.
- Reference utama sekarang mendukung fitting-customer-only; pool baru896 fitur
  berisi main232+prefix customer-local664, bukan referenceV3/V4/V5 teman.
- EDA214 selesai:48.000 fit,8.000 inner-confirmation,14.000 outer tidak dinilai.
  Preprocessing menjadi792 fitur setelah102 duplikat+2 konstanta dihapus.
- Ekspor100.000 baris,ID unik,targetnonfit52.000null,nolInfinity. Refitting
  preprocessing wajib di setiap partition; jangan gunakan satu cache global
  sebagai bukti validasi nested.
- Tidak ada skor model baru atau submission. Target0,53–0,54 belum dibuktikan.
- [Laporan23Okt](../../artifacts/optimization/continuation_23okt/laporan_riset.md);
  kode`eda/214_preprocessing_eda.py`;output`results/eda_23okt/preprocessing/`.

## Status historis 21 Oktober 2026

**Riset dipause atas permintaan user.** Pekerjaan terakhir sesudah pause hanya
merapikan repository untuk GitHub. Jangan otomatis menjalankan model, melanjutkan
checkpoint, melakukan submission atau push.

- Dua sumber terintegrasi pada root yang sama;508file dipetakan, checksum sesuai,
  lisensi utuh. `Logika-main/` sudah dihapus; dataset tetap satu di `data/`.
- Public terbaik: **v31,0,56405**. Submission terakhir **v59,0,56296**, ID57486639,
 21Okt21.48WIB, memakai resep v41 yang sudah tersedia sebelumnya. Bukan breakthrough.
  OOF analogue0,526220 vs0,526126,P>0=0,365,2/5fold. Private belum tersedia.
- Notebook builder sekarang menghasilkan resep v59;5sel kode replay dijalankan
  lokal dan CSV byte-identik. Fresh training GPU tidak diulang. Resep v34 lama
  dipertahankan pada `notebook/archive/production_v34_21okt/`.
- Kumo v56 berhenti pada11/15fit. Pacing parity bit-exact dan inhibitor sementara
  belum mengatasi hang. Seluruh checkpoint tetap lokal; penyebab sistem belum terbukti.
- CPU v57/v58 juga berhenti setelah screening: fullunion0,514859/SSL0,506236;
  cappedunion0,517464/SSL0,512431 vs main0,524007. Tidak diteruskan keouter.
- Pseudo-labeling diizinkan kembali; outer target tetap tidak boleh masuk teacher,
  seleksi, threshold atau stopping. Seluruh run lokal, tanpa data eksternal/LBprobes.
- File besar `results/`, prediksi, dataset, log dan cache dikeluarkan dari Git,
  **bukan dihapus dari disk**. Kode historis dan ringkasan disalin byte-identik
  ke `research/snapshots/` serta `artifacts/history_utama/` sebelum untracking.
- Bukti submission: `artifacts/history_utama/results/v59/v41_lowvar_today/`.
  Cek GitHub ringan: `python3 tools/check_repository.py`; tidak melatih model.

## Riwayat detail konsolidasi dan run21Oktober

Catatan di bawah merupakan snapshot saat eksperimen berlangsung; kata
"berjalan" di bagian historis tidak berarti proses masih hidup sekarang.

- Konsolidasi508 file snapshot teman:69 script→`scripts/`,13 notebook→`notebook/`,
  metadata→`artifacts/`,10 checkpointCSV→`results/teman/`,requirements historis
  terpisah,NOTICE/LICENSE utuh. Tidak ada dataset mentah/OOF/model teman dalam snapshot.
- Inventaris2391file utama+508teman,SHA256:68kelompok duplikat,0lintas sumber.
  Bukti asal/git awal di `docs/integration/`; perubahan awal user dipertahankan.
- Audit struktural159:508checksum,632sel kode notebook,10CSV30k/ID/range,
  15baseline hash lolos. Build-only160:9entrypoint,637sel; tidak melatih model.
- VIPv48 selesai screening fold0: kandidat tau0,5 KS **0,522702** versus baseline
  matched **0,528798**, Δ **−0,006096**, bootstrap SD0,004195/P>0=0,147.
  Inner confirmation+0,003055 tidak bertahan. Tidak diteruskan5fold×3seed;
  tidak ada breakthrough/submission. Source152 dan data historis tetap utuh.
- SmokeSSL156 awal lolos10fit dan replay. Run penuh lama terputus saat laptop
  hang lalu user force-shutdown; bukan hasil null/OOM yang sudah terbukti.
  Amendemen **f142e08** menurunkan chunk500→128,allocator90%→70%,tanpa
  mengurangi context/e4. Smoke parity maxerror0,000283. Preflight170 dengan
  realcontext48k/e4/query128 selesai:allocated3,79GiB,reserved5,38GiB;
  ini pemeriksaan teknis, bukan kualitas/diagnosis penyebab hang.
- Screening SSL konfigurasi baru **selesai, exit0**. Semua 12 policy kalah pada
  inner-search: baseline **0,546077**, pseudo terbaik **0,545334**. Kebijakan
  dibekukan menjadi noSSL; outer baseline/candidate sama **0,524362**, Δ0.
  Tidak diteruskan ke 15 fit. Out`results/v49/ssl_asymmetric_safe_f0`,
  log`eda/out/156_v49_ssl_asymmetric_safe_f0.log`; seluruh real context/e4 tetap.
  Watcher selesai otomatis dengan632 sampel durabel. Penyebab hang pada run
  sebelumnya belum teridentifikasi; hasil run baru tidak membuktikan penyebabnya.
- Diagnosis167:setelah user mengonfirmasi9kpublic/21kprivate,bobotrubrik45/45,
  1.000splitpaired menunjukkan analogΔlokal+0,002175 berbalikpublic24,5%,kalah
  ≥0,003sebanyak12,8%. Bukan probabilitas pasangan teman. Proxy131lama21k
  terlalu besar;ZKumo−pohon kini−1,16,Causilo−2,10. Lihat
  [diagnosis](../../docs/integration/diagnosis_public_private.md).
- Pembaruan user 21 Okt setelah audit162: **pseudo-labeling dan teknik
  semi-supervised kembali diizinkan**, hanya data panitia. Teacher/threshold/
  sampling nested; tanpa LB probing, push kernel/git push. Gerbang submit tetap.
- Prioritas: uji kategori native/input dan pool fitur teman secara nested.
  Protokolinput commit621e359, amendemen chunk128/allocator0,7 **b356ee9**.
  Build fitur inner166 selesai:100k aplikasi,1.000numerik+6kategori,
  48kfit/52kquery;137,64detik,peakRSS6,31GiB. Out
  `results/v50/teammate_features/inner_s42_f0`. Audit175 independen:7median
  kategori dan24contoh mean/median/std/count cocok,0event e-commerce setelah
  cutoff,9counter cutoff sumber0. Tidak membuktikan semua formula sekaligus.
  Shortlist lama hanya memberi71 numerik teman; amendemen sebelum skor
  **ca261e5** memperluas shortlist sampai100. Prepare173 lolos104kolom kedua
  keluarga;kolom utama persis baselineSSL. Adapter172 lulus8checks,1745checks.
  Smoke dan full173 terminal exit0. Inner-search:utama_raw0,546077,
  utama_native0,547050 (+0,000974),teman_raw0,527689,teman_native0,527244.
  Ketiga blend tetap176 juga kalah (terbaik0,543078). Fallback baseline;
  tidak lanjut15fit,outer keluarga input belum dibuka. Daftar90historis metadata.
- User menargetkan kenaikan/submission hari ini, kuota tersisa **2**. Belum ada
  submission dari sesi ini; gerbang breakthrough tetap berlaku.
- Seleksi177: selector gain LightGBM400round fit-only untuk poolutama,
  teman dan union,100numerik+4kategori/e4/context48k,baselinecache173 diverifikasi.
  Protokol sebelumrun **a170272**;4checks178 lulus.177 terminal null:
  utama_gain0,538016,teman_gain0,537118,union_gain0,537014. Blend179 terbaik
  union_rank0,548700 (Δ+0,002623),belum>0,003. Grid bobot180 juga gagal;
  tidak lanjut15fit/outer. Semua artefak tetap di`results/v51/`.
- Geometri181: signed-log12uang terpilih dari daftar42yang diaudit,
  raw/native104kolom,baselinecache173. Protokol **3c8ed1b**,3checks182 lulus.
  Preflight232fitur/context48k/e4 padaallocator0,7 terminal **OOM**;bukan null
  kualitas,all_raw dilewati. Dua money policy selesai:rawsearch0,549461
  (+0,003385),native0,549384. Raw menang search tetapi confirmation0,490192
  vsbaseline0,492904 (−0,002713). Fallbackbaseline;tidak lanjutouter/15fit.
  SourceSDK fit sudah estimator_batch_size1;tidak ada perbaikan batching sederhana.
- Retrieval183:8kelompok query dengan nearestcontext12k,geometrifit-only,
  baselinefull48k+controluniform12k. Protokolb0dacf1/editorial228a5ee;3checks184
  lulus. PrepareCPU selesai,coverage8kquerytepatsekali,12kcontextsubsetfit tiap
  kelompok. Full selesai:retrieval0,541786,probblend0,548534
  (Δ+0,002458),rankblend0,547349 vsfull0,546077;uniformcontrol0,537206.
  Tidak ada>0,003,fallbackbaseline,outerbelumdibuka. Outresults/v53/group_retrieval_f0.
- Augmentasi185: supervised same-class mixup dari real inner-fit,cap2k/8k,
  kontrol8kcloneanchor,seluruh48kreal tetap. Protokol **e2f2333**;4checks186lulus.
  Tidak memakai test/pseudo-label. Sameclass/kategori/history serta missingmask
  anchor dijaga,kolom biner tidakdiinterpolasi. Ini adaptasi augmentasi, bukan
  reproduksi persisVIP-COp atau target baru yang diamati. Full selesai null:
  mix2k0,541874,mix8k0,535558,probblend0,540061;clonecontrol0,546010
  vsbaseline0,546077. Tidaklanjutouter. Outresults/v54/sameclass_mixup_f0.
- EngineeringSGD187:full213,67Mparameter trainable,duaepisode1024/128 dan512/64
  sama-samaOOM di forwardstep0 padaallocator0,7. TidakadaSGDupdate atauKSrun.
  Source pertama diarsipbyteidentik;duaoutputresults/v55 tetap. Janganmenganggap
  preflightini fullfine-tuning yang sudahterbuktinull kualitas.
- Langkahaktif189 **fixedpolicy baru**,preregister **c641a7c**:rankmean50:50
  utama_raw+union_gain,baselinejugarankpercohort/rawdisimpanterpisah. Bukan
  mengubahfallbacklama. Ambang>0,003 user tetapuntukbuktilokalbesar;filterdua
  inner4k>0,003sebelumnya aturanagen. Evaluasiouterf0 dilakukanagar tidak
  menolaksemuagainsubthresholdsebelumbuktibesar. Outerpolicytidakdituning.
  Referencecache166outer56k/44ktargethapus selesai;selector189104kolomtiapmodel,
  union65numerikteman+35utama.3checks190lulus. FullGPU session**82442**,
  PID**15426**,scope`logika-v56-fixed-union-full-21okt.scope`,out
  `results/v56/fixed_union_rank`,log`eda/out/189_v56_fixed_union_f0.log`.
  Fullf0 selesai:blend **0,529800** vsbaseline **0,524362**,Δ **+0,005438**,
  matchedbootstrapSD0,002981/P>0=**0,906**. Vs v31historicalΔ+0,005512/P0,826.
  Masihscreening1fold1seed;belumforecast/submission. Plot/sourcehash diverifikasi.
- **Aktif sekarang:5fold×3seed fixed189**,orchestrator191,session**28308**,
  wrapperPID**16158**,scope`logika-v56-validation15-21okt.scope`,log
  `eda/out/191_v56_validation15.log`. Seeds42/7/2026,fold0–4;166→189 berurutan,
  source/recipe dibekukan,cachef0 direplaytanpainferensiganda. Status/PIDchild/log
  di`results/v56/validation_progress.json`,plan di`validation_plan.json`,
  telemetryperchild di`validation_logs/`. ~17:29WIB sedangmodel_s42_f1.
  PeriksaPID/session/scopeaktual;**jangan mulai jobberat lain**. Cgroup16/20/1GiB,
  CPU400%,NI10,affinity0–11,OMP4,allocator0,7.191otomatis155aggregate setelah15fit,
  **tidak** melakukan submission. Semua gate user danforecast≥0,570 tetap.
- Audit162: 30,0570%–30,1146% event reference harga legacy berasal dari customer
  test. Reference train-only menjadi kontrol penting untuk eksperimen transduktif;
  kebijakan preprocessing harus dinyatakan dan dipertahankan sebanding.
  Audit ini tidak membuktikan target/future leakage atau besaran optimisme KS.

Detail [jurnal21Okt](../04_riwayat_versi/jurnal_21okt_integrasi.md),
[audit](../../docs/integration/perbandingan_repository.md),
[indeks lintas sumber](../../docs/integration/indeks_eksperimen.md).

---


### Checkpoint17:57WIB: kelanjutan aktif

191 tetap berjalan pada scope yang sama, seed7fold0 sedang inferensi.
Seed42 lengkap: pooledcandidate0,527810 vsbaseline0,526741, Δ+0,001069;
3/5fold menang, belum>0,003 dan baru1seed. Auditor192 sudah memverifikasi lima fit lengkap,
termasuk target nonfit benar NaN dan fusion identik. Exporter193 ditulis dengan
blokir GPU sebelum gate155+audit192 lengkap; lima checks194 lulus. Inference
export belum dijalankan. Rencana ada di
[eksekusi sore](../../docs/integration/rencana_eksekusi_21okt_sore.md).
Wrapper activation checkpointing195 disiapkan sebagai cadangan OOM187; lima
checks196 CPU lolos, belum GPU/KS. Jangan menjalankan selama191 masih aktif.

## Status historis per19Oktober2026 malam

Ringkasan satu halaman. Log mentah: `04_riwayat_versi/jurnal_17_18okt.md` (17–18 Okt) dan
`04_riwayat_versi/jurnal_18okt_sore.md` (18 Okt sore – 19 Okt, 35 siklus). Kronologi semua chat:
`04_riwayat_versi/kronologi_chat.md`.

## Posisi
| Item | Nilai |
|---|---|
| Public terbaik | **0,56405** (v31: rank-avg 6 GBDT 1/3 + Causilo 1/3 + Kumo-Tabular large context penuh 1/3) |
| Tim di Kaggle | **"lagi cape lomba asli"** (athillazaidan, nicholaswisee, riantamaputra); dulu "lagi mager lomba bentar" |
| Peringkat (19 Okt ~20:00) | **#33 dari 118**, 22 submission. Top-1 **Datadataan 0,57788**, #2 cupuu 0,57768, #3 0,57458; **9 tim > 0,570**; 33 tim ≥ 0,564 (klaster padat) |
| Kurang ke target | 0,0138 ke top-1; ~0,006 ke batas top-8 (~0,5707) |
| Klaster public resep kita | v31 0,56405 · v32 0,56376 · trees_only 0,56337 · v35 0,56298 · v34 0,56271 (sd ~0,0006) |
| Validasi | validation_v2 (`eda/valid2.py`). Margin jujur v31 vs resep v8 bersih = **+0,0017 s.d. +0,0019** (bukan +0,0022; seed 42 ICL dan set seed pohon v22 sedikit beruntung) |
| Notebook submission | `notebook/build_final.py` → `notebook/final.ipynb`. Lulus run Kaggle 18 Okt (= v34) **dan** run lokal penuh 19 Okt (2,8 jam, Spearman 1,000000 vs v34). Banner tim masih placeholder |
| Kuota | Submit 3/hari reset 07:00 WIB; window 19 Okt **habis** (3 probe). **Kuota GPU Kaggle habis** → semua run lokal |
| Arahan submit aktif | Sampai **21 Okt 06:00 WIB**: submit hanya bila breakthrough. Setelah itu (tanpa breakthrough) slot boleh untuk analisis (rencana sudah ditetapkan di `09_langkah_berikutnya/rekomendasi.md`) |
| Deadline notebook | **30 Okt 2026 23:59 WIB** (telat −1 poin/menit) |
| Semi-supervised | **Diizinkan panitia (20 Okt)**, hanya data panitia; pipeline SSL di dalam tiap outer fold |

## Perjalanan singkat (detail `04_riwayat_versi/`)
1. 12–14 Okt (v1–v8): fitur perilaku v2 (+0,011 public), **price premium `pp_*` v5 (+0,016)** — dua lompatan terbesar.
2. 15–16 Okt (v9–v21, probe S3–S5): stacking/TabICL/tuning null; ditemukan **bias OOF** (early stopping di fold OOF).
3. 17–18 Okt (validation_v2, v22–v36): protokol bersih; zoo TabArena null kecuali **Kumo-Tabular**; v31 0,56405 (terbaik);
   blend tanpa pohon jatuh di public; notebook submission dirakit dan lulus di Kaggle.
4. 18 Okt sore – 19 Okt (sesi ini, 35 siklus): bug fine-tune Kumo ketemu; fine-tune, stacking, distilasi, feature
   bagging, ~20 keluarga fitur baru, seleksi hyperparameter bersarang — **semua null di level keluarga model**.
   3 probe keluarga di public membuktikan **CV ICL tidak transfer** (Causilo-only 0,55333) sedangkan **pohon transfer**
   (trees-only 0,56337). Notebook submission lulus run lokal.

## Temuan paling penting sesi ini
| Temuan | Bukti | File |
|---|---|---|
| Bug `kumo_finetune.py`: urutan kolom kelas `out.numerical` Kumo berubah antar panggilan | Spearman chunk ±0,998; setelah perbaikan baseline 0,52865 | `eda/112`, jurnal siklus 1 |
| **CV keluarga ICL tidak transfer ke public**; pohon dan LR transfer | offset public−CV: pohon +0,0389, LR +0,0372, Kumo +0,0339, **Causilo +0,0292 (z −3,3)** | `eda/131`, probe v39a–c |
| Prediksi test ICL sehat (bukan bug) | Spearman vs pohon OOF/test 0,984/0,987; per chunk konsisten | `eda/130` |
| Penyebab ICL tertinggal tidak terlihat di data | subset alami (`eda/132`), tetangga (`eda/110`, `133`), maju-waktu (`eda/140`), noise label (`eda/148`) semua menolak | jurnal siklus 23, 30 |
| Varians seed outer ±0,0015 di OOF pooled — sebesar ambang keputusan | Causilo 5 seed tunggal 0,5215–0,5258; set seed pohon v22 juga beruntung | `eda/122`, siklus 10–13, 31 |
| **Winner's curse** seleksi fitur dari banyak kandidat | 30 pasangan dari 7.080: +0,0036 (P 0,95) → bersarang −0,0005 | `eda/145`, `146` |
| KS pooled hanya peka di sekitar satu titik potong (rank ~0,69) | thin-view: segmen thin +0,0026 tapi pooled +0,0002; ekstrem terpisah sempurna | `eda/126`, `127`, `148`, `149` |
| Pohon jenuh | setiap gain per anggota (+0,0013…+0,0036) hilang di keluarga 6 GBDT | siklus 17, 20, 25, 26, 34, 35 |
| Data bersih | 0 baris setelah cutoff train/test; previous_applications tidak memuat aplikasi terpilih; tidak ada lantai noise label | `eda/139`, `148` |

## Kenapa stuck
- Fitur: semua kolom mentah 8 tabel sudah dipakai; scan residual vs pohon 1/228; ±20 keluarga konstruksi baru null.
- Model: pohon di plafon (hyperparameter datar, keragaman tambahan tidak menolong); ICL menambah di CV tapi tidak di public.
- Perkiraan plafon pendekatan ini di public **~0,564–0,565**. Gap ke top-1 0,0138 tidak terjelaskan oleh apa pun yang
  kita miliki.
- **Petunjuk eksternal baru (19 Okt malam, dari user):** tim teman di 0,57065 (#8, kemungkinan "BCC mau ke kopdes depok")
  memakai **ensemble Kumo 3 seed**, sementara Kumo-only kita di public 0,56056. Selisih +0,010 → setup Kumo kita
  (fitur `fm_select`, context, preprocessing) kemungkinan besar kurang optimal. Belum diuji (lihat `10_ide_belum_dicoba/`).

## Langkah berikutnya (detail `09_langkah_berikutnya/rekomendasi.md`)
1. **Diskusikan dulu dengan user** sebelum menjalankan apa pun (pertanyaan = jawab saja).
2. Kandidat terkuat yang belum diuji: perbaiki setup Kumo (VIP-COP feature/context optimisation, fitur & preprocessing
   untuk Kumo, ensemble 3 seed seperti tim teman) — `10_ide_belum_dicoba/`.
3. Banner tim (nomor peserta DSCxxxx + nama anggota) — butuh data user — lalu rename notebook
   `ParticipantNumber_TeamName_Notebook`, run ulang lokal, kumpulkan sebelum 30 Okt 23:59 WIB.
4. Setelah 21 Okt 06:00 (bila tanpa breakthrough): slot analisis sesuai rencana yang sudah ditetapkan.

## File penting
| File | Isi |
|---|---|
| `results/v31/submission.csv` | public terbaik 0,56405 |
| `results/v34/submission.csv` | resep notebook submission (public 0,56271) |
| `results/final/submission.csv` | output run lokal notebook submission 19 Okt (= v34, Spearman 1,000000) |
| `results/v22/v22/{gbdt,nn,fm}/` | OOF/test bersih resep v8 (dengan application_id, 3 seed) |
| `results/v28m/` | Kumo context penuh 5 fold (OOF/test, seed 42) |
| `results/v37`, `v38`, `screen/seeds/` | Causilo / Kumo / LGB multi-seed |
| `results/v42/` | kandidat probe yang sudah dipra-registrasi |
| `eda/valid2.py`, `eda/run_valid2.py`, `eda/blend2.py`, `eda/screen2.py`, `eda/120`, `eda/124` | protokol, driver, blend, evaluator |

## Pembaruan 25 Okt 08:53 WIB — merge tiga pipeline (jurnal: `handoff/04_riwayat_versi/jurnal_25okt_merge.md`)
- Pipeline gabungan bersih: **`eda/260_merge_pipeline.py`** + `artifacts/optimization/continuation_25okt/recipe_merge.json`
  (satu skema fold seed 42). Resep dipromosikan **merge_R2DK** = R2 (resep teman goal054, 2 seed) + D (DSC0152 v66,
  inner-stop) + K64 (Kumo strict utama): OOF **0,52886** / AUC 0,84861 (REF 0,52625). Belum 0,53 → tidak disubmit.
- Notebook: `LOGIKA_NOTEBOOK_RECIPE=merge python notebook/build_final.py` → `final_merge.ipynb`, replay byte-identik.
  Default `final.ipynb` tetap v59.
- Perkiraan public R2DK 0,5663 ± 0,0020 (ρ 0,9954 vs goal054) — perlu izin eksplisit user bila mau disubmit.
