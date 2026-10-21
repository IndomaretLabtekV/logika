# Jurnal 20 Oktober 2026: Kumo context, representasi, dan SSL

## Target dan gerbang tetap
- Public terbaik yang dilaporkan user: v31 0,56405; target public 0,57x, ideal >0,57788. Tidak menyatakan target tercapai tanpa hasil submission.
- Kumo-Tabular large; kandidat promosi wajib dibandingkan dengan baseline Kumo dan v31 di validation_v2, 5 fold × seed42/7/2026, seleksi bersarang, paired bootstrap P>0≥0,9, mayoritas fold, gain lokal jelas >0,003.
- Semua run lokal, satu job berat; taskset CPU0–11, nice10, OMP/BLAS/Polars4, VRAM <8 GB. Kaggle hanya submit; tidak push kernel, tidak git push.
- Instruksi terbaru menggantikan izin probe historis: tidak submit tanpa breakthrough. Tidak memakai slot untuk analisis keluarga yang tidak lolos.
- SSL diizinkan panitia hanya dari data panitia. Teacher, threshold, seleksi pseudo-context, dan stopping berada di dalam outer-train; label outer hanya untuk penilaian.

## Pembacaan dan bukti yang sudah ada
Seluruh handoff wajib dibaca, termasuk aturan kerja, kompetisi, validation_v2, tiga berkas pelajaran, rekomendasi, dan semua ide belum dicoba. Jurnal `jurnal_19okt_kumo_vipcop.md` dibaca; v46 hanya memuat smoke dan manifest pencarian, bukan hasil pencarian lengkap.

Audit yang tersedia: `eda/153_kumo_input_audit.py` dan `eda/out/153_kumo_input_audit_*`.
- Inner-fit 48.000, inner-stop 8.000, outer-valid 14.000; 228 numerik +4 kategori, pemilihan 100 numerik.
- 44 kolom terpilih sepenuhnya NaN pada thin-file. Ini temuan representasi, bukan bukti manfaat membuang kolom.
- Audit mencatat 0 pelanggaran cutoff per timestamp/table/split, 0 duplikat ID/customer/baris prediktor, 0 overlap antarsplit yang diperiksa.
- 42 kolom moneter diaudit satuan, 12 masuk baseline100. Signed-log semua numerik tidak persis mempertahankan rank (22 fitur berubah ties; KS `bz_inq_external` −0,002756). Karena itu audit representasi akan menguji log hanya kolom uang yang diaudit, bukan mengklaim invariansi universal.
- Sumber paket lokal `sdm/models/kumo/tabular/recipe.py`: numerik melalui Cast(float64), DropConstantColumns, Standardize, Clip dan ensemble Identity/Power/Robust/RankGaussian. Skala mentah saja belum terbukti bug. Kategori melewati AlignCategories dan ToNumerical sambil mask kategori dipertahankan.
- `eda/155_kumo_nested_eval.py` terdeteksi berisi 5.654 byte NUL seluruhnya, tidak dapat dijalankan. Tidak ada hasil evaluator yang diklaim. Pemulihan evaluator akan memakai kontrak validation_v2 dan artefak ID yang ada.

## Siklus 1: VIP-COP-inspired feature selection, preregistrasi
- Kode: `eda/152_vipcop_kumo.py`, adaptasi ridge koalisi, bukan reproduksi paper.
- Output baru: `results/v47/vip_raw_f0/` agar tidak mencampur manifest/smoke v46.
- Raw numeric seperti baseline; 100 dari 228 fitur numerik +4 kategori kode numerik.
- Inner-fit48k: context search20k stratified; inner-stop dibagi search4k/confirmation4k terpisah.
- 3 temperatur1/0,5/0,25, 40 ronde ×8 koalisi. Surrogate AUROC; pilihan antara VIP dan baseline di confirmation KS dengan context seluruh inner-fit dan e4. Urutan fitur koalisi tetap.
- Pilihan disimpan sebelum outer dinilai. Outer context seluruh train-fold56k, e4, seed generator4200. Outer fold0 hanya screening, tidak untuk memilih koalisi/temperatur.
- Praregistrasi gerbang: hanya pipeline yang menjanjikan secara paired pada screening akan diteruskan ke 15 outer fits, pemilihan fitur diulang dalam tiap fold. Tidak ada perkiraan public atau submission pada tahap ini.

## Slice independen saat GPU bekerja
- Riset sumber primer VIP-COP: periksa detail augmentasi, relevansi row-selection dan batas extrapolation AUROC→KS.
- Implementasi SSL context Kumo di berkas terpisah; tanpa menjalankan model bersamaan. Teacher Kumo dari train partition saja, threshold/kelompok dipilih di inner, fallback baseline jika confirmation kalah. Kapasitas pseudo-context dibatasi sebelum hasil dilihat.

## Status bukti
Pencarian VIP-COP dimulai; belum ada skor kandidat yang dinyatakan. Target belum tercapai. Tidak ada submission.

## Siklus 2: presisi input, analisis CPU nyata
- Script: `eda/157_kumo_precision_audit.py`; run lokal CPU4 dengan GPU disembunyikan, 41,18 detik.
- SDM `TableTensor.from_pandas` **mengonversi numerik ke default torch.float32 sebelum** resep Cast(float64). Konversi aktual dibuktikan dengan TableTensor CPU dan perbandingan nilainya.
- Dari 228 numerik, **99** memperoleh tambahan ties pada float32; **48** termasuk baseline100.
- Perubahan KS univariat terbesar: `bz_inq_external` **−0,002875**, `pv_rate_delta` **−0,001533** (keduanya tidak terpilih baseline100), `pv_rate_mean` **−0,000604** (terpilih).
- **42 kolom uang: perubahan KS univariat akibat float32 tepat 0**, baik sebelum maupun sesudah signed-log. Gambar `eda/out/157_kumo_precision_audit.png` diperiksa: kehilangan unique besar didominasi fitur diskret dengan noise float sangat kecil, bukan uang.
- Keputusan: jangan mengklaim raw-money precision sebagai penyebab gap public. Uji log tetap hipotesis perubahan geometri distribusi, bukan pemulihan informasi yang terbukti hilang.
- Artefak: `eda/out/157_kumo_precision_audit.{csv,json,png}`. Tidak membuat fitur baru, tidak mengganti cache, tidak menyentuh statistik label held-out.
- Run pertama menampilkan warning NumPy read-only pada pembuktian 64 baris, tanpa operasi tulis ke tensor. Buffer pembanding kini diminta writable eksplisit; hasil model tidak terlibat.

## Koreksi riset sumber primer VIP-COP
Sumber: [paper v2](https://arxiv.org/html/2605.12904v2), [kode augmentasi](https://anonymous.4open.science/api/repo/VIP-COp/file/VIPCOp/augmentation/sample.py), [kode solver](https://anonymous.4open.science/api/repo/VIP-COp/file/VIPCOp/optimizer/solver.py), [Kumo](https://huggingface.co/blog/nvidia/kumo-tabular).
- Paper **tidak menguji pseudo-label test**. Augmentasinya sintetis dari train berlabel; implementasi publik memakai same-class mixup Beta(0,2;0,2). SSL kita adalah hipotesis baru, bukan reproduksi eksperimen augmentasi paper.
- Gain TabICLv2 augmentasi **+1,91% relatif terhadap VIP tanpa augmentasi**, bukan +1,91 poin KS. Tidak ada ceiling KS kompetisi yang bisa diturunkan.
- Kode publik memakai cumulative diagonal ridge dan classification margin, tidak identik dengan SGD/AUROC pada algoritma paper. Adaptasi lokal sklearn ridge juga bukan salinan persis.
- Risiko proxy152: e1 hanya jalur Identity dari `Choice(round_robin)`, sementara e4 memasukkan Identity/Power/Robust/RankGaussian. Transfer search→confirmation perlu dilihat; pencarian lebih lama tidak otomatis menyelesaikan mismatch fidelity.
- Risiko sample-level VIP: ~48k parameter tidak layak diestimasi dari ~1.000 koalisi. Kelompok baris merupakan hipotesis yang lebih terukur, tetapi belum bukti gain.
- Untuk SSL inference test kelak, query tidak boleh hadir sebagai pseudo-context dirinya sendiri: perlu cross-block test inference dengan rule/threshold beku. Belum ada resep notebook yang diganti.

## Evaluator 155 dipulihkan dan diuji pada prediksi nyata
- Kode `eda/155_kumo_nested_eval.py` menggantikan berkas NUL; memakai helper bootstrap `screen2.boot`, ID alignment eksplisit, menolak fold/seed hilang dan tidak menganggap OOF gabungan sebagai bukti 3 seed.
- Smoke nyata: cache baseline lama `eda/out/kumo_finetune_f0_base.parquet`, 14.000 baris, bootstrap1.000, 23,03 detik CPU4. KS **0,528648**; vs Kumo seed42 historis **Δ+0,003092, sd0,002440, P>0=0,848**; vs v31 pada baris sama **Δ+0,004360, sd0,003252, P>0=0,815**.
- Ini **bukan gain baru**: cache memakai generator seed0, sedangkan Kumo historis seed4200. Angka menunjukkan satu draw dapat tampak +0,003 tanpa lolos signifikansi atau replikasi seed. Evaluator benar menandai SCREENING ONLY dan gerbang lokal False.
- Artefak `eda/out/155_existing_baseline_f0.{json,png}`. Tidak ada keputusan fitur/resep dari skor evaluator ini.

## Siklus 3: audit representasi dan SSL, protokol sebelum run
- `eda/154_kumo_recipe_audit.py`: bandingkan raw, native-categorical, gain-ranked100, seluruh232, dan signed-log42 kolom uang yang telah diaudit. Search20k-context/e4/4k-query, konfirmasi baseline+1 challenger pada inner-fit48k/e4/4k-query terpisah.
- Sebelum itu, verifikasi prediksi cached `fit/predict` vs direct pada context/query sama dan invariansi query-chunk. Cached mempertahankan ensemble estimator yang sama antarchunk; direct lama menggambar ulang preprocessing. Perbedaan ini dicatat, bukan diam-diam dianggap model identik.
- Pilih representation/ranking policy di inner, lalu refit ranking/vocabulary hanya pada outer-train. Outer menyimpan kandidat dan baseline cached benar-benar paired. Thread LGB gain dipatok4, bukan n_jobs20 bawaan modul.
- `eda/156_kumo_ssl_context.py`: teacher Kumo large/e4 dari inner-fit saja memprediksi test panitia/search/confirmation; kandidat confidence0,02/0,05 dan koalisi empat confidence-shell, cap2k/10k, seluruh real context tetap ada. Ini seleksi kelompok confidence, **bukan** VIP individual/baris64blok.
- Challenger search harus ΔKS>0,003; konfirmasi terpisah juga >0,003, selain itu noSSL menang. Policy disimpan sebelum outer teacher direfit pada outer-train dan pseudo test dibuat ulang. Baseline dan kandidat memakai fitur/context/seed matched.
- Audit memori SSL menyimpan max allocated/reserved per inference. Smoke berbudget kecil akan dipisahkan dan **bukan** bukti kualitas.
- Belum ada hasil154/156; satu job berat tetap VIP152. Tidak ada submission.

## Pemulihan setelah laptop hang (laporan user)
- Hang diterima sebagai fakta; tidak mencoba mereproduksi hang. Setelah kelanjutan, tidak ada job model aktif dan `v47/vip_raw_f0` hanya berisi manifest. Tidak ada koalisi/selection/outer yang bisa dipulihkan.
- Snapshot yang diperiksa: RAM tersedia28GiB; GPU360MiB, suhu64°C. Log boot sebelumnya yang tersedia tidak memberi bukti OOM atau NVIDIA Xid; penyebab hang **belum diketahui**, bukan diklaim selesai diperbaiki.
- Semua job berat berikutnya dibungkus cgroup scope: MemoryHigh16GiB, MemoryMax20GiB, MemorySwapMax1GiB, CPUQuota400%; tetap taskset0–11/nice10/OMP4. Allocator PyTorch dibatasi90% VRAM; bukan batas untuk proses GPU lain.
- `Predictor.predict_cached` kini menjadi satu implementasi cache yang dipakai152/154/156; helper duplikat154 dihapus dan semua pemakainya dimigrasi. Cache selalu dibersihkan melalui finally. Query chunk500 pada pencarian.
- Pencarian152 menyimpan koalisi, AUROC, koefisien, ronde, dan state RNG tiap ronde secara atomic+fsync. Resume memeriksa config/source/package/data hashes.
- Smoke engine154 (bukan kualitas model): context512, query128, e4, native-categorical/raw/all; peakGPU1,703GiB. Direct vs cached **max error0**; cached satu query vs dipecah dua **max error0,000153**. Selesai23,11detik.
- Smoke checkpoint152 nyata: context512, search128, dua ronde×dua koalisi; scope dihentikan sengaja sesudah checkpoint ronde1, lalu command identik dijalankan ulang. Output **`resume tau=1 after round=1`**, melanjutkan ronde2, konfirmasi inner dan menyimpan pilihan. PeakGPU1,23GiB; peakRSS2,47GiB. Tidak menilai outer, tidak dianggap gain.
- Run penuh baru: `results/v48/vip_raw_cached_f0/`, 3 temperatur ×40ronde×8koalisi, searche1/context20k dan confirmatione4/context48k; outerfold0 hanya setelah pilihan inner tersimpan. Log runtime `eda/out/152_v48_vip_raw_cached_f0.log`. Run ini memakai cached fixed-estimator, tidak mencampur hasil direct yang terputus.
- Tidak ada submission; target public belum tercapai.
