# Aturan kerja dari user

## Pembaruan user terbaru 21 Oktober, setelah audit162

User: "Ada update dari panitia bahwa boleh melakukan pseudo-labeling dan teknik
semi-supervised lainnya. Jadi, coba kembangkan dan pikirkan apapun caranya agar
mendapatkan skor publik dan val KS setinggi mungkin."

Jalur SSL/pseudo-label data panitia diizinkan kembali. Ini mengatasi larangan
pseudo-label pada lampiran awal yang dicatat sesudah bagian ini. Teacher,
threshold, sampling, bobot dan stopping tetap di inner split; outer hanya evaluasi.
Eksperimen transduktif harus menyatakan penggunaan covariate tak berlabel dan
memiliki kontrol sebanding. Tetap tanpa data eksternal, future leakage, LB probing,
push kernel/git push; satu job berat lokal dan gerbang submission tetap berlaku.

## Instruksi lampiran awal 21 Oktober — SSL kemudian diperbarui di atas

Lampiran user7336e7d7: konsolidasi dan riset secara otonom,tanpa menunggu
persetujuan rutin. Semua run lokalRTX3070Ti8GB,satu job berat,taskset0–11,nice10,
OMP4–8. **Dilarang pseudo-label test,fitting distribusi test,LB probing,
data eksternal,leakage masa depan,push kernel,dan git push.** Pretrained publik boleh.
Jangan menjalankan156_SSLtest meskipun izin panitia historis dicatat di bawah.

Submission hanya breakthrough yang lolos gain>0,003 vs pembanding relevan,
5fold×minimal3seed,mayoritasfold,pairedbootstrapP≥0,9,audit leakage/seleksi bersih,
serta alasan kuat perkiraan public≥0,570. Maks3/hari reset07WIB,tulis forecast±sd
sebelum submit;satu per satu. Tanpa breakthrough/ragu: jangan submit.
Pertanyaan user dijawab dahulu;bukan izin runbaru. Bila user minta berhenti,hentikan.
Jangan menyebut kandidat/hasil sebagai kata terlarang yang pernah ditegur user.
Jaga perubahan awal,OOF,hasil unik,resep,bukti skor,lisensi,dataset/environment.

Bagian setelah ini merupakan riwayat; aturan yang bertentangan tidak berlaku.

> ⚠ Sejak 17 Okt 2026 laptop diganti (CachyOS, RTX 3070 Ti 8 GB) dan aturan "jangan run model lokal" **tidak berlaku
> lagi**: screening model lokal di GPU diperbolehkan. Lihat bagian "Tambahan 17–18 Okt" di bawah. Bagian berikut
> dipertahankan sebagai riwayat.

## Batasan keras lama 12–16 Okt (verbatim)
> "JANGAN RUN MODEL ATAUPUN NOTEBOOKNYA DI LOCAL! Cukup lakukan EDA dan Preprocessing yang memang bisa dijalankan di laptop gua yang punya spesifikasi CPU Intel Core Ultra 7 155H dan GPU Intel Arc Graphics (Meteor Lake-P integrated GPU)."

Tafsiran yang dipakai sepanjang chat dan tidak pernah dikoreksi user:
- **Boleh di lokal:**
  - statistik, counting, binning, dan EDA unsupervised;
  - membangun fitur dan menguji leakage;
  - aritmetika pada prediksi yang sudah tersimpan: blending, logistic stacking dengan Newton numpy pada kolom prediksi, bootstrap;
  - dry run notebook dengan semua library model di-stub (`notebook/dryrun_stub.py`).
- **Tidak boleh di lokal:** melatih LightGBM, XGBoost, CatBoost, TabM, foundation model, dan sejenisnya, atau menjalankan notebook asli.
- Catatan dari Astra (review v7): beberapa script lama, misalnya `eda/61` (naive Bayes MIL) dan `eda/72`, sebenarnya model supervised sederhana. Ini area abu-abu, jadi lebih aman dihindari.

## Instruksi berulang (ditempel user di hampir setiap putaran)
- Lakukan DEEP ANALYSIS, karena "kunci lomba data itu EDA".
- **Setiap analisis jadi satu file `.py`** yang mengeluarkan angka **dan** gambar, lalu outputnya dianalisis.
- Preprocessing diturunkan dari temuan EDA lalu **diverifikasi**: simpan sampel dari setiap kelas dan cek apakah hasilnya sesuai harapan ("secara teori benar tapi rusak secara evaluasi"). Ulangi sampai mentok.
- Tangani duplikasi dan leakage supaya model tidak menghafal pola.
- Pakai analisis matematis, misalnya correlation matrix, untuk mengerucutkan penyebab.
- Cari model SOTA (Hugging Face, GitHub, "bahkan model China"), lalu optimasi dan enhance.
- Buat notebook yang siap di-run di **Kaggle T4x2**, pakai skill `ref/notebook/`, dan patuhi `docs/context.md`.
- "Cek selalu val KS nya agar tidak jomplang antara public dan private."
- "Jangan stop sampai nemu breakthrough."

## Alur kerja user
1. Assistant membuat notebook vN dan menjalankan dry run lokal.
2. User menjalankannya di Kaggle (T4x2, Internet on, secret `HF_TOKEN` / `TABPFN_TOKEN`).
3. User menaruh output `/kaggle/working` di `results/vN/` dan notebook hasil run di `notebook/vN.ipynb`.
4. User bertanya "layak jadi first/second/last submission of the day atau tidak". Jawab dengan keputusan tegas plus angkanya.
5. User melapor skor public, lalu minta putaran berikutnya.
6. User kadang memakai asisten lain (**GPT-6 Astra**) untuk audit. Hasilnya ada di `eda/deep/` dan `eda/v5_review/` sampai `eda/v8_review/`. Assistant diminta membaca dan menanggapi audit itu.

## Preferensi komunikasi
- Bahasa Indonesia. User santai dan kasar kalau kesal, itu normal.
- **Jangan berulang kali bilang "final", "final picks", atau "submission final".** User marah di akhir chat: masih 15 hari, jadi bahas pemilihan akhir hanya kalau diminta.
- User tidak suka kalau assistant terkesan "menyerah". Tetap jujur kalau memang tidak ada sinyal, tapi tawarkan eksperimen konkret berikutnya.
- User menghargai keputusan tegas ("submit / jangan") yang disertai angka.
- User pernah memakai `/goal` dengan Stop hook ("jangan stop sampai local val 0.54–0.56"). Hook itu terus menolak berhenti karena target memang tidak tercapai. Assistant menjelaskan dengan jujur dan user bisa `/goal clear`.

## Lingkungan
- Laptop: Intel Core Ultra 7 155H dengan iGPU Intel Arc (tanpa CUDA).
- vast.ai (RTX5090) sempat dipakai untuk v7, lalu **ditinggalkan karena download lambat**. Semua run sekarang di Kaggle T4x2.
- User bilang "ga perlu ada constraint lamanya training". Tapi sesi Kaggle tetap dibatasi 12 jam, jadi notebook memakai guard `DEADLINE`.

## Tambahan 16 Okt
- User memasang `/goal`: "capai target >0.57768, jangan berhenti sebelum itu terjadi, jangan submit jika belum lebih
  bagus dari v8, jangan run model ataupun notebook di local, … akses kaggle saja untuk run model". Artinya:
  - assistant **boleh dan diharapkan** menjalankan model sendiri lewat Kaggle API (push kernel, ambil output);
  - submit hanya kalau validasi lebih baik dari v8 (kecuali user eksplisit minta probe);
  - Stop hook `/goal` menolak berhenti selama target belum tercapai. Ketika user minta berhenti dan izin submit ditolak,
    hook ini membuat loop. User akhirnya `/goal clear`. Jangan menyarankan `/goal clear` atas inisiatif sendiri saat
    hook aktif kecuali memang buntu karena keputusan user.
- "keknya batalin semua dulu aja dah, jujur ini ga nambah nambah dari v8" → user capek dengan eksperimen model yang
  tidak menambah; lebih menghargai eksperimen yang memberi **informasi** (probe yang bisa dianalisis).
- Preferensi submit (16 Okt): "coba submit aja sendiri deh, biar lu bisa mengecek isiannya, terus nama filenya tetep
  submission.csv aja, tapi di descriptionnya baru jelasin ini apa, satu per satu dulu submitnya, selesai 1 terus
  analisis skornya, terus lanjut lagi, takutnya ada yang mau diubah".
- "intinya gua pengen 2 submission ini bisa bermanfaat buat research kita" → setiap submit harus menjawab pertanyaan,
  bukan sekadar coba-coba.
- Handoff harus selengkap mungkin, dibagi ke beberapa folder, tanpa menyalin notebook/results; file tidak relevan
  dihapus supaya repo bisa di-git push oleh user sendiri (**assistant tidak push**).

## Tambahan 17–18 Okt (sesi `/goal` "public > 0,57005")
### Instruksi `/goal` (inti, dipasang user 17 Okt)
- Target public > 0,57005 (stretch > 0,57768) dari kandidat yang unggul atas resep v8 di **validasi bersih**
  (paired bootstrap P>0 ≥ 0,9, menang di mayoritas fold) dan reproducible. "Fitur jenuh: cari informasi baru, bukan tuning."
- Alokasi ~70% riset/EDA, ~30% modelling. Catat bukti dan keputusan tiap siklus di `handoff/`.
- Audit skema validasi; kalau masih bias buat validation_v2 (nested), evaluasi ulang v8 dan semua kandidat di skema
  yang sama; jangan pilih skema karena skornya lebih tinggi. Rerun bersih, simpan OOF/test per model dengan application_id.
- Blend: rank-average bobot positif, dipilih di held-out; hindari stack LR berbobot negatif.
- Lokal: screening 1 fold beberapa epoch di GPU lokal, dibandingkan resep v8 di fold dan protokol yang sama; VRAM < 8 GB,
  satu job sekaligus. Hanya yang mengalahkan v8 di screening lanjut full fold (awalnya: di Kaggle T4×2).
- Kuota GPU Kaggle 30 jam/minggu, catat versi paket di manifest. Kalau CLI "Authentication required", minta user
  `! kaggle auth login --force`.
- EDA: setiap analisis satu file `.py` di `eda/` yang mencetak angka dan menyimpan gambar ke `eda/out/`; fitur baru lewat
  `eda/features.py` lalu diverifikasi (sampel per kelas, uji leakage cutoff `decision_datetime`).
- **Larangan (verbatim, 17 Okt; dua butir pseudo-label/fitting test DICABUT 20 Okt oleh izin panitia, lihat bawah):**
  "data eksternal, leakage masa depan, pseudo-label test, fitting distribusi test, LB probing
  (rule 11). Pretrained publik boleh."
- Submission: CSV 30.000 baris, disimpan di `results/vN/`; nama `submission.csv` hanya untuk kandidat yang dipromosikan.
  Maks 3/hari, reset 07:00 WIB; satu per satu; description berisi isi dan pertanyaan riset; tulis perkiraan ± sd sebelum
  submit dan analisis sesudahnya. Submit sendiri hanya kalau lebih bagus dari v8 di validasi bersih full fold dan gain
  jelas di atas noise. Sebelum 07:00 WIB manfaatkan 1 submission untuk kandidat paling informatif.
- Notebook final harus menghasilkan file yang disubmit, pakai `ref/notebook/SKILL.md`, deadline 30 Okt 2026 23:59 WIB.
- Skill yang diminta: mlcraft:domain-tabular, mlcraft:data-rigor-and-leakage, mlcraft:rigorous-evaluation,
  mlcraft:accuracy-improvement-loop, ml-training-recipes, 0-autoresearch-skill.
- "Bahasa Indonesia, jujur, klaim disertai angka, jangan git push."

### Arahan tambahan selama sesi (kutipan user)
| Kapan | Kutipan | Artinya |
|---|---|---|
| 17 Okt sore | minta uji semua model leaderboard TabArena, jangan terpaku patokan v8, coba skema CV lain, cek repo supaya EDA/preprocessing tidak diulang | zoo screening + `eda/104` |
| 17 Okt malam | "kaggle nya ke blocked kenapa? kan gua bilang kalau full run di kaggle aja" | user keberatan full fold dijalankan lokal karena status Kaggle salah dibaca |
| 17 Okt malam | "kalau sampai jam 6 belum ada 1 pun breakthrough dari v8, gua ingin lu coba lakukan submission apapun yang memang membantu kita untuk analisis" | submission informatif sebelum 07:00 tetap dipakai walau tanpa breakthrough |
| 17 Okt malam | "next time kalau memang lebih cepat di local, mending di local aja, kecuali memang lebih cepat di kaggle" | **aturan baru: full fold di tempat yang lebih cepat** (menggantikan "full fold di Kaggle") |
| 17 Okt malam | "gua mau tidur, intinya lu bebas lakukan apapun as long as goal tercapai. cari sampai mabok gimana caranya biar dapet > 0.57" | kerja otonom semalam |
| 18 Okt pagi | "lanjutkan sampai goal tercapai, masih ada sisa 13 hari. jangan stop sampai dapat 0.57768" | lanjut riset |
| 18 Okt pagi | "kayaknya kalau seluruh run ini beres udah dulu aja" | berhenti setelah run yang sedang jalan selesai |
| 18 Okt siang | "kalau analisis dari final run nya udah beres, udahin dulu aja, gua mau matiin laptop dulu, karena udah hampir 24 jam nyala" | stop, lalu `/goal clear` |
| 18 Okt siang | minta handoff selengkap mungkin dibagi beberapa folder, tanpa notebook/results, hapus file tidak penting, rapikan untuk push (assistant tidak push) | folder ini |

### Keamanan dan kredensial (wajib)
- Token HF dan PriorLabs (TabPFN) diberikan user; disimpan **hanya** di `~/.config/logika/secrets.env` dan
  `~/.cache/huggingface/token`. Jangan pernah menulis token ke file repo, notebook, commit, log, atau dataset.
- Membuat dataset Kaggle berisi token **ditolak** oleh classifier izin (eksfiltrasi data). Jangan dicoba lagi.
  Akibatnya model gated (TabPFN) hanya diuji lokal; kernel Kaggle hasil push CLI tidak punya secret.
- Access token Kaggle dibuat saat runtime ke variabel environment saja (lihat `02_proyek_dan_tools/kaggle_cli_kernel_dan_submit.md`).

### Preferensi yang terlihat di sesi ini
- User menghargai kerja otonom panjang tapi ingin bisa menghentikan dengan satu kalimat; arahan "udah dulu" lebih
  tinggi dari Stop hook `/goal`. Stop hook tetap memaksa lanjut sampai user `/goal clear`.
- Pertanyaan "TabFM bukannya udah pernah ya?" → sebelum 17 Okt TabFM belum pernah dijalankan (status C3 ❌); 17 Okt OOM
  lokal, v27 Kaggle v1 gagal (OOM/NaN), v2 jalan dan null. Cek katalog sebelum mengusulkan model "baru".
- User bertanya mana yang lebih cepat, lokal atau Kaggle → jawaban berangka ada di `02_proyek_dan_tools/kecepatan_lokal_vs_kaggle.md`.

## 18 Okt malam (teguran)
- Kuota GPU Kaggle habis → **semua run di lokal**, termasuk uji notebook. Jangan push kernel kecuali user minta saat itu.
- **Jangan menyebut "final"** (notebook final, resep final) selama lomba berjalan; pakai "notebook submission" / nama builder.

## Sesi `/goal` 18 Okt sore – 19 Okt (kutipan dan aturan baru)
Teks `/goal` 18 Okt 12:43 (inti): lanjutkan lomba, target public > 0,57768, baca handoff dulu, langkah pertama lacak
bug `eda/kumo_finetune.py` lalu screening fine-tune Kumo; "fitur dan model zoo sudah jenuh: cari informasi atau cara
baru, bukan tuning"; riset ~70% / modelling ~30%; validation_v2 wajib; semua run di lokal; Kaggle hanya untuk uji
reproducibility notebook; submit hanya bila lolos validation_v2 dengan gain > 0,002; jangan tulis token; bahasa
Indonesia, jujur, klaim berangka, jangan git push.

| Kapan | Kutipan user | Artinya / aturan |
|---|---|---|
| 18 Okt ~20:30 | "Stop sebut bahwa itu adalah notebook final … Kuota kaggle gua sudah habis, bukankah gua udah bilang … seluruh run dilakukan di local saja" | **tidak ada push Kaggle**; **jangan sebut "final"** selama lomba berjalan |
| 18 Okt ~22:00 (setelah laptop hang) | "hari ini submisison kan sudah kepakai 3/3 … gua ingin lu memakai seluruh submissionnya jika memang benar benar sudah mendapatkan breakthrough … jika dalam rentang waktu besok jam 07:00 sampai 06:00 2 hari lagi lu tidak mendapatkan breakthrough, disitu lu boleh memakai seluruh submisisonnya untuk bahan analisis … seluruh hal yang lu lakukan wajib 100% masih mengikuti aturan yang ada di @docs/context.md" | sebelum **21 Okt 06:00** hanya breakthrough yang boleh disubmit; sesudahnya (tanpa breakthrough) semua slot boleh untuk analisis |
| 19 Okt ~07:40 | "Kenapa lu malah submit sekarang? Bukannya gua udah biiang untuk eksperimen dulu sampai besok jam 6 pagi? Atau memang lu tadi bener bener pede dengan ke 3 submission itu?" | 3 probe 19 Okt 07:00 **melanggar** arahan (salah baca jendela). Ragu soal jendela waktu → tunggu/tanya |
| 19 Okt siang | "Oh ya, lu udah pernah coba temporal & relational feature engineering belum?" | dijawab (sudah banyak; celah: snapshot cutoff mundur) lalu dikerjakan → null |
| 19 Okt malam | "btw lu udah nyoba optimizing context sampling belum? temen gua yang dapet skor publik 0.57065 dia pakai kumo ensemble 3 seed, coba jawab dulu udah atau belum" + paper arXiv 2605.12904 | jawaban: belum (lihat `10_ide_belum_dicoba/vipcop_context_optimization.md`) |
| 19 Okt malam | "gua ga bilang lu suruh jalanin loh? gua cuma nanya udah atau belum, batalin seluruh proses yang ada" | **pertanyaan = jawab saja**, jangan menjalankan apa pun tanpa diminta; semua proses dihentikan |
| 19 Okt malam | minta folder handoff selengkap mungkin, beberapa folder, tanpa notebook/results, plus rapikan repo siap commit GitHub | folder ini (tidak di-push) |

### Preferensi yang terlihat di sesi ini
- User marah bila arahan eksplisit (lokasi run, jendela submit, "jangan jalankan") dilanggar walau dengan niat baik.
  Arahan user > dorongan Stop hook `/goal`.
- User memberi petunjuk dari luar (teman tim lain, paper) dan ingin jawaban jujur "sudah/belum" lebih dulu.

## 20 Okt: izin panitia untuk semi-supervised
User meneruskan jawaban panitia: "Maaf ya baru menjawab untuk chatnya, Untuk penggunaan teknik semi-supervised learning diperbolehkan yaa dan hanya menggunakan data yang disediakan oleh panitia saja."
- Pseudo-labeling, self-training, dan teknik semi-supervised lain **boleh**, dengan **data panitia saja** (train + test
  yang disediakan). Larangan user 17 Okt "pseudo-label test, fitting distribusi test" dicabut.
- Tetap dilarang: data eksternal, leakage masa depan (cutoff `decision_datetime`), LB probing / manipulasi leaderboard.
- Syarat validasi: seluruh pipeline SSL dijalankan di dalam tiap outer fold (pseudo-label dari model train fold); label
  outer fold tidak boleh dipakai untuk membuat, memilih, atau memberi threshold pseudo-label. Pipeline SSL harus ada di
  notebook submission.
