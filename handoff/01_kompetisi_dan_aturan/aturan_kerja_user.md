# Aturan kerja dari user

## Batasan keras (verbatim)
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
