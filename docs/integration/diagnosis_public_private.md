# Diagnosis pembalikan val/public — 21 Oktober 2026

**Val naik dari 0,529 ke 0,531 sementara public turun dari 0,570 ke 0,567 belum membuktikan validasinya salah.** Pada analog prediksi kita, selisih lokal sekitar 0,002 cukup sering berbalik ketika dinilai pada 9.000 baris. Penyebab spesifik dua resep teman belum teridentifikasi: OOF, split, seed, definisi KS, dan resep pasangan itu belum tersedia.

User mengonfirmasi Kaggle memakai **30% public / 70% private**, yaitu sekitar **9.000 / 21.000** dari 30.000 test. Rubrik lomba memberi **45% public + 45% private + 10% notebook**. Bobot rubrik dan ukuran sampel menjawab hal berbeda.

## Simulasi yang sudah dijalankan

[Analisis 167](../../eda/167_public_private_reversal.py) memakai prediksi OOF tersimpan, dengan 1.000 replikasi dan RNG 20261021. Setiap replikasi memilih 30.000 dari 70.000 OOF tanpa pengembalian, lalu membaginya menjadi 9.000 public dan 21.000 private yang saling lepas. Semua model dibandingkan pada **baris yang sama**. Komposisi dan pasangan ditentukan sebelum run; tidak ada pelatihan, penyetelan bobot, atau submission.

| Pasangan tetap | ΔKS pooled 70k | SD Δpublic 9k | SD Δprivate 21k | Frekuensi kalah public | Kalah public ≥0,003 |
|---|---:|---:|---:|---:|---:|
| Kumo seed42 − keluarga 6 pohon | +0,002175 | 0,005276 | 0,003256 | 24,5% | 12,8% |
| Referensi v31 − referensi v8 | +0,001864 | 0,003514 | 0,002198 | 22,6% | 6,5% |
| Kumo seed42 − Kumo 3 seed | +0,000952 | 0,002447 | 0,001490 | 56,5% | 13,2% |

Ini **frekuensi pada analog OOF kita**, bukan probabilitas untuk dua resep teman, bukan posterior efektivitas, dan bukan forecast private. Rataan selisih pada subset dapat berbeda dari selisih pooled karena KS mengambil maksimum gap CDF pada masing-masing sampel. Pada Kumo seed42 versus 3 seed, rataan selisih pada subset bahkan berbalik tanda.

SD KS absolut public sekitar 0,0113–0,0116, sedangkan private 0,0066–0,0068. Untuk membandingkan resep, gunakan **SD selisih berpasangan** pada tabel, karena kesalahan model berkorelasi. Dalam analog ini SD selisih public sekitar 1,6 kali private. Simulasi tidak mencakup variasi akibat pelatihan ulang, seed, context, pemilihan kandidat berulang, atau perbedaan ensemble OOF dan test.

## Koreksi bukti historis dari analisis 131

Script 131 memakai 30% dari 70.000 OOF, sehingga sampelnya **21.000 baris**. Itu bukan 30% dari 30.000 test. Sampel yang terlalu besar meremehkan ketidakpastian public. File dan output historis tetap disimpan.

| Kontras public historis terhadap pohon | Δpublic diamati | Z dengan proxy 21k | Z dengan proxy 9k | Frekuensi 9k ≤ gap diamati |
|---|---:|---:|---:|---:|
| Kumo seed42 − pohon | −0,00281 | −1,75 | −1,16 | 13,7% |
| Causilo 3 seed − pohon | −0,01004 | −3,31 | −2,10 | 1,2% |

Z di sini bersifat deskriptif, bukan bukti kausal; keluarga model berkorelasi dan beberapa perbandingan telah diamati. Bukti penalti ICL dari public **lebih lemah daripada narasi lama**, terutama untuk Kumo. Gap Causilo tetap perlu kehati-hatian, tetapi belum terbukti ada penalti intrinsik pada semua ICL. Kumo tetap layak diteliti.

## Faktor yang perlu dibedakan

1. **Definisi metrik dan split.** Kumo seed42 mempunyai pooled KS **0,526655**, tetapi mean fold KS **0,529803**, berbeda **0,003148**. Perbedaan definisi ini sendiri melebihi gap val 0,002. Dua resep harus dibandingkan pada split, seed, baris, definisi metrik, dan prosedur seleksi yang sama.
2. **Variasi sampel public.** Selisih lokal +0,002 dan public −0,003 masuk akal pada analog di atas. Pergeseran distribusi tidak diperlukan untuk menghasilkan pembalikan, tetapi simulasi juga tidak menolaknya.
3. **Pelatihan dan inferensi.** Audit 154 menemukan perbedaan nyata pada input 104 versus 90 fitur, kode kategori versus kategori native, API cached versus direct, generator, context, serta ensemble seed/fold. Val dan test perlu memakai kebijakan yang sepadan; angka KS saja tidak mengidentifikasi faktor penyebab.
4. **Seleksi berulang.** Memilih pemenang dari banyak resep pada validasi yang sama dapat menimbulkan optimisme. Bukti lokal 145/146: gain +0,0036 dari seleksi tidak bersarang berubah menjadi −0,0005 setelah bersarang. Ini kemungkinan yang perlu diaudit, bukan kesimpulan tentang resep teman. Pseudo-labeling yang diizinkan tetap memerlukan teacher dan seleksi yang bersarang.

## Implikasi keputusan

Kedua komponen rubrik perlu mendapat perhatian setara. Public lebih berisik karena sampelnya lebih kecil, meskipun bobotnya sama. Pemilihan kandidat tetap memakai validasi nested, beberapa seed, paired bootstrap, dan audit kesetaraan inferensi. Belum ada bukti bahwa salah satu resep teman pasti lebih baik di private, atau bahwa ensemble keduanya pasti membaik. Tidak ada submission untuk diagnosis ini.

Jika konversi rubrik linear dalam KS, komponen model proporsional dengan 0,45 × KS_public + 0,45 × KS_private. Rumus konversi poin/rank belum diperiksa, sehingga mean KS simulasi bukan skor rubrik resmi. Mean KS dua bagian juga berbeda dari KS pada gabungan 30.000 baris.

Bukti tersedia dalam [JSON](../../eda/out/167_public_private_reversal.json), [replikasi CSV](../../eda/out/167_public_private_reversal.csv), [gambar](../../eda/out/167_public_private_reversal.png), [log](../../eda/out/167_public_private_reversal.log), dan [protokol sebelum run](protokol_diagnosis_public_private.md).

Kernel KS lolos 21 perbandingan dengan SciPy pada skor tied/untied dan baris berulang. Sebelas input OOF dicatat SHA256, dengan 70.000 ID unik, target selaras, dan prediksi finite dalam [0,1]. Pooled KS referensi v31 sama persis dengan evaluator 155. Saat review sebelum pelaporan, rerank tambahan pada keluarga pohon dalam komposisi v31 dibetulkan dan analisis diulang. Artefak yang ditautkan sudah memakai definisi evaluator 155; baseline historis tidak diubah.


## Tambahan audit identitas context21Oktober18:03

Auditor197 membuktikan `results/v31/kumo_only.csv` (public historis0,56056)
identik dengan percentile-rank test ensemble56k v28m:Spearman1,maxrankerror0.
File70k v30/v34 identik satu sama lain, tetapi terhadap submission Kumo-only
Spearman0,9986873/maxrankerror0,0726833. Jadi context70k bukan penjelasan untuk
skor Kumo-only tersebut. Lihat
[catatan sumber dan batas inferensi](../../research/notes/21okt_checkpointing_context_audit.md)
dan `eda/out/197_public_context_fingerprint.json/png`.
