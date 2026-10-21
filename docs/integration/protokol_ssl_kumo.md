# Protokol Kumo SSL: confidence terpisah per kelas

Dicatat setelah user 21 Oktober kembali mengizinkan pseudo-labeling dan teknik
semi-supervised lain, sebelum run model SSL. Data hanya dari panitia.

## Bukti dan hipotesis

Audit163 tanpa label dari 30.000 prediksi test historis: threshold simetris0,95
menyisakan12.725 pseudo-negatif dan8 pseudo-positif. Untuk0,98:7.058 negatif,1 positif.
Agreement hard-label LGB/Cat100% pada tail itu **bukan bukti benar**, karena tidak
ada label test. Prediksi historis adalah rata-rata fold; teacher baru dapat berbeda.
Prediksi historis tidak boleh dipakai sebagai teacher untuk validasi nested.

Hipotesis: confidence simetris hampir hanya menambah kelas negatif dan dapat
memperburuk bias class-prior. Threshold positif yang berbeda dapat memasukkan
informasi kelas positif yang lebih berguna, dengan tradeoff pseudo-label noise.
Ini belum membuktikan gain. Semua context berlabel asli tetap dipertahankan.

Motivasi umum threshold per kelas didukung
[FlexMatch](https://arxiv.org/abs/2110.08263) dan
[FreeMatch](https://arxiv.org/abs/2205.07246). Keduanya terutama metode SSL deep
training; implementasi lokal ini **bukan reproduksi** kedua algoritma, bukan
klaim hasil vision berpindah ke Kumo. Ini kebijakan threshold eksplisit sederhana.

## Perubahan satu faktor utama

- Kumo large/e4,104fitur seperti baseline,raw numerik,kategori numerik; cached
  fixed-draw,realcontext penuh. Inference seed=100×outerseed+fold.
- Teacher hanya fit inner-fit48k. Query: test30k,inner-search4k,confirmation4k.
  Teacher outer dibuat ulang hanya dari outer-train56k setelah policy terkunci.
- Baseline noSSL dan threshold simetris0,02/0,05 dipertahankan sebagai kontrol.
- Challenger: negative threshold0,02/0,05 dan positive threshold0,7/0,9;
  cap pseudo-context2k/10k. Cap uniform fixed-seed, bukan class balancing.
- Group coalition dinonaktifkan pada eksperimen ini: audit163 menunjukkan dua
  shell positif hanya1 dan7baris; tidak menghabiskan anggaran untuk memberi
  koefisien surrogate seolah kelompok positif besar. Adaptasi VIP fitur tetap152.
- Satu faktor utama: threshold kelas. Jangan sekaligus mengganti fitur90/native
  categorical/teacherfamily/normalisasi/caps setelah melihat outer-score.
- Source156 asli diarsipkan byte-identik di research/locked_recipes sebelum
  helper `ssl_policies.py` dipisah. Default symmetric selector harus tetap sama.

## Seleksi, evaluasi, dan kontrol

Seluruh label untuk memilih threshold/cap hanya inner-search. Konfirmasi satu
challenger terbaik pada inner-confirmation terpisah; harus ΔKS>0,003 pada search
dan confirmation, selain itu fallback noSSL. Pilihan tersimpan sebelum outer.
Pemilihan ranking fitur dan vocabulary diulang pada fitting partition masing-masing.
Catat jumlah pseudo perkelas,ID,teacherprobability,sourcehash,pakethash,cachehash,
memory allocated/reserved. Tidak ada real row dikurangi demi memori.

Baseline dan SSL memakai fitur legacy yang sama, termasuk normalisasi harga
transduktif yang ditemukan162. Tidak menyebutnya preprocessing strictly inductive.
Train-only price reference tetap kontrol terpisah yang berguna; jangan mencampur
perubahan population/reference dengan atribusi gain pseudo-label.

1. Smoke: real512,query128,test512,cap64,e4,threshold positif0,7/0,9,group0.
   Bukan bukti kualitas; cek pipeline/cache/ID/contextquerydisjoint/memori.
2. Screening:fold0seed42,fit/search/confirmation penuh,outer14k,hanya sesudah
   smoke lolos dan tidak ada job berat lain. Bandingkan matched noSSL dan v31.
3. Menjanjikan→5fold×seed42/7/2026. Semua seleksi diulang nested perfit.
   Pooled/perfold KS,variasi seed,paired bootstrap application/customer setelah
   rata-rata seed; unit bukan seed×row. Kriteria user gain>0,003,P≥0,9,
   mayoritasfold,audit bersih,forecastpublic≥0,570 tetap berlaku.
4. Sebelum membuat prediksi submission,implementasi test inference harus
   mengecualikan query dari pseudo-context dirinya (misalnya cross-block test).
   Script156 saat ini hanya mengevaluasi train outer; belum ekspor submission.

Satu job berat lokal,cgroup MemoryHigh16GiB/Max20GiB/SwapMax1GiB,CPU400%,
taskset0–11,nice10,OMP4,VRAM allocator90%. Tidak ada submission/push kernel/git push.
Manifest pra-run bersumber SHA aktual ada di `protokol_ssl_kumo.json`.
