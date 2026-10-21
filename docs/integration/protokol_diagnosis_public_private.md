# Diagnosis public/private — 21 Oktober 2026

User mengonfirmasi pembagian Kaggle **30% public / 70% private** dari30.000test,
yakni9.000/21.000baris. Rubrik lomba memberi masing-masing45%;10%lainnya tidak
ditentukan di analisis ini. Bobot rubrik bukan ukuran sampel. Bentuk konversi
KS/rank ke poin rubrik belum tersedia; simulasi rata-rataKS bukan skor lomba.

Pertanyaan: mengapa resep denganval0,529/public0,570 dapat mengalahkan resep
denganval0,531/public0,567? Belum ada OOF/recipe pasangan teman tersebut. Jangan
mengklaim penyebab spesifik atau probabilitas untuk pasangan itu tanpa prediksi.

## Rencana sebelum run167

- Prediksi **tersimpan**, tanpa melatih model, memilih bobot, atau submission.
- Sumber validation_v2: enamGBDTv22, Kumo42v28m, Kumo rata-rata dua seedlainv38,
  Causilo42v22 dan dua seedlainv37. Alignment70kID/target,finite dan[0,1]wajib.
- Komposisi tetap: trees=rankmean6GBDT; Kumo42; Kumo3=(v28m+2*v38)/3;
  Causilo3=(v22+2*v37)/3; v31=(trees+rankCausilo42+rankKumo42)/3;
  v8 dengan bobot historis blend2. Tidak dipilih berdasarkan hasil simulasi.
- Pasangan tetap: Kumo42−trees, Causilo3−trees, Kumo42−Kumo3, v31−v8.
  Ini analog lokal, **bukan dua resep teman**, bukan seleksi model baru.
-1.000replikasi, RNG20261021. Tiap replikasi memilih30k dari70kOOF tanpa
  pengembalian, membagi9kpublic/21kprivate yang saling lepas. Baris/application
  adalah unit; model dibandingkan pada **baris yang sama**. Assignment acak
  tidak dipaksa stratified agar fluktuasi prevalensi juga tercakup.
- Laporkan absolut/perfold/pooledKS, per-pasanganΔmedian/SD/persentil2,5–97,5,
  frekuensiΔpublic<0 serta≤−0,003,private dan mean(public,private). Frekuensi
  empiris bersyaratOOF, bukan posterior efek pada test atau prediksi private.
- Periksa ulang skor public historis131 pada **9k**, lalu bandingkan dengan
  simulasi lama21k yang keliru sebagai ukuranpublic. Laporkan dua ukuran,
  tidak mengubah atau menghapus bukti lama. Zdeskriptif bukan uji kausal;
  banyak perbandingan dan keluarga tidak independen.
- MeanKS dua bagian tidak sama dengan KS dari gabungan30k: supremumCDF
  dihitung tersendiri. Bila rubrik linear dalam KS, komponen45%+45%
  proporsional pada meanKS; bila berdasarkan rank/normalisasi, perlu formula.

## Batas interpretasi

FixedOOF tidak menangkap variasi training ulang, foldseed, fulltrain/test
ensembling, pemilihan kandidat berulang, atau pergeseran distribusi takteramati.
OOF historis tidak diubah dan analisis tidak membuat split validasi baru.
Selisihval harus dibandingkan pada split/seed/pooledmetric dan provenance sama.
Tidak menyimpulkan publiclebihbaik karena semi-supervised tanpa uji nested.

Keputusan: pembalikan yang sering pada analog berarti noise masuk akal, bukan
terbukti penyebab pasangan teman. Kelainan spesifik baru dapat diuji dengan
OOFber-ID,recipe/seed/split, dan detail pipeline inferensi pasangan tersebut.
