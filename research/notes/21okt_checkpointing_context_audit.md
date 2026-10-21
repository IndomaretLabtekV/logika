# Checkpointing dan identitas context historis

21 Oktober, observasi terpisah dari evaluasi fixed-union191. Tidak mengganti
policy189 dan tidak memakai label test.

Sumber primer:

- [Artikel NVIDIA Kumo Tabular](https://huggingface.co/blog/nvidia/kumo-tabular),
  diterbitkan29September2026, dibaca21Oktober: pretraining bertahap, tahap
  terakhir sampai60kbaris dan100kolom. Artikel memperingatkan generalisasi
  di luar rentang training; belum memberikan resep fine-tuning task ini.
  Ini alasan memeriksa perbedaan context, bukan bukti penyebab skor kompetisi.
- [PyTorch checkpoint2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html):
  varian non-reentrant mendukung keyword arguments dan tidak mensyaratkan
  input checkpoint bertanda requires_grad. Preservasi RNG dipakai eksplisit.
  Source torch lokal ikut dihash wrapper195; parity output/gradient/update,
  dropout RNG serta autocastCPU diperiksa196.
- [Model card NVIDIA](https://huggingface.co/nvidia/Kumo-Tabular): contoh
  inference/classification; tidak menjadi bukti bahwa full-gradient adaptation
  pasti meningkatkan KS atau cukup memakai satu learning rate tertentu.

Auditor197 memeriksa artefak, bukan menebak dari nama resep:

- `results/v31/kumo_only.csv` (public historis0,56056) **persis sama** dengan
  percentile ranks `results/v28m/test_kumo_large_c56000.parquet` di30kID.
  Spearman1,0 dan maksimum beda rank0.
- Prediksi70k di`results/v30/v30/test_kumo_full70k.parquet` dan
  `results/v34/test_kumo_full70k.parquet` byte-hashnya sama. Keduanya berbeda
  dari submission Kumo-only: Spearman0,9986873, maksimum beda percentile
  rank0,0726833.
- Karena itu, hipotesis bahwa public Kumo-only tertinggal **akibat memakai
  context70k melampaui pretraining60k** tidak cocok dengan identitas file
  aktual. Perubahan70k padav34 adalah perbandingan historis lain; satu skor
  tidak memisahkan efek model, subset public dan nonlinieritasKS.

Hasil/gambar di`eda/out/197_public_context_fingerprint.*`; gambar diperiksa.
Tidak ada skor hipotetis test yang dihitung, model dilatih, atau submission
baru. Perbedaan seed/split, input, serta variasi subset public masih relevan;
audit ini tidak menetapkan penyebab pasti pasangan skor tim teman.
