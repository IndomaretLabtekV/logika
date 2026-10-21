# Preflight recomputation untuk full-gradient Kumo

Disiapkan sambil run191 berjalan; **tidak menjalankan GPU kedua**. Ini cadangan
engineering bila fixed-union tidak lolos, bukan modifikasi resep189.

Bukti baru:187 kehabisan memori pada forward pertama, baik episode1024/128
maupun512/64. Tidak ada pembaruan bobot atau hasil kualitas. Source SDM lokal
`RowEmbedding.forward` menyimpan aktivasi dalam jalur grad, berbeda dari jalur
inference yang memakai buffer. Pengurangan episode saja belum cukup.

Hipotesis engineering: non-reentrant activation checkpointing pada cell
embedding, setiap outer column block/row block dan setiap layer ICL akan
mengurangi aktivasi tersimpan. Parameter, optimizer, episode, label, RNG,
AMP, lr, scaler, clipping dan allocator tidak diubah. Bukan menurunkan
context inference atau ukuran model. Rekalkulasi menambah biaya komputasi.

Implementasi harus berupa wrapper runtime pada instance model, tanpa mengedit
site-packages, cache pretrained,187 atau sumber yang dikunci191. Hash wrapper,
source SDM/torch yang relevan dan protokol ditambahkan ke manifest terpisah.

Urutan:

1. Uji CPU model kecil: kesetaraan output, gradien dan update SGD terhadap
   model tanpa wrapper; state_dict utuh; bypass dalam no_grad; wrapper ganda
   ditolak. Fixture sintetis ini hanya pemeriksaan numerik, bukan data latihan.
2. Hanya setelah satu job berat sebelumnya terminal dan tidak ada job berat
   lain, jalankan ulang **dua step engineering187 small512/64** melalui wrapper,
   output baru. Bandingkan episode/kolom/label dengan preflight kecil lama.
3. Lolos hanya jika dua step menghasilkan gradien finite pada row embedding
   dan ICL, benar memperbarui bobot, optimizer tanpa momentum/state,
   memory allocator0,7 tidak terlampaui, tidak ada data heldout untuk training.
   Tidak menyimpan bobot, tidak menghitung KS dan tidak membuka outer.
4. Bila lolos, quality experiment membutuhkan protokol terpisah yang memilih
   jumlah step/lr hanya pada inner. Preflight tidak membenarkan submission.

PyTorch checkpoint lokal mendukung `use_reentrant=False`, keyword arguments,
preservasi RNG serta state autocast. Penentuan block menghindari wrapper
bertingkat di dalam induced transformer block. CPU parity harus benar sebelum
GPU; perilaku numerik GPU tetap perlu diperiksa jika jalur kualitas diteruskan.
