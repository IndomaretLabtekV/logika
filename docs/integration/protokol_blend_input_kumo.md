# Blend tetap antarpool input Kumo, 21 Oktober 2026

Disusun setelah smoke173 teknis (64 query per role), ketika full173 masih
berjalan dan sebelum membaca skor full empat policy. Tidak mengubah173 atau
memakai outer. Ini eksperimen lanjutan terhadap keluaran empat policy yang sama,
bukan pengulangan fit atau perubahan bobot dari leaderboard.

Hipotesis: pool utama dan teman mungkin menangkap informasi berbeda, sehingga
ensemble prediksi membantu meskipun salah satu model tunggal lebih lemah.
Tidak ada jaminan bahwa selisih nama fitur berarti informasi independen.

Kandidat ditentukan sekarang:

- Keempat single173, dengan utama_raw sebagai matched baseline.
- Mean probabilitas utama_raw dan teman_native, masing-masing0,5.
- Mean probabilitas keempat single, masing-masing0,25.
- Mean percentile rank utama_raw dan teman_native, masing-masing0,5;
  rank average-ties dihitung dalam query cohort masing-masing role, tanpa label.

Tidak ada pencarian alpha, offset, threshold, subset pelanggan atau clipping.
Search4k memilih satu kandidat dari tujuh; harus ΔKS>0,003. Confirmation4k hanya
baseline dan kandidat terpilih; harus ΔKS>0,003, selain itu fallback baseline.
Simpan keputusan dan hash sebelum materialisasi outer. Jika layak, implementasi
outer hanya melatih konstituen yang dibutuhkan pada reference/vocab/seleksi56k
yang direfit; gate user tetap5fold×3seed, pooled/mayoritasfold/bootstrapP≥0,9.

Confirmation adalah split inner yang sama dengan173;173 mungkin sudah menilai
baseline dan satu single. Jangan menyebutnya holdout konfirmasi baru antarproyek
atau menganggap hasil173 dan176 bukti independen. Seluruhnya masih optimasi
inner; outer belum dibuka untuk keluarga input ini. Dilarang memilih ulang blend
dari outer atau skor public. Nilai public/private diperlakukan setara pada rubrik,
dengan ketidakpastian public yang lebih besar karena hanya9k baris.

Prediksi inner dari cache wajib lolos hash manifest/fit/prediction, ID/role,
reference partition dan tanpa outer IDs. Jika tidak ada kandidat menjanjikan,
jangan menjalankan15fit atau melakukan submission hanya untuk mencoba slot.
