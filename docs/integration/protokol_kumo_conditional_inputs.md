# Seleksi kondisional input Kumo, 21 Oktober 2026

Disusun setelah173/176 terminal null, sebelum melatih selector kondisional.
Outer keluarga input belum dibuka. Baseline tetap utama_raw dari173, dihitung
pada context48k/querysearch4k+confirmation4k, e4,generator4200,chunk128/allocator0,7.

Hipotesis: KS univariat menyisihkan fitur yang berguna secara bersyarat. Pool
teman1000numerik banyak varian berkorelasi; ranking univariatnya belum mengalahkan
poolutama228numerik. Seleksi menurut gain pohon menguji relevansi gabungan, bukan
menganggap seluruh fitur teman buruk. Riwayat Causilo utama dengan selector gain
(jurnal18Okt,siklus16) null; ini alasan membatasi tiga kandidat dan tidak mengulang
eksperimen Causilo. Kumo/poolteman/gabungan dengan seleksi ini belum diuji setara.

Tiga kandidat, semuanya kategori kode float/raw seperti baseline:

1. utama_gain: poolutama,100numerik hasil selector kondisional+4kategori.
2. teman_gain: poolteman,100numerik hasil selector yang sama+4kategori.
3. union_gain: gabungan poolutama+teman,100numerik+4kategori.

Selector: LightGBM dengan v.LGB,400round tetap, num_threads4, deterministicCPU,
seed=seed*100+fold. Hanya numerik dan target inner-fit dipakai; tanpa stopping di
search/confirmation. Ranking menurut importance_type=gain, stable ties menurut
urutan kolom. Greedy |Spearman|<0,95 pada shortlist200; bila belum100 gandakan
shortlist sampai seluruhpool. Tidak mencari tree hyperparameter atau budget.
Kolom konstan/rankinggain0 tidak otomatis dibuang jika budget belum terpenuhi;
tetap ikuti rank/filter dan gagal eksplisit bila seluruhpool tidak cukup.

Union mempertahankan nama utama; seluruh nama numerik teman diberi prefix
`teman__` untuk mencegah bentrok. Empat kategori utama dipakai sekali; kesamaan
nilai kategori terhadap cache teman dicek pada fitting/query IDs. Sumber fitur
dan reference teman tetap cache166 dengan fit IDs tepat; tidak memakai90kolom
historis sebagai daftar input. Union adalah bundle, bukan isolasi pengaruh satu
formula atau jumlah sumber. Overlap/daftarfitur/gain/missingness disimpan.

Audit/caching baseline173: ID,role,fitlabels,kolom,model/e4/generator/chunk,hash
source/paket/checkpoint dan distribusi query harus cocok. Seluruh prediksi baseline
dipakai utuh, tidak diinterpolasi. Tidak melatih ulang baseline bila cache sah.
Masing-masing kandidat memprediksi query8k dalam urutan yang sama.

Search memilih satu kandidat dari baseline+tiga challenger. HarusΔKS>0,003;
confirmation hanya baseline dan satu challenger yang terpilih, harusΔKS>0,003.
Selain itu fallback. Freeze sebelumouter. Confirmation mengulang split inner173,
bukan bukti independen antareksperimen; outer tetap evaluasi, tanpa memilih ulang.
Kandidat yang lolos memerlukan refit reference/selector/vocab pada56k outertrain,
lalu validasi5fold×seed42/7/2026 dan seluruh gerbang user sebelum submission.
Tidak mengubah173 yang sudah memiliki manifest; implementasi177 terpisah.
