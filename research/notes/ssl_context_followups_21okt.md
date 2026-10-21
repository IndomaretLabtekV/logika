# Arah lanjutan SSL context — 21 Oktober 2026

Catatan literatur, bukan hasil LOGIKA. Prioritas tetap VIP yang sedang berjalan,
lalu smoke dan screening 156 sesuai protokol yang dikunci. Catatan ini tidak
mengubah policy 156 atau mengizinkan job berat kedua.

## Regularized pseudo-labeling

Kim, Kim, Song (2023), [Revisiting Self-Training with Regularized Pseudo-Labeling
for Tabular Data](https://arxiv.org/html/2302.14013v3). Paper menggabungkan
curriculum pseudo-labeling dengan confidence yang dikoreksi memakai likelihood
agar pemilihan pseudo-label memperhatikan kepadatan data. Motivasi relevan:
confidence tinggi sendiri tidak menjamin titik berada di daerah yang didukung
data. Eksperimennya mencakup beberapa model tabular, bukan Kumo dengan context
48k. Hasilnya tidak membuktikan peningkatan KS pada LOGIKA. Jangan menyebut
filter confidence 156 sebagai reproduksi metode paper tersebut. Paper juga
mencatat keterbatasan distribusi empiris dalam menangkap karakteristik fitur.

Hipotesis lokal yang belum diuji: bila pseudo-context pada 156 tidak membantu,
periksa dukungan data di fitting partition sebelum membuat filter density baru.
Jarak mentah ratusan fitur dengan skala dan pola missing berbeda tidak otomatis
bermakna. Transform, ambang, dan representasi harus dipasang dan dipilih di inner.

## Representasi untuk context selection

Ipas dkk. (2026), [LUCoS: Latent Unsupervised Context Selection for Tabular
Foundation Models](https://arxiv.org/abs/2605.27254). Paper memilih medoid dalam
embedding PFN tanpa supervisi dan menguji context pada anggaran label kecil.
Pesan yang relevan adalah pentingnya ruang representasi untuk mengukur coverage;
jarak fitur mentah dapat buruk. Setting ini berbeda dari LOGIKA yang sudah punya
puluhan ribu label. Paper tidak membuktikan pengurangan context Kumo penuh akan
membaik, atau bahwa embedding Kumo tersupervisi setara dengan PFN tanpa supervisi.

Keputusan: simpan sebagai arah cadangan. Jangan mengulang null sampling balanced
dari repo teman atas nama metode baru, atau membuang real context dari 156.
Implementasi density/medoid memerlukan hipotesis, kontrol, dan protokol baru.
