# Evaluasi outer blend tetap Kumo dan union-gain

21Okt setelah187 dua episode OOM dan sebelum membuka outer keluarga input.
Ini prosedur baru dengan **policy tetap**, bukan mengubah keputusan fallback
173/177/179/180 atau menyebut hasil inner-nya telah lolos gate lama.

Alasan: syarat user ΔKS>0,003 berlaku untuk bukti lokal besar, mayoritas fold,
minimal3seed dan bootstrap. Filter tambahan >0,003 pada search4k **dan**
confirmation4k dibuat oleh eksperimen sebelumnya; filter itu dapat menolak
gain yang kecil dan stabil sebelum mendapat evaluasi yang cukup besar.
Kandidat tetap terbaik yang sudah dilihat pada inner adalah mean rank utama_raw
dan union_gain,masing-masing0,5:Δsearch+0,002623. Bukan bukti generalisasi, tetapi
alasan eksplisit untuk satu evaluasi outer dengan policy yang tidak dituning lagi.

Resep dikunci sekarang:

- Kumo large cached/e4,seluruh56k outer-train,chunk128/allocator0,7,
  generator100×seed+fold. Dua konstituen menerima context/query yang sama.
- Utama:100numerik KS/Spearman baseline+4kategori kode. Selektor dipasang pada
  outer-train saja. Pool dan reference legacy utama tidak berubah.
- Union:poolutama228+poolteman1000 numerik (prefixteman__),4kategori sekali.
  Reference teman dibangun ulang dengan166 pada outer-train yang tepat,query
  targets dihapus. LGBgain400round v.LGB/threads4/deterministic CPU,100numerik
  dan4kategori raw. Filter Spearman/shortlist sama protokol kondisional.
- Model/feature selection hanya outer-train. Tidak memilih alpha,pasangan,
  kandidat,selector atau budget dari outer. Daftar90historis tidak dipakai.
- Candidate = mean percentile rank dari kedua prediksi pada cohortouter14k.
  Average-ties dihitung sebelum normalisasi,float64. Matched baseline utama
  juga dinormalisasi rank dalam cohort yang sama; simpan raw probabilities
  terpisah. Ini mencegah perubahan kalibrasi antarfold pada pooled KS diklaim
  sebagai manfaat konstituen kedua. KS/AUC per fold baseline raw/rank harus sama.

Screening seed42fold0 hanya evaluasi. Bila hasilnya layak diteruskan, jalankan
resep tetap yang sama pada5fold×seed42/7/2026; tidak memilih bobot perouter.
Seluruh reference,selector danvocab difit ulang peroutertrain. Tidak ada inner
optimization baru pada prosedur tetap ini. Jangan menjalankan pola fallback lama
yang akan mengembalikan baseline hanya karena gaininner tidak>0,003.

Gate breakthrough user **tetap**:Δlokal>0,003 terhadap Kumo matched dan pembanding
v31 yang relevan,mayoritasfold,3seed dirata-rata sebelum pairedbootstrap baris
customer, P(Δ>0)≥0,9,audit kebocoran/seleksi bersih,forecastpublic≥0,570 sebelum
submit. Bootstrap conditional tidak menghapus bias eksplorasi riset berulang;
laporkan pemilihan keluarga dari innerf0 dan keterbatasan tersebut.

Simpan raw/ranked baseline,union dancandidate denganID,target,fold,seed danhash.
Tandai mode fixed_policy; tidak mengadopsi selection.json lama sebagai policy
yang lolos. Tidak menggabungkan OOF historis teman. Tidak ada submission sebelum
seluruh gate;dua slot tidak dipakai untuk menguji ide langsung di leaderboard.
