# Blend tetap setelah seleksi kondisional, 21 Oktober 2026

Disusun ketika177 masih berjalan (utama_gain dan teman_gain selesai, union belum
dibaca). Seluruhnya tetap optimasi inner; outer keluarga input belum dibuka.
Teman_gain meningkatkan inner-searchAUC0,84293→0,84942 terhadap teman_raw,
mendekati baseline0,84941, tetapi KS0,53712 masih di bawahbaseline0,54608.
Ini alasan baru untuk menguji komplementaritas, tanpa mengulang blend pool KS
univariat176 yang sudahnull. Tidak mengasumsikan AUC sama berarti KS sama.

Kandidat baru tetap, tanpa alpha-search:

- Rata-rata probabilitas0,5utama_raw+0,5teman_gain.
- Rata-rata probabilitas0,5utama_raw+0,5union_gain.
- Rata-rata percentile rank0,5utama_raw+0,5teman_gain.
- Rata-rata percentile rank0,5utama_raw+0,5union_gain.

Bandingkan bersama baseline dan tiga single177. Percentile rank average-ties
dalam query cohort masing-masing role, tanpa label. Search memilih satu dari
delapan;gerbangΔ>0,003. Confirmation hanya baseline dan satu terpilih;Δ>0,003.
Tidak mengganti pasangan/bobot/rank atau mencoba kandidat kedua berdasarkan
confirmation. Bekukan seluruh keputusan sebelum outer. Ketidakpastian/gerbang
5fold×3seed dan submission tetap. Tidak menampilkan hasil ini sebagai holdout
baru:split inner sama173/176/177. Ini bukan empat bukti independen.

Input hanya cache173baseline+177challengers, diverifikasihash/fit/role/ID/
labelcontext/model/e4/generator/query order. Tidak menghitung ulang model untuk
screeningblend. Jika lolos, outer melatih hanya konstituen yang dibutuhkan dengan
reference danselector fit-only outertrain, lalu menghitung blend yang sudahbeku.
