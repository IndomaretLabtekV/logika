# Amendemen engineering SGD: episode512/64

21Okt, setelah preflight1024/128 terminal OOM, sebelum percobaan kedua.
Tidak ada langkahtraining atau skor validasi yang selesai pada percobaan pertama.
OOM terjadi di forward pada peakallocated5,123GiB, bukan saat membuat state SGD.
Seluruh213.668.250 parameter disetel trainable (rowembedding9.964.560,
ICL203.703.690). Ini informasi kapasitas, bukan hasil model.

Percobaan baru mempertahankan seluruh parameter, SGDlr0,01/momentum0/foreachFalse,
AMPFP16/GradScalerinitial128,clip1,generator4200 dan dua langkah. **Hanya** ukuran
episode training berubah menjadi context512/query64. Parent semuanya inner-fit.
Tidak mengurangi context inferensi/validasi penuh48k–56k atau memakai labelheldout.
Jika muat, tidak boleh menyebutnya bukti peningkatan KS atau bukti bahwa SGD saja
menyelesaikan OOM AdamW; episode juga berbeda dari historis.

Implementasi187 menambahkan flag eksplisit --small-episode; default lama tetap
1024/128. Sumber lama dipertahankan byteidentik di research/locked_recipes/.
Direktori/log/manifest baru, tidak menimpa hasil OOM sebelumnya. Allocator tetap0,7,
satujobberat,cgroup/affinity/NI sama. Tidak ada fallback otomatis ke episode lain.
Protokol kualitas penuh akan dibuat terpisah hanya setelah gradient/update terbukti.
