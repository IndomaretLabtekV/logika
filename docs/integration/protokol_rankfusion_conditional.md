# Bobot rank fusion kondisional yang dipilih nested

21Okt, setelah179 selesai; sebelum menilai grid bobot. Fixed50% union_gain_rank
memberi searchΔ+0,002623, sehingga **tidak lolos** gerbang>0,003. Teman_gain
rank/probability kalah. Hipotesis lanjutan: kontribusi pool gabungan perlu lebih
kecil dari0,5 karena single union_gain lebih lemah, meski prediksinya komplementer.
Ini eksperimen bobot baru, bukan mengubah aturan177/179 atau menyebut null lolos.

Hanya pasangan utama_raw+union_gain, rank average-ties dalam cohort query tanpa
label. Bobot union ditentukan dari grid **0;0,1;0,2;0,3;0,4;0,5**, sisanya utama.
Tidak mencoba alpha tambahan setelah hasil, pasangan lain, offset/subset atau
optimasi tepat terhadap setiap perpindahan urutan. Budgetnya nol fit tambahan.

Search4k memilih satu bobot (ties pilih bobot kecil). ΔKS>0,003 terhadap alpha0
baru boleh menilai confirmation4k pada bobot itu, wajibΔ>0,003; selain itu fallback0.
Tidak menilai kandidat kedua dari confirmation. Simpan bobot/pilihan/hash sebelum
outer. Confirmation memakai split inner yang sama; tidak diklaim bukti independen
antarproyek173/176/177/179 atau holdout baru setelah riset terdahulu.

Jika lolos, validasi model selection sebagai prosedur nested: masing-masing outer
fold×seed membangun ulang reference/selector/vocab, menentukan bobot pada inner,
lalu fit dua konstituen pada outertrain dan menggabungkan prediksi outer dengan
bobot yang beku. Bukan satu bobot dari seluruhouterOOF. Gerbang5fold×3seed,
Δlokal>0,003,mayoritasfold,pairedbootstrapP≥0,9,auditbias dan forecastpublic≥0,570
tetap berlaku. Tanpa lolos jangan memakai slot submission untuk probing.
