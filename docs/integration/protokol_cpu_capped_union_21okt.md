# Tambahan100fitur teman pada seluruh fiturutama

21Okt21:36 sebelum206.204fullunion inner0,514859 vs main0,524007;
SSL0,506236. Tidak adaouter/testmodel/refit15 yang dijalankan untuk204.
Hipotesis baru:1000turunan teman mendominasi samplingkolom/menambahnoise;
pertahankan semua232kolommain dan tambah100numerikteman saja.

Semua aturan CPU/inner-stop/rounds/soft-SSL/finalblend0,75v31+0,25model
dari protokol_cpu_union_ssl_21okt tetap. Satu perubahan keluarga input:
gunakan gain_columns dariconditional_input_pools.py pada poolteman saja
(1000numerik+4kategori), gainLightGBM400round+Spearman<0,95 memilih100numerik.
Selector dipasang pada48kinnerfit untukkeputusaninner, dan56koutertrain
untukrefitouter. Empatkategori tetap sekali darimain. Outer tidak memilih
jumlahfitur,cap,bobot ataukebijakanSSL. Tidak mengambil selector lama yang
memakai reference/input berbeda. Simpan daftar dan metadata selector.

206menyimpan lineage sumber204 sebagai derivasi, bukan menimpa runnullnya.
Screening42/0: lanjut5fold×3seed hanya jikamax(union,SSL)>main diinner.
Gateincremental danbreakthrough tidak berubah. Jangan menyebut hasil
inner sebagai bukti public;15fit danpairedbootstrap tetapdibutuhkan.
