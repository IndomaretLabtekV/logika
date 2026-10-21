# Context Kumo menurut lingkungan query

21Okt setelah181 terminal; sebelum fitting retrieval atau skor baru. Full-context
baseline utama_raw173/e4 masih pembanding. Ini mengubah kebijakan context secara
eksplisit, bukan mengurangi realcontext pada runSSL atau mengubah hasilnull lama.

Motivasi: [LoCalPFN,Thomas etal2024](https://arxiv.org/abs/2406.05207) mengeksplorasi
retrieval+fine-tuning pada TabPFN. [TabDPT](https://arxiv.org/abs/2410.18164) juga
memakai retrieval. Tidak mengasumsikan hasil tersebut berlaku untuk Kumo. Metode
di sini adalah **shared-neighborhood approximation**, tanpa fine-tuning; bukan
reproduksi LoCalPFN atau exactkNN perquery. Query-specificfit ribuan kali terlalu
mahal, sehingga query berbagi neighborhood dari8centroid yang fit pada innerfit.

Riwayat yang membatasi hipotesis: randomcontext20k Kumo−0,0018OOF; segmentasi
Causilo dan persona/cluster fiturtidak membantu. Jangan menyebut cluster sebagai
tipe customer diskret baru. Yang diuji adalah pemilihan context menurut jarak,
bukan penambahan fiturcluster atau mengulang randomsubsample sebagai kandidat.

Konfigurasi tetap sebelumrun:

- Pilih100numerik+4kategori dari innerfit denganselector baseline yang sama;
  kolom/vocab raw Kumo tidak berubah. Retrieval memakai20numerik pertama dari
  ranking tersebut dan4kategori. Tidak menambah prediksi model sebagaifitur.
- QuantileTransformer256 keuniform[0,1],fit innerfit saja;NaN→0,5,missingmask×0,5.
  Empatkategori onehotfit-only/unknownignore,onehot×0,5. Tidak memakaiquerylabel.
- MiniBatchKMeans8,n_init3,batch2048,max_iter100,seed=100×seed+fold,fit48k saja.
  Query ditugaskan ke centroid terdekat berdasarkan fitur saja.
- Setiapcentroid memilih12.000baris innerfit terdekat (Euclidean),tanpalabel
  balancing. Ties menurut rowIDorderkanonik;context diurutkan kembali kanonik.
  Context antarcentroid bolehoverlap;setiapquery menerima satu kelompok/e4.
  Vocab Kumo dipasang pada12kcontext yang dipakai,unknownquery→missing.
- Kumo large/e4/generator100×seed+fold,chunk128/allocator0,7. Baseline tetap48k;
  pembandingukuran adalah satu subset uniform12k dengan seed tetap untukinner-search saja,
  bukan calonpromosi. Totalfitcontextretrieval8×12k lebihmahal dari1×48k;
  estimatorperquery samae4. Tidak mengklaim totalcompute persis identik.

Calon: baseline;retrieval;meanprob baseline/retrieval masing0,5;meanpercentilerank
masing0,5 (average-ties dihitung di cohortrole,integer doubledranks sebelumnormalisasi).
Tanpa alpha/cluster/k/geometry search atau prior correction dari querylabel.
Inner-search4k memilihsatu,ΔKS>0,003. Confirmation4k hanya baseline+satuterpilih,
Δ>0,003,selainitu fallback. Sameinnercohort denganrisetterdahulu,bukan bukti
holdoutbaru. Bekukan sebelumouter;outer refit selector,geometry,centroid,context
danvocab di56koutertrain. Gateuser5fold×3seed/bootstrap/audit/forecast tetap.

Audit wajib: query/contextdisjoint,semuacontextsubsetfit,coveragequerytepatsekali,
IDlabelalignment,kategoriunknown,ukuran/classratecontext,geometryqueryinvariance.
Modelcache setiapcentroid denganID/label/kolom/RNG/source/data hash;tidakmemakai
outerrows dalam innerTensor. Satujobberat;tanpa perubahanSDK/checkpoint/system.
