# Indeks eksperimen lintas dua basis riset

Indeks per entrypoint,asal,deskripsi dan protokol: [indeks_eksperimen.csv](indeks_eksperimen.csv).
Klaim metrik historis dengan file dan JSON object path:
[indeks_metrik_dilaporkan.csv](indeks_metrik_dilaporkan.csv).
Indeks modul teman (fungsi/import/path): [indeks_modul_teman.json](indeks_modul_teman.json).
Pembuatan ulang indeks: `OMP_NUM_THREADS=4 .venv/bin/python eda/161_experiment_catalog.py`.

Indeks metrik berisi laporan bertingkat/berulang, **bukan jumlah eksperimen independen**.
Tidak menggabungkan OOF,tidak menginferensikan skor yang hilang,tidak menyetarakan versi/split.


Pembaruan setelah pause21Okt: v56belum lengkap11/15fit; CPU204/206 union/softSSL
kalah diinner dan tidak diteruskan. v59mengirim resep v41lama yang diaudit ulang:
public0,56296, belum melewati v31 0,56405. Bukti tersedia pada
[arsip submission](../../artifacts/history_utama/results/v59/v41_lowvar_today/).
`results/` dan contoh diagnostik ber-ID hanya lokal; laporan agregat dan kode
arsip tetap dipublikasikan. Tidak ada proses eksperimen yang dilanjutkan.

| Sumber/keluarga | Lokasi bukti | Keputusan dan status |
|---|---|---|
| utama,v1–v45 | [katalog utama](../../handoff/05_katalog_eksperimen/),[pelajaran/null](../../handoff/08_pelajaran/),[jurnal18–19](../../handoff/04_riwayat_versi/jurnal_18okt_sore.md) | Pohon jenuh,ICL transfer belum terjelaskan; fitur baru harus bersarang |
| utama,v46–v48,Kumo input/VIP | [jurnal19](../../handoff/04_riwayat_versi/jurnal_19okt_kumo_vipcop.md),[jurnal20](../../handoff/04_riwayat_versi/jurnal_20okt_kumo_context.md),[jurnal21](../../handoff/04_riwayat_versi/jurnal_21okt_integrasi.md) | VIPv48 fold0 selesai:null Δouter−0,006096/P0,147;tidak lanjut15fit. Audit153/157 dan smoke154 selesai |
| integrasi165/166,reference fiturteman | [protokol](protokol_reference_bersarang.md),[adapter](../../eda/teammate_feature_adapter.py),[runner](../../eda/166_teammate_nested_features.py) |7checkssintetis lulus;fullbuild100k selesai,48kfit/52kquery,1.000numerik+6kategori,rumusV3–V5asli |
| diagnosis167,public/private | [laporan](diagnosis_public_private.md),[JSON](../../eda/out/167_public_private_reversal.json) |1.000splitpaired9k/21k;proxy131lama21k terlalu besar,bukti penaltiICL melemah;bukan seleksimodel/submission |
| audit168,checkpoint Kumo | [JSON](../../eda/out/168_kumo_checkpoint_audit.json) | SHA256 cache large benar; revisi/commit/SDM sesuai deklarasi teman; bobot historis teman tidak tersedia untuk perbandingan byte |
| audit169,seleksiVIP/split | [JSON](../../eda/out/169_vip_selection_audit.json) |48/104fitur overlap;stableIDdigests melengkapi metadataobjecthash;tidak membaca skorouter |
| engineering170,memoriKumo | [JSON](../../eda/out/170_kumo_memory_preflight.json) |48kcontext/e4/query128,allocator0,7;3,79GiBallocated;tanpaskorKS |
| riset171,reliabilitasteacherSSL | [JSON](../../eda/out/171_ssl_teacher_reliability.json) |hanya labelinnersearch;tailpositif langka;fullSSLv49 selesai,12policy kalah,baseline0,546077 vs bestpseudo0,545334;fallbacknoSSL,Δouter0 |
| input172–174,Kumo paired | [protokol](protokol_fitur_teman_validation_v2.md),[runner](../../eda/173_nested_kumo_inputs.py) |8adapter+5gatechecks lulus;full173 selesai:null,utama_native+0,000974 saja,temanraw/native−0,018lebih;outerbelumdibuka |
| audit175,fitur nyata | [JSON](../../eda/out/175_inner_s42_f0.json) |7referenceharga+24summarycontoh cocok independen,0futureevente-commerce,9countercutoff0;tidak menguji seluruh formula/kualitas |
| blend176,inputfamilies | [protokol](protokol_blend_input_kumo.md),[runner](../../eda/176_input_family_blends.py) |Tiga blend tetap tanpa alpha-search,preregister60e828c sebelum membaca fullscore challenger;semuanya kalah,best0,543078vsbaseline0,546077 |
| conditional177/178,Kumo input | [protokol](protokol_kumo_conditional_inputs.md),[runner](../../eda/177_conditional_kumo_inputs.py) |LGBgain400round fit-only,utama/teman/union;4CPUchecks lulus;ketigasingle kalah di inner-search |
| blend179–180,kondisional | [protokol fixed](protokol_blend_kumo_conditional.md),[protokol grid](protokol_rankfusion_conditional.md) |Bestunionrank+0,002623<0,003;grid0–0,5 tidakmenolong;stableinteger-ranks melindungi ties,keputusan tetapfallback |
| geometry181/182 | [protokol](protokol_kumo_geometry.md),[runner](../../eda/181_kumo_input_geometry.py) |3checksCPU lulus;moneyrawsearch+0,003385 tetapiconfirmation−0,002713,null;full232engineeringOOM,bukanskorburuk |
| retrieval183/184 | [protokol](protokol_kumo_group_retrieval.md),[runner](../../eda/183_kumo_group_retrieval.py) |8shared-neighborhood×12k,e4;3checksCPU+membership/cat checks lulus;bestprobblend+0,002458<0,003,tidaklanjutouter |
| same-class mixup185/186 | [protokol](protokol_kumo_sameclass_mixup.md),[runner](../../eda/185_kumo_mixup_context.py) |4checksCPU+parent/formula/category/mask audit lulus;mix2k−0,004203,mix8k−0,010518,clonecontrol≈baseline;null |
| gradient187/188 | [protokol](protokol_kumo_sgd_preflight.md),[amendemen](protokol_kumo_sgd_small_episode.md) |FullSGD213,67Mparameter,episode1024/128 dan512/64 OOMforward,sebelumupdate/KS;2CPUclassmappingchecks lulus |
| fixed189–191 | [protokol](protokol_fixed_union_rank_outer.md),[runner](../../eda/189_fixed_kumo_union_rank.py),[orchestrator](../../eda/191_validate_fixed_union.py) |Fixedrank50:50utama/uniongain;outerf0Δ+0,005438/Pmatched0,906;full5fold×3seed terputus11/15fit;risetdipause,bukanbreakthroughlengkap |
| teman,v1–v5 | [laporanv3](../../artifacts/optimization/tenfold/laporan_submission_v3.md),[v4](../../artifacts/optimization/v4/laporan_submission_v4.md),[v5](../../artifacts/optimization/v5/laporan_submission_v5.md) | Nama versi berbeda; TabM+0,00153 OOF tetapi public turun; bukan kandidat promosi |
| teman,reverse/latent | [laporan](../../artifacts/reverse_engineering/laporan_reverse_engineering.md) | Seed generator tidak ditemukan; fitur laten tidak memberi gain konsisten |
| teman,Kumo90 | [laporan](../../artifacts/optimization/kumo/laporan_kumo.md),[kode](../../scripts/kumo_tabular.py) | Native categorical,cache,e8; perlu pembandingan setara di validation_v2 |
| teman,Kumo context/ensemble/features | [laporan null](../../artifacts/optimization/kumo_improvement/laporan_improvement.md) | Balanced/full56k/e16/core37/denoised/latent/TabICL belum memberi gain kuat; jangan ulang tanpa alasan baru |
| teman,Kumo+spline/fine-tune | [hybrid](../../artifacts/optimization/kumo_hybrid/laporan_hybrid.md),[pilot](../../notebook/11_kumo_finetune_pilot.ipynb) | Hybrid1/5fold menang; fine-tune inner memilihstep0 |
| teman,goal054 | [laporan](../../artifacts/optimization/goal054/laporan_riset.md),[batas bukti](../../artifacts/optimization/goal054/README.md) | TabPFN/EBM null; offset+0,000789 CI mencakup0; relational belum lengkap |

Daftar null dan kandidat integrasi berangka ada pada [audit perbandingan](perbandingan_repository.md).
Catatan riset asli teman dan cara reproduksi historis: [riset_teman.md](riset_teman.md).
Semua notebook sumber teman berada di folder notebook tunggal:
[indeks historis](../../notebook/README_teman.md).
