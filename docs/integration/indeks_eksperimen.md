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
| audit215/218,distribusi/CV/entitas | [laporan](../../artifacts/optimization/continuation_23okt/laporan_riset.md),[215](../../eda/215_distribution_cv_audit.py),[218](../../eda/218_entity_cutoff_audit.py) | PSImaks0,0016;0key lintas-customer;0timestamp≥cutoff;0label matang12bulan sebelum batas bulan;StratifiedKFold tetap |
| prep216,affine view | [kode](../../eda/216_affine_preprocessing.py) | 792→777,15kolom affine fit-only,0gagal pada52k nonfit;tanpa klaim gain |
| VIP217,blok baris Kumo beku | [laporan](../../artifacts/optimization/continuation_23okt/laporan_riset.md),[kode](../../eda/217_row_vip_context.py) | null: confirmation VIP0,48587 vs all48k0,48886(Δ−0,0030,P0,32) vs uniform-best0,48686;jangan lanjut5fold |
| rencana SSL lanjutan | [protokol](protokol_ssl_lanjutan_23okt.md) | rencana saja: self-distillation cross-fitted, R-CPL;bukan ulang grid threshold v49/v57/v58 |
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
| resep acuan strict 220/221, 24Okt | [jurnal24](../../handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md),[amendemen](../../artifacts/optimization/continuation_24okt/amendment_01.json) | 5fold s2026: cap500@8k 0,51885; nested_small(union) 0,50936; K 0,51960 <REF0,52625; uncapped/s2027 belum. Seleksi union (fitur teman) −0,017…−0,023/fold vs legacy |
| diagnosis223,baseline strict f0 | [kode](../../eda/223_strict_vs_legacy_features.py) | pp_* strict=legacy (ρ1,0000); baseline main100 f0 0,52629≈legacy0,52556 → penurunan dari seleksi union |
| screening222 context/median/SSL | [protokol](protokol_context_ssl_24okt.md),[kode](../../eda/222_context_ssl_screen.py) | inner f0: ctx8k0,50494→16k0,51436(+0,0094); median +0,0001/−0,0037 null; S3 pseudo-d0 −0,0016 (d1/perm lihat jurnal) |
| v62 ctx16k+main_small 224, submit v63 K62 | [protocol](../../artifacts/optimization/continuation_24okt/protocol_v62_ctx16k.json),[kode](../../eda/224_strict_tfm_variant.py) | ctx16k 0,52426(+0,0054 vs 8k,5/5); main_small e8 0,52553; K62 0,52662/KT62 0,52654/KTC62 0,52634; **public K62 0,56203** (v31c legacy 0,56056) |
| screening226 lebar vs context, v64 | [kode](../../eda/226_width_context_tradeoff.py),[protocol](../../artifacts/optimization/continuation_24okt/protocol_v64_cap300_ctx28k.json) | inner: cap300@28k 0,52533 vs cap500@16k 0,51436 vs cap200@40k 0,51831; outer cap300@28k 0,52549 (5/5 vs 16k); K64w 0,52724 (P>0 vs REF 0,75); SSL S3 pseudo null −0,0022 |
| 227/228/229, v65 main150 | [jurnal24](../../handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md) | e16 cap300@28k OOM; pool 777 dedup −0,0052 (P0,18) null; wide cap128@48k OOM; main150 inner +0,0034 → outer 0,52597/0,52606 (≈main100), K65 2seed 0,52568 < K64 2seed **0,52740** |
| sumber DSC0152 (root/Agas/Dimaz), integrasi 25Okt | [docs/dsc0152](../dsc0152/README.md),[pemetaan](../dsc0152/pemetaan_file.csv) | public terbaik 0,56424 (Agas v66_04 ext0.5·tm30·lgb_lean); CV 10–15 fold AUC ±0,848; OOF+test per kaki di results/dsc0152/agas/v6x/output; 592 file terverifikasi SHA256 |
| 231–240 literatur & struktur, 24Okt | [jurnal24](../../handoff/04_riwayat_versi/jurnal_24okt_resep_lokal.md) | 8×e1 sekuensial (context penuh muat, AUC plateau 0,841); anchored −0,0086; offset bulan cross-fit +0,0007; stacking −0,0009; DART (AmEx#1) −0,0062 f0; series OOF ecom −0,0003 f0 / cicilan −0,0013 5f (f0 +0,0055 kebetulan); KNN target (HC#1) −0,0040 f0; CoLES SSL ecom −0,0026 / cicilan −0,0007 5f; simulasi ruin arus kas (241) −0,0004 5f; GRU-NN (242) +0,0019 vs LGB tapi −0,0001 di atas K64 / −0,0005 di atas REF (243). seed ketiga K64 (v69) 0,52685 < 2-seed 0,52740; e24 sekuensial AUC datar; 246 berburu 42 agregat baru di pita KS −0,0024; 247 target analog pinjaman lama +0,0006 (P0,62); 248 bobot kelas/label smoothing null. Notebook replay v59 + banner lulus (byte-identik). Semua null untuk blend terbaik |
| merge 250–263, 25Okt | [jurnal25](../../handoff/04_riwayat_versi/jurnal_25okt_merge.md),[protokol](../../artifacts/optimization/continuation_25okt/),[entry point](../../eda/260_merge_pipeline.py) | seed42: R(goal054 direproduksi) 0,52693 / R2 0,52722 (ρ test vs goal054 0,9997); offset re_* −0,0003/−0,0016; D(DSC0152 v66 inner-stop) 0,52614 (v66 eksplorasi 0,52871 optimistis); KF −0,0007 null; KD 0,52556/KD2 0,52593; merge R2DK **0,52886** (P>0 vs REF 0,97, dipromosikan, replay byte-identik), R2D 0,52942, R2DK_KD2 0,52892; cross-fit bobot null; T5 LGB-V5 0,52065 null; audit referensi populasi DSC0152 tidak material (ρ≥0,9999); belum 0,53, tidak disubmit |

Daftar null dan kandidat integrasi berangka ada pada [audit perbandingan](perbandingan_repository.md).
Catatan riset asli teman dan cara reproduksi historis: [riset_teman.md](riset_teman.md).
Semua notebook sumber teman berada di folder notebook tunggal:
[indeks historis](../../notebook/README_teman.md).
