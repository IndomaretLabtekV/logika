/goal
Lanjutkan lomba DSC LOGIKA UI 2026 di /home/veinsan/Projects/logika (credit risk, metrik KS). Deadline notebook 30 Okt 2026 23:59 WIB. Bahasa Indonesia, jujur, klaim + angka. Jangan sebut "final". Kalau gua bertanya, jawab dulu, jangan langsung jalankan.

Baca dulu:
1. serah_terima_25okt/ (mulai 00_mulai_di_sini/README.md; wajib 06_kesalahan_dan_pelajaran/ dan 07_langkah_berikutnya/rencana_merge.md)
2. docs/dsc0152/README.md dan scripts/dsc0152/README.md
3. docs/integration/indeks_eksperimen.md (jangan ulang yang null tanpa alasan baru)

Kondisi: satu repo, tiga pipeline tim, belum pernah digabung secara model.
a. Utama: eda/, OOF/test 6 GBDT + Causilo di results/v22/, Kumo strict K64 2-seed OOF 0,52740 (fold seed 42); v31 public 0,56405.
b. Teman: Kumo V5 + spline 25% + koreksi offset laten re_* (scripts/research_goal054.py, hybrid_kumo.py); goal054 OOF 0,52930 (fold seed 31027), public 0,56470 = best. OOF/test per fold tidak tersimpan; fitur V5 per fold seed 42 ada di results/v56/teammate_features/outer_s42_f*/.
c. DSC0152 (root + Agas + Dimaz): scripts/dsc0152/, public 0,56424 (Agas v66 ext0.5·tm30·lgb_lean); OOF+test per kaki di results/dsc0152/agas/v6x/output/preds_*.npz (by ID).
Fakta: Kumo-only mentok ~0,562 public; pohon paling bagus transfer ke public; offset laten di atas Kumo transfer; OOF tertinggi tidak selalu menang di public.

Tugas:
1. Review ketiga pipeline (fitur, validasi, model, blend, OOF vs public): kekuatan, kelemahan, overlap, dan kombinasi yang belum pernah dicoba.
2. Padukan: (i) blend OOF yang sudah ada (pohon/Causilo utama + kaki DSC0152 + K64); (ii) reproduksi resep teman di fold seed 42 (Kumo V5 ctx 50k e8 + spline + offset re_*); (iii) gabung fitur terkuat lintas pipeline (pp_*, factor model DSC0152, re_*, V5) bila beralasan.
3. Arahkan ke SATU pipeline gabungan yang bersih, reproducible, dan bisa dilanjutkan (satu entry point, satu skema fold), lalu iterasi sampai pooled OOF KS setinggi mungkin. Target 0,53–0,54.

Validasi:
- StratifiedKFold 5 seed 42, OOF pooled 70k. Samakan fold sebelum klaim; OOF beda skema fold hanya eksplorasi.
- Bobot blend/resep dikunci sebelum lihat skor outer (tulis protocol JSON); seleksi/tuning hanya di inner/cross-fit; preprocessing & reference difit per partition.
- Per kandidat: KS/AUC pooled & per fold, Δ vs REF 0,52625 dan vs komponen terbaik, paired bootstrap P>0. Kandidat murah langsung 5-fold (satu fold menipu).

Eksekusi:
- Semua run lokal (RTX 3070 Ti 8 GB). Satu job berat sekaligus, taskset -c 0-11 nice -n 10, OMP/n_jobs 4.
- Wajib eda/thermal_guard.py (pola ter-anchor), cache atomik tmp+fsync+rename, job dipecah per fold (laptop sudah hang 4x).
- Tiap analisis satu .py di eda/, cetak angka + gambar ke eda/out/. Catat bukti & keputusan di jurnal baru handoff/04_riwayat_versi/ dan indeks_eksperimen.md.
- Pipeline gabungan yang dipromosikan → update notebook/build_final.py dan uji replay byte-identik.

Submission:
- Sisa kuota dipakai HANYA untuk kandidat pooled OOF KS 0,53–0,54 yang lebih baik dari best dan jelas berbeda dari submisi sebelumnya (cek Spearman). Tulis perkiraan public ± sd sebelum submit, analisis sesudahnya. Satu per satu, maks 3/hari (reset 07:00 WIB).
- CSV 30.000 baris application_id,default_probability; ID unik lengkap, nilai [0,1], join by ID.

Larangan: data eksternal, leakage masa depan (cutoff decision_datetime), LB probing/tuning ke public, menulis token ke file, git push.

Pakai skill relevan (mlcraft, ml-training-recipes, 0-autoresearch-skill). Jangan berhenti di rencana; terblokir → lapor jujur.
