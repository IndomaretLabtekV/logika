# Riset dan literatur yang dibaca sesi ini

## Dokumen proyek yang dibaca (urut perintah goal)
1. `artifacts/optimization/continuation_23okt/laporan_riset.md` + `protocol.json` (resep acuan, pool strict, VIP-COP
   baris null, insiden freeze).
2. `handoff/00_mulai_di_sini/status_sekarang.md`, `handoff/01_kompetisi_dan_aturan/`, `handoff/10_ide_belum_dicoba/`.
3. `handoff/07_validasi_dan_leaderboard/validation_v2.md`, `docs/integration/protokol_ssl_lanjutan_23okt.md`,
   `docs/integration/indeks_eksperimen.md`.
4. `eda/212–218`, `strict_kumo_engine.py`, `strict_feature_pool.py`, hasil `results/eda_23okt/*`.
5. Siang: `handoff/08_pelajaran/yang_tidak_efektif.md`, `artifacts/reverse_engineering/laporan_reverse_engineering.md`,
   `artifacts/optimization/goal054/{README,laporan_riset}.md`, `docs/integration/riset_teman.md`,
   `data/DATA_DICTIONARY.md`, source `sdm` (`models/kumo/tabular/{model,row_embedding}.py`, `relational/model.py`).

## Literatur eksternal
| Sumber | Teknik | Dipakai di | Hasil di data ini |
|---|---|---|---|
| VIP-COP (Chen, Ding, Akoglu; arXiv 2605.12904) | optimasi context/fitur black-box TFM | sesi lama (v48, 217); sesi ini varian "context maksimum" | context besar membantu; seleksi baris/fitur null |
| Kode juara 1 AmEx (github jxzly) | LGB DART ff 0,05; series OOF per statement; GRU + MLP; ensemble 0,3/0,35/0,15/0,1 | 236, 237, 242 | null / null / +0,0019 hanya vs LGB |
| Juara 1 Home Credit (ringkasan Medium) | rata-rata target 500 tetangga di ruang fitur kunci; FE > tuning | 239 | null |
| CoLES (Babaev dkk., SIGMOD'22, arXiv 2002.08232) | contrastive self-supervised event-sequence embedding | 240 | null |
| R-CPL (Kim dkk. 2023, arXiv 2302.14013) | curriculum pseudo-label berbobot densitas | direncanakan 23 Okt | belum diuji |
| Home Credit Credit Risk Model Stability 2024 | (write-up tidak terjangkau; hanya press release) | – | – |

Pencarian web dibatasi: halaman Kaggle/Medium memblokir fetch (403/429); kode AmEx dibaca via GitHub API.

## Reverse engineering generator (laporan teman, dibaca ulang)
- Aturan pasti: cicilan = ceil(anuitas); harga kelipatan Rp1.000 × qty; saldo akun konsisten; delay keputusan campuran
  3 uniform (0,2–3 j 45%, 6–30 j 35%, 36–120 j 20%).
- Noise terestimasi: log harga item sd 0,654, preferensi customer sd 0,224, order 0,061; skor biro noise ≈23 poin per
  provider; noise log income bulanan 0,26/0,34/0,42 (salaried/self-employed/variable).
- Seed RNG tidak ditemukan (PCG64/MT19937 10k seed, ID sebagai output RNG, dll).
- 21 fitur laten → tidak membaik sebagai input CatBoost/spline; tapi sebagai koreksi offset Kumo (goal054) membaik.
