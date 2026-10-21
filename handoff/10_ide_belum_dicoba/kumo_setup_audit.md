# Audit setup Kumo-Tabular kita (snapshot19Okt)

Pembaruan21Okt: audit153/157 selesai; native/gain100 dan pool teman telah diuji
dalam173/177, belum lolos gate. Log-uang181 lolossearch+0,003385 tetapi
confirmation−0,002713. Full232OOM padaallocator0,7;f154/198 juga nullhistoris.
VIPfeature152 tau0,5 turunouter−0,006096. Retrieval183 bestblend+0,002458,
belum>0,003. Same-class augmentation185 sedang diuji. Angka/protokol di bawah
merefleksikan rencana19Okt; status terkini ada di
[status aktif](../00_mulai_di_sini/status_sekarang.md) dan
[jurnal21](../04_riwayat_versi/jurnal_21okt_integrasi.md).

## Petunjuk dari user (19 Okt malam)
Teman user di public **0,57065** (#8 leaderboard 19 Okt, kemungkinan tim "BCC mau ke kopdes depok") memakai
**"kumo ensemble 3 seed"**. Kumo-only kita di public **0,56056** (v31c: Kumo large context penuh, 5 fold seed 42).
Selisih +0,010 hampir seluruh gap kita ke top-8. Karena pohon kita sendiri sudah 0,56337 dan jenuh, jalur paling
masuk akal ke 0,57+ adalah **Kumo yang lebih baik**, bukan blend.

## Apa yang dilihat Kumo kita sekarang (`eda/valid2.py::kumo_fit`, dipakai v28m/v31/v34/notebook)
| Aspek | Sekarang | Catatan |
|---|---|---|
| Fitur | `fm_select(X, y, tr, 100)`: 200 numerik teratas menurut **KS univariat** di train fold → greedy buang \|Spearman\| ≥ 0,95 → 100 + 4 kategori | 44 dari 104 kolom 100% NaN untuk thin-file; fitur "bersyarat" (cf_nonincome_share, pv_appr_max, tenure) tidak terpilih (siklus 2, 16) |
| Kategori | `loan_purpose`, `region`, `employment_type`, `housing_status` → `cat.codes` float, stype dipaksa **`numerical`** (`overrides={c: "numerical" for c in cols}`) | Kumo punya stype `categorical`; tidak pernah diuji |
| Preprocessing | tidak ada (nilai mentah; NaN dibiarkan; ±inf → NaN di `frames`) | Kumo menormalisasi internal; skala ekstrem (saldo jutaan IDR) tidak di-log |
| Context | seluruh train fold (56k) untuk OOF; seluruh 70k train untuk test (Kumo-70k, notebook) | subsample acak 20k −0,0018 |
| Estimator | `num_estimators=4`, generator seed `seed*100 + b + 1000*g_off` | e8 ≈ e4 |
| Ukuran | large (medium/small lebih buruk di fold 0) | – |
| Output | `to_pandas()["1"]` (benar; `numerical[...,1]` salah, bug 18 Okt) | – |
| Multi-seed | OOF 3 seed outer 0,52570 vs seed 42 0,52665; test Kumo-70k 3 draw estimator (`results/v41`) | public multi-seed belum pernah diuji |

## Varian yang layak diuji (fold 0, tanpa test, pembanding = Kumo ctx penuh fm_select e4 generator seed 0 → cache
`eda/out/kumo_finetune_f0_base.parquet`, KS 0,52865; ~20 menit per varian)
1. Kategori sebagai stype `categorical` (bukan kode numerik).
2. Seleksi fitur menurut gain LGB (`fm_select(rand=-1)`) atau gabungan KS ∪ gain (≈ 140 kolom).
3. Transformasi log/rank untuk kolom uang (saldo, outstanding, income) sebelum masuk Kumo.
4. Buang kolom yang 100% NaN untuk thin-file dari context thin-file? (seg_causilo null untuk Causilo; untuk Kumo belum.)
5. VIP-COP tingkat fitur (lihat `vipcop_context_optimization.md`).
6. Ensemble 3 seed outer + 3 draw estimator dari konfigurasi terbaik.
Kriteria: paired bootstrap fold 0 dulu, lalu 5 fold × 3 seed (Kumo ~20 menit/fold tanpa test lokal), bandingkan
keluarga Kumo dan v31 dengan **rata-rata seed**. Karena CV ICL tidak transfer ke public, keputusan akhir butuh probe
public (patuhi jendela submit user).

## Biaya
Kumo large lokal: fold 56k context + 14k query ~18–20 menit (e4); Kumo-70k test ~1,75 jam; forward 20k context +
3.000 query e1 ~4 s.
