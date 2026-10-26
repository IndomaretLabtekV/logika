# Riwayat proyek sebelum sesi ini (11–23 Oktober 2026), ringkas

Sumber lengkap: `handoff/04_riwayat_versi/*` (v1…v45, jurnal 17–21 Okt, `ringkasan_skor.md`, `kronologi_chat.md`),
`docs/integration/indeks_eksperimen.md`, `handoff/08_pelajaran/yang_tidak_efektif.md`.

## Fase 1 — 11–16 Okt: fitur dan pipeline dasar
- v1 0,53139 → v2 0,54192 (fitur perilaku v2, +0,011 public) → v3 0,54545 → **v5 0,56145 (price premium `pp_*`,
  +0,016 — lompatan terbesar lomba)** → v6 0,56123, v8 0,56145, v9 0,55917.
- Probe S3–S5 (16 Okt): stack logit-LR 0,55778, bagian NN 0,55866, bagian GBDT 0,56133.
- Temuan: **OOF lama bias** (early stopping di fold OOF) → semua OOF v1–v9 tidak bisa dipakai sebagai pembanding.

## Fase 2 — 17–18 Okt: validation_v2 dan zoo
- `validation_v2` (`eda/valid2.py`): outer StratifiedKFold 5 seed 42/7/2026; inner 1/7 untuk budget; refit ×1,15;
  blend cross-fit. v22 = resep v8 bersih **OOF 0,52426**.
- Zoo TabArena (TabPFN-3.5, TabICLv2, RealMLP, AutoGluon, iLTM, Mitra, TabFM…) null kecuali **Kumo-Tabular large**.
- **v31 = 6 GBDT 1/3 + Causilo 1/3 + Kumo context penuh 1/3 → public 0,56405** (OOF analog 0,52613; rekor sampai
  24 Okt malam). v31b (tanpa pohon) 0,55709; v31c Kumo saja 0,56056; v32 pohon 1/3 + Kumo 2/3 0,56376;
  v34 (resep notebook, Kumo-70k) 0,56271; v35 0,56298.
- Notebook submisi (`notebook/build_final.py`) lulus run Kaggle 18 Okt.

## Fase 3 — 18–19 Okt: 35 siklus riset, semua null di level keluarga
- Bug `kumo_finetune.py` (urutan kolom kelas) diperbaiki; fine-tune Kumo null; ~20 keluarga fitur baru null; stacking,
  distilasi, feature bagging null.
- Probe public 19 Okt: **pohon saja 0,56337**, Causilo saja 0,55333, LR rank-gauss 0,55835 → **CV keluarga ICL tidak
  transfer ke public sebaik pohon** (offset public−CV: pohon +0,0389, LR +0,0372, Kumo +0,0339, Causilo +0,0292).
- Winner's curse seleksi fitur dibuktikan (30 pasangan dari 7.080: +0,0036 palsu → nested −0,0005).

## Fase 4 — 19–21 Okt: Kumo setup, VIP-COP, SSL, integrasi
- VIP-COP fitur (v48) fold 0: −0,0061 → null. SSL v49 pseudo-context (12 policy) kalah inner. Retrieval v53 +0,0025
  (<gate). Same-class mixup null. v56 Kumo dengan fitur teman: 11/15 fit, pooled seed42 0,52781 / seed7 0,52744
  (reference legacy), berhenti karena hang. CPU v57/v58 union fitur teman null untuk pohon.
- **v59** (6 GBDT / Causilo 5 seed / Kumo-70k 3 draw) disubmit 21 Okt → **0,56296**.
- Repo dikonsolidasi 21 Okt: 508 file teman masuk (`scripts/`, `notebook/`, `artifacts/`, `results/teman/`);
  **OOF/model/dataset olahan teman tidak ikut**.

## Fase 5 — 23 Okt: EDA & preprocessing strict (tanpa training, atas arahan user)
- Pool strict: main232 (reference harga fit-only) + 664 fitur prefix teman = **896** → 792 (dedup/konstan) → 777
  (affine). PSI fit/test maks 0,0016; stratified 5-fold tetap. Audit cutoff/entitas bersih.
- Protocol 23 Okt (`artifacts/optimization/continuation_23okt/protocol.json`): 3 view × 2 seed, cap500 & uncapped
  dengan context 8.192 (VRAM), nested_small context penuh. Preflight: uncapped 8192/e8 **OOM**.
- VIP-COP baris (217): VIP 0,48587 vs all48k 0,48886 → null. Run pertama berakhir **hard freeze** (GPU 87 °C, CPU
  103 °C, dugaan termal).

## Pipeline teman (paralel, di repo yang sama)
- V3 public 0,55845, V5 0,55680, Kumo large (3 fold seed 8247 OOF 0,52807) public **0,56262**, Kumo 75% + spline 25%
  (5 fold seed 31027 OOF 0,52851) public **0,56393**, goal054 (koreksi kalender/offset, TabPFN, EBM) OOF terbaik
  0,52930 — **belum disubmit sampai sesi ini** (lihat `05_riset_dan_temuan/pipeline_teman_goal054.md`).
