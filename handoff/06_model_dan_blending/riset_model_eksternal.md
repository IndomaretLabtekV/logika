# Riset model eksternal (web, 13–15 Okt 2026)

## Ringkasan TabArena / literatur (per 15 Okt)
- Teratas di TabArena: **LimiX-2**, **TabPFN-3.5**, TabFM, dan **Xiaomi-TabLDM**.
- Dari model-model itu, yang realistis di T4x2 dan sudah kita coba: TabPFN-3.5 (0,519), Causilo (0,524), EXAONE (probe saja), LimiX-2 (hang), TabICLv2 (OOM). Hasilnya 0,519–0,524 atau tidak muat.
- Xiaomi-TabLDM dan TabFM belum dicoba. Kelayakannya di T4 belum dicek.
- Benchmark umum bukan estimasi KS untuk dataset ini. Di sini, GBDT/TabM dengan fitur manual yang bagus sudah setara atau lebih baik dari FM.

## Sumber
- LimiX-2 tops TabArena: https://www.intelligentliving.co/limix-2-tops-tabarena-tabular-ai/
- State of tabular foundation models (2026): https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models
- Xiaomi-TabLDM (arXiv): https://arxiv.org/html/2609.03880v1
- TabPFN-3.5 report (arXiv): https://arxiv.org/html/2609.17895v1
- TabArena overview: https://www.codesota.com/tasks/tabular-ml
- LimiX README: https://cdn.jsdelivr.net/gh/limix-ldm-ai/limix@main/README.md
- LimiX-2M HF: https://huggingface.co/stableai-org/LimiX-2M
- TabPFN-3.5 HF: https://huggingface.co/Prior-Labs/tabpfn_3_5 · repo https://github.com/PriorLabs/TabPFN
- TabPFN-Rel: https://github.com/PriorLabs/tabpfn-rel · RelArena https://github.com/PriorLabs/relarena
- Causilo: https://huggingface.co/nums-ai/causilo · https://github.com/nums-ai/causilo
- EXAONE-Tabular: https://github.com/LGAI-Research/EXAONE-Tabular
- Mitra-v2: https://huggingface.co/autogluon/mitra-classifier-2
- TabICL: https://github.com/soda-inria/tabicl · paper https://arxiv.org/abs/2602.11139
- PyTabKit (RealMLP): https://github.com/dholzmueller/pytabkit
- LimiX-2 (link user): https://huggingface.co/stable-ai/LimiX-2

## Batasan teknis yang ditemukan
- Kaggle: Python 3.13, torch 2.11, T4 (compute capability 7.5, **tanpa flash-attn**, tanpa bf16 cepat).
- Model yang wajib flash-attn atau butuh VRAM di atas 15 GB per GPU tidak praktis. Dua T4 tidak bisa digabung jadi satu pool memori untuk satu model.
- In-context learner: biaya kira-kira kuadratik terhadap context dan linear terhadap query. **Query harus di-chunk.**

## Tambahan 16 Okt
- TabM API dan default dicek di https://github.com/yandex-research/tabm (`tabm.py`); dipin tabm 0.0.3,
  rtdl_num_embeddings 0.0.12 (v11/v12).
- EBM: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html, https://github.com/interpretml/interpret (v13).
- TabICL (`pip install tabicl`, `TabICLClassifier(n_estimators, batch_size=1, device, kv_cache=False)`): kunci agar
  muat di T4 adalah **memecah query** (`predict_proba` per 2.000 baris). Context 56k + 8 estimator ≈ 1 jam per fold
  di satu T4; 40k + 4 estimator ≈ 16 menit.
- Causilo (`pip install --no-deps causilo`, `CausiloClassifier(n_estimators, random_state, device)`), TabPFN
  (`TabPFNClassifier(n_estimators, device, ignore_pretraining_limits=True, categorical_features_indices)`, butuh token).
- LightGBM objective `rank_xendcg` dan XGBoost `rank:pairwise` butuh grup; dipakai grup acak 1.000 baris (v15).
- Kesimpulan riset 16 Okt: tidak ada model eksternal yang mengubah gambaran; masalah utama ada di validasi, bukan model.

## Tambahan 17–18 Okt
- **NVIDIA Kumo-Tabular** (paket PyPI `structured-data-models` 0.2.1, import `sdm`): `sdm.models.KumoTabular(task=
  "classification", size="small|medium|large", device)`; input `sdm.TableTensor.from_pandas(df, stypes)` dengan
  `sdm.infer_stypes(df, overrides={"target": "categorical", <fitur>: "numerical"})`; prediksi
  `model(x_context=..., y_context=..., x_query=..., num_estimators=4, generator=g)`, probabilitas kelas "1" =
  `out.to_pandas()["1"]` (= `out.numerical[..., 1]`). Bobot large fp32 816 MB, diunduh otomatis. Paket yang sama memuat
  **TabFM** (Google, lisensi non-komersial → `accept_license=True`, ~7 GB, wajib fp32) dan **TabICLv2**.
  KumoRelational ditunda: sampler graf mengambil tetangga terbatas (merusak agregat seperti pp_mean), basis TabICLv2
  lemah di data ini, v14 set-encoder baris mentah null, biaya ~½ hari.
- **TabArena** (`tabarena` 0.1.0 + `autogluon.tabular` 1.6.3, `.venv-zoo`): model zoo lewat AutoGluon (LimiX, TabFM,
  Mitra, TabDPT, Nori, iLTM, TabICL/v2, RealMLP, TabSwift, LGB/XGB/Cat/NN/FastAI/Linear/RF/XT/KNN). Estimasi memori
  AutoGluon menolak banyak model di 8 GB / 15 GB; rasio memori bisa dinaikkan. TabDPT gagal load checkpoint
  (`Missing key enc_cell_dim`) karena versi wrapper tidak cocok.
- TabPFN-3.5 (paket `tabpfn` 9.0.0, gated, token PriorLabs/HF) hanya bisa lokal (kernel CLI tanpa secret).
- Kesimpulan riset 17–18 Okt: zoo TabArena praktis tuntas; satu-satunya model eksternal yang menambah adalah Kumo-Tabular
  large dengan context penuh. Model > 8 GB VRAM tidak memberi apa-apa di T4.

## 19 Okt
- **VIP-COP** (Chen, Ding & Akoglu, CMU; arXiv 2605.12904, "Black-box Context Optimization for Tabular Foundation
  Models"): pilih satu context global (baris dan/atau fitur) untuk TFM beku dengan importance sampling softmax(φ/τ) +
  regresi KernelSHAP atas AUROC validasi; dibandingkan acak, ensemble 20 context, context penuh, KMeans, DT+TFM.
  Ringkasan lengkap + rencana implementasi untuk Kumo: `10_ide_belum_dicoba/vipcop_context_optimization.md`.
- WebSearch kompetisi: tidak ada tulisan publik untuk DSC LOGIKA UI 2026; tesis UMN (kc.umn.ac.id eprint 43840) hanya
  mendeskripsikan penyelenggaraan edisi 2025. Tidak ada simulator data kredit sintetis yang cocok.
