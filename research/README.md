# Research papers for LOGIKA UI 2026 credit risk

This is a curated local library for the contest's multi-table, point-in-time credit-risk task. The PDFs support method selection and notebook design; they are not training data and must not be joined to competition records. The [EDA and research report](../EDA_RESEARCH_REPORT.md) connects these papers to the measured dataset.

`papers.json` records titles, source pages, download URLs, and relevance. `download_status.json` records the outcome, file size, and SHA-256 digest for each PDF. Run `python research/download_papers.py` from the repository root to retry downloads and refresh the status file. The script checks for a PDF header before accepting a file.

## Available PDFs

| Paper | Local PDF | Why it matters |
|---|---|---|
| [LightGBM, Ke et al. (2017)](https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | [lightgbm_2017.pdf](papers/lightgbm_2017.pdf) | Fast gradient boosting baseline |
| [XGBoost, Chen and Guestrin (2016)](https://arxiv.org/abs/1603.02754) | [xgboost_2016.pdf](papers/xgboost_2016.pdf) | Regularized tree boosting baseline |
| [CatBoost, Prokhorenkova et al. (2018)](https://arxiv.org/abs/1706.09516) | [catboost_2018.pdf](papers/catboost_2018.pdf) | Ordered boosting and categorical treatment |
| [Why trees outperform deep learning, Grinsztajn et al. (2022)](https://arxiv.org/abs/2207.08815) | [trees_tabular_2022.pdf](papers/trees_tabular_2022.pdf) | Method choice for medium tabular data |
| [TabReD, Rubachev et al. (2024)](https://arxiv.org/abs/2406.19380) | [tabred_2024.pdf](papers/tabred_2024.pdf) | Temporal splits and engineered tables |
| [TabArena, Erickson et al. (2025)](https://arxiv.org/abs/2506.16791) | [tabarena_2025.pdf](papers/tabarena_2025.pdf) | Fair tabular benchmarking |
| [Credit-risk foundation-model benchmark, Baesens et al. (2026)](https://arxiv.org/abs/2605.18147) | [credit_foundation_2026.pdf](papers/credit_foundation_2026.pdf) | Closest task-specific model study; reports KS |
| [TabM, Gorishniy et al. (2025)](https://arxiv.org/abs/2410.24210) | [tabm_2025.pdf](papers/tabm_2025.pdf) | Neural baseline for a later iteration |
| [TabICL, Qu et al. (2025)](https://arxiv.org/abs/2502.05564) | [tabicl_2025.pdf](papers/tabicl_2025.pdf) | Strongest mean KS in the credit-risk benchmark above |
| [Deep Feature Synthesis, Kanter and Veeramachaneni (2015)](https://people.csail.mit.edu/kalyan/dsm/) | [deep_feature_synthesis_2015.pdf](papers/deep_feature_synthesis_2015.pdf) | Relational feature aggregation; apply decision-time cutoffs here |
| [TabPFN v2, Hollmann et al. (2025)](https://www.nature.com/articles/s41586-024-08328-6) | [tabpfn_v2_2025.pdf](papers/tabpfn_v2_2025.pdf) | Foundation-model origin and scale limits |
| [TabPFN-2.5 (2025)](https://arxiv.org/abs/2511.08667) | [tabpfn_2_5_2025.pdf](papers/tabpfn_2_5_2025.pdf) | Intermediate scale results |
| [TabPFN-3 (2026)](https://arxiv.org/abs/2605.13986) | [tabpfn_3_2026.pdf](papers/tabpfn_3_2026.pdf) | Larger-table foundation-model trajectory |
| [TabPFN-3.5 (2026)](https://arxiv.org/abs/2609.17895) | [tabpfn_3_5_2026.pdf](papers/tabpfn_3_5_2026.pdf) | Latest family report with temporal and grouped benchmarks |
| [LimiX (2025)](https://arxiv.org/abs/2509.03505) | [limix_2025.pdf](papers/limix_2025.pdf) | Original LimiX family report |
| [LimiX-2 (2026)](https://arxiv.org/abs/2609.17488) | [limix_2_2026.pdf](papers/limix_2_2026.pdf) | Recent model prioritized for future comparison |
| [LimiX-2M (2026)](https://arxiv.org/abs/2606.04485) | [limix_2m_2026.pdf](papers/limix_2m_2026.pdf) | Smaller model in the same family |

These 17 files were downloaded and opened by a PDF parser. Their reported results are not LOGIKA validation scores. The current contest has no paper in this library reporting its official split and KS metric.

## PDF not obtainable here

**Please supply:** Büşra Kurun and Okan Bursa, *Explainable AI-Based Credit Default Risk Modeling on Multi-Table Data*, SIU 2026, DOI [10.1109/SIU71813.2026.11636508](https://doi.org/10.1109/SIU71813.2026.11636508). Its [IEEE record](https://ieeexplore.ieee.org/abstract/document/11636508/) was accessible for bibliographic details and abstract, but the publisher PDF endpoint timed out during download. Put an authorized copy at `research/papers/credit_multitable_ieee_2026.pdf`; rerun the downloader to add its digest to `download_status.json`.

**Please supply:** Ligang Zhou, Hamido Fujita, Hao Ding, and Rui Ma, *Credit risk modeling on data with two timestamps in peer-to-peer lending by gradient boosting*, Applied Soft Computing 110 (2021), DOI [10.1016/j.asoc.2021.107672](https://doi.org/10.1016/j.asoc.2021.107672). The [publisher record](https://www.sciencedirect.com/science/article/pii/S1568494621005937) provides the abstract but offers the PDF through institutional access or purchase; the PDF endpoint did not return a usable file here. Put an authorized copy at `research/papers/credit_two_timestamps_2021.pdf` and rerun the downloader to record it.
