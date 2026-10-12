# LOGIKA UI 2026 credit risk: EDA and research brief

Prepared 11 October 2026 as the first-pass EDA and research brief. It uses only the supplied competition CSVs for data analysis. Three predictive models were run later in the saved Kaggle result notebooks under `notebooks/0-initial-iteration/`; see the repository [agent guide](README.md) and [extended risk EDA](RISK_EXPLORATION_REPORT.md) for the current state.

## Problem and data

The target is whether the selected loan reaches at least 90 end-of-day days past due within 12 calendar months of origination. The supplied [overview](overview.md) specifies two-sample Kolmogorov-Smirnov (KS) on predicted scores as the competition metric. Scores must be in `[0, 1]`, but KS evaluates separation, not probability calibration. The [data dictionary](data/DATA_DICTIONARY.md) makes each selected application's `decision_datetime` the feature cutoff.

| Item | Local result |
|---|---:|
| Labeled applications | 70,000 |
| Unlabeled test applications | 30,000 |
| Positive labels | 9,507 (13.5814%) |
| Negative labels | 60,493 |
| Unique customers across train and test | 100,000; no overlap |
| Decision range | 1 January to 28 December 2025 |
| Historical source rows, excluding application/profile tables | 7,805,123 |
| Unique e-commerce orders from 4,359,817 item rows | 2,986,855 |
| Prepared features per application | 63 |

All declared primary keys are unique. The supplied history rows inspected in the audit are before the selected decision: zero bureau reports with later availability, zero later repayment due dates or recorded payments, zero incomplete account months, zero later transactions, and zero later previous applications. All e-commerce orders have a consistent total equal to the sum of their item values. This supports the point-in-time construction, though the competition's own generation logic is not independently available.

Customer coverage is 100% for profiles, bureau, and e-commerce; 70% for previous applications and account behavior; and 66.16% of labeled customers / 66.15% of test customers for repayment history. Coverage is nearly identical across train and test. A missing history is a separate condition from a recorded zero. Bureau utilization is missing for 51.99% of labeled loans, largely because utilization is undefined without a positive revolving limit; replacing it with zero would change its meaning. Among prepared features, repayment measurements are missing for 33.84% and account measurements for 30.00% of train rows.

![Dataset size, target trend, coverage and missingness](outputs/01_audit/audit_overview.png)

## Observed relationships

These are descriptive rates within the labeled data. They are neither cross-validated estimates nor causal effects. KS values below compare the *raw feature distributions by label*, not prediction scores, so they are not competition scores.

| Feature comparison | Observed 90-DPD rate | Interpretation |
|---|---:|---|
| Mean bureau score, lowest vs highest decile | 29.83% vs 3.77% | Strong and nearly monotone score gradient |
| Historical maximum DPD, at most 30 vs above 120 | 6.26% vs 45.25% | Prior repayment behavior is a major candidate signal |
| Completed e-commerce spending in prior 90 days, lowest vs highest nonmissing decile | 4.25% vs 31.83% | Large association; test interactions and stability before trusting it |
| Proposed installment / declared monthly income, lowest vs highest decile | 10.56% vs 17.23% | Plausible affordability gradient |
| No prior application history vs any | 11.72% vs 14.38% | Missing history itself may carry signal |

Historical maximum DPD has raw-feature KS 0.371 among customers with repayment records. E-commerce recency has 0.279; 90-day e-commerce spending has 0.272; mean bureau score has 0.253; annual interest rate has 0.230. The high association for interest rate may reflect the lender's earlier underwriting, so it should be kept only if known at prediction time as the dictionary states. Loan purpose, age, and broad region have much weaker univariate rate differences. `term_months` is constant at 12 and should be dropped at model input.

The spend signal warrants an ablation. Item rows repeat `order_total`; the preprocessing deduplicates by order before computing spending. Its association may reflect other customer traits or a synthetic data-generating rule. It does not prove that increasing spending causes default. The [numeric rate plots](outputs/03_explore/target_rate_numeric_bins.png) and [score versus affordability heatmap](outputs/03_explore/score_affordability_heatmap.png) show the shapes and interactions; exact counts are in `outputs/03_explore/numeric_bins.csv` and the matching heatmap CSVs.

![Raw feature class separation](outputs/03_explore/numeric_separation.png)

The labeled target rate varies by decision month from 12.01% in May to 14.73% in February. The largest measured numeric train-to-test distribution KS is only 0.0133, for bureau utilization, and source coverage matches closely. This suggests little visible split drift on these measurements, but the hidden test labels are unavailable and conditional drift cannot be checked. Time evolution matters for validation even when train and test marginal features look similar. [TabReD](https://arxiv.org/abs/2406.19380) found that model rankings can change under time-based evaluation.

## Reproducible local artifacts

Use Python 3.12 or later, then run from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-eda.txt
.venv/bin/python eda/01_audit.py
.venv/bin/python eda/02_preprocess.py
.venv/bin/python eda/03_explore.py
```

Each script prints a numeric summary and writes its files under `outputs/`. The first writes inventory, missingness, coverage, monthly cohorts, cutoff checks, and an overview chart. The second writes a 63-feature train and test Parquet pair, a split manifest, and feature missingness. The third writes numeric/class associations, bin rates, train/test drift, source-coverage rates, a two-feature rate table, and six charts. The Parquet files and charts are local generated artifacts and are ignored by Git; the scripts and this report remain in the workspace.

The prepared tables keep one row per selected application and put the ID and target outside the 63 features. `customer_id` and raw timestamps are stored separately in the split manifest, never used as numerical predictors. Continuous quantities with no observations remain null; event counts with no history are zero. No global scaler, target encoder, or imputer was fitted. If a later model needs such transforms, fit them inside each training fold. The script uses only prior application decisions, historical installment due dates and payments, complete account months, bureau reports available by the cutoff, and prior e-commerce events. Bureau reports linked to previous applications are joined by customer with the same cutoff; the latest available report per provider forms the current snapshot. It counts completed e-commerce order totals once per order and excludes cancelled basket values from spending.

## Literature and model fit

The papers below are evidence about method families, not performance predictions for LOGIKA. No published result located for this exact dataset and official KS split provides a defensible numeric performance ceiling.

| Source | Relevant evidence | What it implies here |
|---|---|---|
| [TabReD, 2024](https://arxiv.org/abs/2406.19380) | Time splits and feature-engineered industrial tables change model rankings; MLP-like models and gradient-boosted trees perform well. | Use a temporal stress test and keep boosted trees and TabM in the comparison. |
| [TabArena, 2025](https://arxiv.org/abs/2506.16791) | Boosted trees remain strong; foundation models excel on smaller tables; larger budgets and ensembles narrow gaps. | Compare on the same local split and compute budget. |
| [Baesens et al., credit-risk benchmark, 2026](https://arxiv.org/html/2605.18147v2) and [code](https://github.com/andreasgoethals/TabPFNCredit) | Fourteen probability-of-default datasets with five-fold evaluation. Mean KS was 0.3987 for TabICL and 0.3982 for CatBoost; only 22 of 406 paired method comparisons were significant after multiple-comparison correction. LimiX-2 and TabPFN-3.5 postdate this study. | This is the closest task-specific evidence. Foundation models deserve a trial, but the best credit-risk methods can be close and its mean KS is not this dataset's attainable score. |
| [Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815) | Trees are robust on medium-sized tabular problems with irrelevant features and irregular relationships. | LightGBM, XGBoost, and CatBoost are the first supervised baselines. |
| [TabM, ICLR 2025](https://arxiv.org/abs/2410.24210) and [official code](https://github.com/yandex-research/tabm) | Efficient ensembling makes a strong tabular MLP baseline. | Train only on supplied data; compare after trees with fold-safe numeric scaling and category handling. |
| [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) | The evaluated version was strongest on up to 10,000 rows and 500 features. | Original TabPFN v2 is a poor full-data size match for 70,000 training rows. |
| [TabPFN-3.5, September 2026](https://arxiv.org/abs/2609.17895) | Authors report support up to 1 million rows and 6,000 features and compare on grouped and temporal benchmarks. They also report that tuned and ensembled MLPs lead some larger temporal subsets. | A current foundation-model challenger for 70,000 rows and 63 features; evaluate locally and report memory/runtime. |
| [LimiX-2, September 2026](https://arxiv.org/abs/2609.17488) and [official code](https://github.com/limix-ldm-ai/LimiX) | The 400-million-parameter model leads the authors' TabArena/TALENT/BCCO comparisons and uses in-context prediction. | High-priority research challenger, but its aggregate Elo and classification accuracy do not establish LOGIKA KS superiority. |
| [Multi-table credit-risk study, IEEE 2026](https://ieeexplore.ieee.org/abstract/document/11636508/) | Applicant-level aggregation with LightGBM and Optuna reported AUC 0.78682 and PR-AUC 0.28015 on Home Credit. | Supports relational aggregation as an implementation pattern, not a numerical target here: different data, split and metric. |
| [Home Credit LightGBM implementation](https://github.com/sdolcecanto/home-credit-default-risk) | Public code aggregates multiple historical tables to one applicant row and maintains train/test feature parity. | Useful engineering reference, but recheck this competition's own cutoff rules. |

LimiX-2 is recent and its authors report very strong benchmark results. The [TabPFN-3.5 report](https://arxiv.org/html/2609.17895v2) was released in the same month and reports a different set of benchmark wins. Neither establishes one universally best model; their leaderboard summaries use different comparisons and mostly metrics other than this competition's KS. The LimiX-2 authors' TALENT/BCCO classification protocol ranks accuracy, which is especially different from KS for a 13.6% positive class. The local machine has an RTX 4060 Laptop GPU with 8 GB memory; published foundation-model shape support does not guarantee full-data inference fits that memory. Their pretrained weights also raise a competition-rule question: the supplied rules ban outside data and sources, so obtain an organizer interpretation before using those weights for an actual submission. This research and the local EDA did not use external data in features.

## Recommended next experiment plan, not executed

1. **Freeze one evaluation design.** Use stratified five-fold cross-validation on the 70,000 labeled customers for the primary competition estimate; customer IDs are unique and disjoint across the supplied train/test split. Score every candidate on the same folds with the official `scipy.stats.ks_2samp` statistic. Also report ROC-AUC and PR-AUC as diagnostics, not as substitutes for KS. Keep the 30,000 unlabeled test rows out of model selection.
2. **Add a retrospective time stress test.** Train on earlier 2025 decisions and validate on later decisions, for example January-September versus October-December, without fitting encoders on the later rows. A strictly prospective test of label availability is impossible with only 2025 decision dates and a 12-month target horizon: labels of early 2025 loans would not all be mature by October 2025. Call this a retrospective drift check, not a simulated live deployment.
3. **Fit the first model family.** Start with LightGBM, then XGBoost and CatBoost with equal feature sets, folds, early stopping and budget. Use probabilistic binary objectives; select by out-of-fold KS and inspect fold spread. Moderate imbalance does not justify weighting before an unweighted reference. Keep native missingness and handle the four low-cardinality categorical fields inside each fold. Drop constant `term_months` and opaque IDs.
4. **Ablate source families.** Compare application/profile only, then add bureau, previous/repayment, account, and e-commerce histories. In particular, measure whether the large e-commerce signal survives validation and whether the 30-34% missing history groups remain stable. Make any additional 30/90/180-day windows with the same per-application cutoff logic.
5. **Test one advanced challenger at a time.** Compare TabM, TabPFN-3.5, and LimiX-2 against the best tree baseline on identical held-out rows and KS. Note inference time, peak GPU memory, and any context subsampling. Use only a locally permitted checkpoint if the competition permits pretrained weights. Blend out-of-fold scores only when it improves held-out KS, not because a benchmark reports a global lead.

KS is invariant to a strictly increasing transformation of scores, so probability calibration is not expected to improve the competition metric. For actual credit decisions, calibration would be a separate validation objective. Use fold and bootstrap uncertainty for model comparisons; the raw-feature KS numbers above cannot serve as an expected model KS or accuracy ceiling.

## Boundary of this work

At the time this first-pass report was prepared, no model had been trained. Subsequent saved notebook runs report held-out OOF and retrospective scores, final training, and creation of test submission files; those artifacts are indexed in the repository [agent guide](README.md). No official public/private leaderboard score is recorded here. The report's EDA quantities come from its generated CSV/JSON files; paper results are attributed to their authors. The later expanded EDA and its candidate feature plan are in `RISK_EXPLORATION_REPORT.md`.
