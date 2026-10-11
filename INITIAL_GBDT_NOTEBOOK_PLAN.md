# Handoff plan: first LightGBM, XGBoost, and CatBoost notebooks

This plan is for the agent that will build the first three modeling notebooks. It is a plan only: no notebook or model is created by this document. The three notebooks must be independently runnable from the committee CSV files and comparable on identical validation rows.

## Read first

1. Read [notebook-skill/SKILL.md](notebook-skill/SKILL.md), then its [introduction](notebook-skill/references/introduction.md), [initialization](notebook-skill/references/initialization.md), and [cell-style](notebook-skill/references/cell-style.md) references. Use [build_template.py](notebook-skill/build_template.py) as the builder-script pattern. Treat builder scripts as the editable source and regenerate `.ipynb` files after edits. Do not edit generated notebooks directly.
2. Read [overview.md](overview.md), [data.md](data.md), [rules.md](rules.md), and [DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). The selected loan's `decision_datetime` is the information cutoff; the competition score is the absolute two-sample KS statistic on risk scores.
3. Read [EDA_RESEARCH_REPORT.md](EDA_RESEARCH_REPORT.md), [eda/02_preprocess.py](eda/02_preprocess.py), and the [paper index](research/README.md). The paper PDFs are background only. Never use them as external data in features.

## Scope and fixed facts

- Task: binary risk ranking for reaching 90+ end-of-day DPD within 12 calendar months of origination. `target=1` is the positive class. The submission column is `default_probability`, bounded to finite values in `[0,1]`.
- Supplied data: 70,000 train applications, 30,000 test applications, 100,000 distinct customers, 9,507 positives (13.5814%). Train and test customers do not overlap. Selected decisions span January through December 2025.
- Existing local preprocessing produces 63 features per application. `term_months` is constant at 12, leaving 62 candidate model features after dropping it. Four raw categoricals are `loan_purpose`, `region`, `employment_type`, and `housing_status`. The generated `outputs/02_preprocess/*.parquet` files are **local reference artifacts**, not valid Kaggle runtime dependencies.
- Data table sizes and quality checks are in `outputs/01_audit/`. Historical repayment DPD, e-commerce recency/spending, bureau score, and affordability have strong descriptive associations. Those raw-feature KS values are not model scores.
- This first iteration compares only LightGBM, XGBoost, and CatBoost. No Optuna sweep, stacking, TabM, TabPFN, or LimiX inference belongs in these notebooks.

## Deliverables to create later

| Artifact | Purpose |
|---|---|
| `notebooks/build_common.py` | One source for shared builder helpers, data pipeline cell strings, fold setup, metrics, common charts, and submission checks. This module is used at **build time**. Generated notebooks must run without importing it. |
| `notebooks/build_01_lightgbm.py` and `notebooks/01_lightgbm.ipynb` | First CPU baseline. |
| `notebooks/build_02_xgboost.py` and `notebooks/02_xgboost.ipynb` | Same data and folds with XGBoost. |
| `notebooks/build_03_catboost.py` and `notebooks/03_catboost.ipynb` | Same data and folds with CatBoost. |
| `requirements-notebooks.txt` | Tested package versions for local regeneration and notebook execution. |

Put all code needed to load CSVs, build features, fit the model, score it, and write a submission **inside each generated notebook**. The builder helper can share source text across notebooks, but a Kaggle notebook must not rely on `eda/`, `notebooks/build_common.py`, the ignored `outputs/` directory, a prebuilt feature file, or an external dataset at execution time. Save generated artifacts under a model-specific directory in `/kaggle/working` or `outputs/notebooks/` locally. Use neutral working notebook names until the participant number and team name are known; rename the committee submission notebook to the required `ParticipantNumber_TeamName_Notebook` pattern when those details are supplied.

## Shared notebook structure

Follow `notebook-skill/` rather than inventing a cell format. Each notebook begins with the three banner cells: project, team information, and table of contents. Section 1 is Introduction with Overview, Aim, Metric, Dataset, and Approach. Section 2 is Initialization with Environment Setup, Import Libraries, Seed Everything, Settings, and Load Dataset in that order. Later sections should be Data Audit, Point-in-Time Features, Validation Design, Model Training, Evaluation, Submission, and Reproduction Notes. Add a brief bridging paragraph before code and an `#### Insights` block after each analysis subsection. Keep all imports in the Libraries cell and all paths, seeds, and parameters in Settings. The `!nvidia-smi` probe in the skill's environment section is informational; CPU execution remains the default. Use `%pip`, not `!pip`, for package installation and record actual versions and runtime rather than placeholders.

### Loading and feature construction

1. Resolve the dataset directory by finding exactly one directory containing `train_applications.csv` under `/kaggle/input` on Kaggle; use this repository's `data/` locally. Fail with a clear message if none or more than one match. Read only committee-supplied CSVs. Loading `sample_submission.csv` is for schema and ID validation, not features.
2. Port the logic of `eda/02_preprocess.py` to parameterized code emitted by `build_common.py`. DuckDB is suitable for the 4.36 million e-commerce item rows, but the notebook must install/import its own dependency. Produce the same feature names and column order for train and test. Locally compare a sample and aggregate summaries against the existing Parquet artifacts before accepting a port.
3. Make every auxiliary table one row per selected application before joining. Previous applications require an earlier `decision_datetime`; repayment installments must belong to an eligible prior approved loan and have due/payment information before the selected decision; account months must be complete; bureau reports require `available_datetime <= decision_datetime` and should include reports attached to previous applications via `customer_id`; transactions require `transaction_datetime < decision_datetime`. Deduplicate `order_id` before summing `order_total`. Cancelled basket values are excluded from spending.
4. Keep `application_id`, `customer_id`, raw `decision_datetime`, split, and `target` as metadata. Never feed opaque IDs, raw selected timestamps, the label, or a post-decision value to a model. `decision_month` is an existing numeric derived feature; keep it in the initial comparison and later test its stability. Drop constant `term_months` at model input. Preserve null continuous features and explicit zero event counts. Do not fit a scaler, imputer, one-hot encoder, or target encoder to all labeled rows before cross-validation.
5. Assert 70,000 distinct labeled IDs, 30,000 distinct test IDs, one feature row per ID, aligned train/test feature columns, binary labels, finite nonmissing model inputs where required, and exact equality between test IDs and sample submission IDs.

### Common validation and metric contract

- Sort labeled rows by `application_id`. Generate **the same** five stratified shuffled folds with seed 42 in each notebook and save `folds.csv` (`application_id`, `fold`) plus a SHA-256 digest. All three notebooks must report the same digest. With one selected application per customer and disjoint supplied customers, the application-level folds are also customer-disjoint; still assert this rather than relying on the current dataset shape.
- Keep the outer validation fold untouched by hyperparameter selection and early stopping. Within each outer training partition, make a deterministic stratified inner split for early stopping. Obtain its best iteration, then refit with that fixed iteration count on the whole outer training partition before predicting the outer validation fold. Fit fold-specific encoding on outer training rows only. This preserves a clean outer KS estimate.
- Match the provided scorer: `scipy.stats.ks_2samp(scores[y == 1], scores[y == 0]).statistic`. Report per-fold KS, mean and standard deviation, and pooled out-of-fold KS. Also report ROC-AUC and average precision as diagnostics; neither replaces KS. Save `oof_<model>.csv` with ID, fold, target, and score. Never describe EDA feature-separation KS as model KS.
- Add one retrospective time stress check using January-September decisions for training and October-December for validation, with its own inner early-stop split. Label it retrospective: because outcomes mature over 12 months, this one-year dataset cannot simulate strictly prospective label availability at an October 2025 deployment date. Keep this result separate from the primary stratified-fold comparison.
- Use the official metric for model selection. Use log loss or AUC as the smooth early-stopping surrogate, applied identically in spirit across models; record the choice. Start without class weights. Any later weights or thresholding require a separately documented experiment. KS does not need a decision threshold or probability calibration for leaderboard ranking.

### First model configurations

Use a single documented starter configuration per library, CPU by default, then stop. Values below are starting points, not claims of optimality. Keep the same feature information and outer folds across models.

| Notebook | Initial configuration | Category and missing-value handling |
|---|---|---|
| LightGBM | Binary objective, approximately 2,000 maximum boosting rounds, learning rate 0.03, 31 leaves, 100 minimum child samples, modest L2 regularization, seed 42, inner early stopping around 100 rounds. Use the library's callback API. | Pass the four categorical columns as pandas categories with fold-safe category vocabularies; retain numeric nulls. |
| XGBoost | `binary:logistic`, histogram tree method, approximately 2,000 rounds, learning rate 0.03, depth about 5, modest row/column subsampling and L2 regularization, seed 42, inner early stopping around 100 rounds. | For the initial comparison, use a fold-fitted `OneHotEncoder(handle_unknown='ignore')` for the four low-cardinality strings; keep missing numeric values. Fit the encoder again on full labeled data for final test prediction. |
| CatBoost | `Logloss`, approximately 2,000 iterations, learning rate 0.03, depth about 6, L2 regularization, seed 42, inner early stopping around 100 rounds. | Pass the four categorical feature names to `Pool`/`CatBoostClassifier`; use a fixed missing token for categorical nulls and retain numeric nulls. Disable incidental CatBoost file writes or direct them to the model output directory. |

The future agent should check the then-installed APIs against the [LightGBM classifier and early-stopping docs](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html), [XGBoost estimator docs](https://xgboost.readthedocs.io/en/stable/python/python_api.html), and [CatBoost classifier docs](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier). Library interfaces and package versions may change; pin the versions actually used in `requirements-notebooks.txt` and display them in each notebook.

### Evaluation and output cells

Each notebook should plot its fold KS values, positive/negative score empirical CDFs with the maximum KS gap, ROC and precision-recall curves, a risk-decile table/chart, and model-specific feature importance. Label plots with the split and sample count. Do not repeat the large EDA gallery. Briefly interpret whether a model's strongest features match the existing audit, and show performance for customers with and without previous/repayment/account history. Do not claim a feature is causal or leak-free from importance alone.

After cross-validation, select the median best iteration from the five clean outer folds, fit one final model on **all 70,000 labeled rows** with that fixed round count and fold-independent preprocessing fitted on labeled rows, then predict all 30,000 test rows. Save `submission_<model>.csv` with exactly `application_id,default_probability`, plus `metrics_<model>.json`, `folds.csv`, OOF scores, environment versions, final feature names, and a serialized model/encoder if practical. Assert that the submission has every sample-submission ID exactly once, no extras, finite values in `[0,1]`, and a score range wider than a constant. Follow [rules.md](rules.md) for the committee notebook filename and execution requirements.

## Build order and acceptance checks

1. Implement the shared builder cells and generate the LightGBM notebook first. Confirm local feature parity with `eda/02_preprocess.py` and a clean run through submission creation. Use the skill's builder workflow and check the printed `Wrote ... cells -> ...` line.
2. Generate XGBoost and CatBoost from the same shared data/fold/metric cells, changing only their model-specific preprocessing, fit, importance, and serialization cells. Confirm the fold digest and feature-family inventory agree across all three.
3. Open each generated notebook by parsing its JSON to confirm valid notebook format, expected cell order, no stale outputs, and no runtime import of the builder or local EDA files. Execute each independently in the intended environment when the notebook-build task authorizes modeling; record real fold KS, runtime, package versions, and output paths. Inspect the submission file and charts through their real surfaces.
4. Hand off a short comparison table of the three models on identical folds, with mean KS, pooled OOF KS, temporal-stress KS, ROC-AUC, average precision, training time, and final test-prediction time. Any stronger model or blend belongs to a later iteration.

The current request ends at this handoff plan and the paper library. No notebook training or modeling is part of this turn.
