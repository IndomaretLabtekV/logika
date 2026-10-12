# LOGIKA UI 2026 credit-risk repository guide

This repository supports the LOGIKA UI 2026 credit-risk competition. It contains the supplied multi-table data, point-in-time EDA and feature construction, three executed gradient-boosting experiments, research references, and an extended behavioral-risk exploration. This guide is an index: follow the linked source of truth for details instead of assuming every script or report describes the latest state.

## Start here

1. Read [`overview.md`](overview.md) for the prediction target, official KS metric, and submission contract; [`rules.md`](rules.md) for competition constraints.
2. Read [`data.md`](data.md) and the actual schema/key definitions in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) before joins or feature changes. `decision_datetime` is the selected application information cutoff. Only committee data may be used as model input.
3. Read [`RISK_EXPLORATION_REPORT.md`](RISK_EXPLORATION_REPORT.md) for the expanded EDA, financial interpretation, evidence-backed feature priorities, and current limits. Its companion [script index](eda/risk_exploration/README.md) explains how to reproduce that analysis.
4. Read [`notebooks/0-initial-iteration/README.md`](notebooks/0-initial-iteration/README.md) for the model comparison. Open the `*-result.ipynb` files when you need the executed outputs, not just the notebook source.
5. Read [`EDA_RESEARCH_REPORT.md`](EDA_RESEARCH_REPORT.md) for the earlier baseline EDA and literature notes. It predates the model runs and expanded risk exploration; where its historical statements conflict with current result notebooks or the newer report, use the dated artifacts described below.

## Task and data at a glance

The label is 1 when the selected originated loan reaches at least 90 end-of-day days past due within 12 calendar months; otherwise it is 0. The data has 70,000 labeled applications, 30,000 unlabeled test applications, and one selected application per customer. There are 9,507 positive labels (13.5814%). The official score is the two-sample KS statistic on predicted scores, so exploratory feature KS and model KS have different meanings. Neither KS nor this target establishes calibrated probability or total credit loss.

The CSVs are local competition inputs. Do not infer schemas or joins from names; consult the dictionary. Opaque identifiers belong in join keys and audit manifests, never as numerical predictors. Account months are complete calendar-month records, bureau observations have separate report and availability timestamps, repayment rows are historical installment snapshots, and e-commerce rows are order items whose order total repeats across items. The newer EDA documents what can and cannot safely be inferred from those shapes.

## Repository map: what to read for each meaning

| Path | Meaning and when to open it |
|---|---|
| [`overview.md`](overview.md) | Business objective, target, metric, official scoring definition, submission schema and competition timeline. |
| [`rules.md`](rules.md) | Permitted data, leakage constraints, notebook delivery, scoring weights and conduct rules. |
| [`data.md`](data.md) | Dataset file inventory, high-level temporal guidance and suggested feature families. This guide explicitly defers exact columns and keys to the dictionary. |
| [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) | Authoritative column types, nullability, definitions, opaque keys, timestamps and order-item semantics. Read before changing any join or cutoff. |
| [`data/train_applications.csv`](data/train_applications.csv), [`data/test_applications.csv`](data/test_applications.csv) | The selected loan rows. Only train contains the target. `decision_datetime` is the point-in-time feature cutoff. |
| [`data/customer_profiles.csv`](data/customer_profiles.csv) | Application-linked customer demographics, declared income, employment, dependents, housing and fictional region. |
| [`data/previous_applications.csv`](data/previous_applications.csv) | Prior requests and decisions. Earlier decision time is required; approved does not always prove the loan remains active. |
| [`data/repayment_history.csv`](data/repayment_history.csv) | Installment snapshots with amounts due/paid and current/historical DPD fields. It is not a timestamped daily payment event log. |
| [`data/account_behavior.csv`](data/account_behavior.csv) | Complete monthly balances, inflows/outflows, income-like credits and low-balance days. The month must end before the selected cutoff. |
| [`data/bureau_observations.csv`](data/bureau_observations.csv) | Provider reports, score, balances and inquiries. Enforce both `available_datetime` and `report_datetime` cutoffs. Provider balances can overlap. |
| [`data/ecommerce_transactions.csv`](data/ecommerce_transactions.csv) | Order-item rows with repeated order totals, status, payment method and category. Deduplicate order totals and exclude cancellations from completed spending. |
| [`data/sample_submission.csv`](data/sample_submission.csv) | Required application ID and prediction column names for submission. |
| [`eda/01_audit.py`](eda/01_audit.py) | Original input inventory, key/split checks, missingness, coverage, cohort and cutoff audit. |
| [`eda/02_preprocess.py`](eda/02_preprocess.py) | Original 63-feature application-grain DuckDB feature pipeline. Use as the initial baseline definition, not as a substitute for the more precise calendar-window extension. |
| [`eda/03_explore.py`](eda/03_explore.py) | Original univariate rates, raw-feature separation, missingness, train/test marginal drift and score/affordability cross-tab. |
| [`eda/common.py`](eda/common.py) | Shared original EDA DuckDB connection, CSV views, and train/test application union. |
| [`outputs/01_audit/`](outputs/01_audit/) | Committed original audit outputs: inventory, quality checks, missingness, coverage, monthly cohorts and audit figure. |
| [`outputs/02_preprocess/`](outputs/02_preprocess/) | Local, ignored Parquet feature snapshots and summary from original preprocessing. Not a required input to the Kaggle notebooks. |
| [`outputs/03_explore/`](outputs/03_explore/) | Local, ignored original EDA tables and charts. Recreate by running `eda/03_explore.py`. |
| [`RISK_EXPLORATION_REPORT.md`](RISK_EXPLORATION_REPORT.md) | Current extended EDA: financial hypotheses, timeline handling, cross-table interactions, relationship graphs, candidate feature policy and required future data. |
| [`eda/risk_exploration/sql/01_timelines.sql`](eda/risk_exploration/sql/01_timelines.sql) | Application-aligned account lags, prior application behavior, mature repayment denominators and within-provider bureau changes. |
| [`eda/risk_exploration/sql/02_commerce.sql`](eda/risk_exploration/sql/02_commerce.sql) | Deduplicated commerce, purchase cadence, category graph, matched cash/commerce months and cross-source ratios. |
| [`eda/risk_exploration/sql/03_contracts.sql`](eda/risk_exploration/sql/03_contracts.sql) | Source key/ownership/reconciliation checks, event eligibility counts and relationship-edge inventory. |
| [`eda/risk_exploration/diagnostics.sql`](eda/risk_exploration/diagnostics.sql) | Descriptive label slices, support-aware interaction cells, seasonality, account transitions and aggregate graph diagnostics. |
| [`eda/risk_exploration/build.py`](eda/risk_exploration/build.py), [`diagnose.py`](eda/risk_exploration/diagnose.py), [`plot.py`](eda/risk_exploration/plot.py) | The ordered extended EDA pipeline: build, analyze, render figures. |
| [`eda/risk_exploration/validate.py`](eda/risk_exploration/validate.py) | Hand-calculable SQL fixtures and source-to-output invariants for cutoffs, missingness, repeated order totals and target independence. |
| [`outputs/04_risk_exploration/`](outputs/04_risk_exploration/) | Committed extended EDA tables, figures, code/input hash manifest and fixture-validation result. Four large Parquet snapshots are generated locally and ignored by Git. |
| [`notebooks/0-initial-iteration/01_lightgbm.ipynb`](notebooks/0-initial-iteration/01_lightgbm.ipynb), [`02_xgboost.ipynb`](notebooks/0-initial-iteration/02_xgboost.ipynb), [`03_catboost.ipynb`](notebooks/0-initial-iteration/03_catboost.ipynb) | Standalone source notebooks that embed the data load, point-in-time feature SQL, folds, training and submission creation. |
| [`notebooks/0-initial-iteration/lightgbm-result.ipynb`](notebooks/0-initial-iteration/lightgbm-result.ipynb), [`xgboost-result.ipynb`](notebooks/0-initial-iteration/xgboost-result.ipynb), [`catboost-result.ipynb`](notebooks/0-initial-iteration/catboost-result.ipynb) | Saved Kaggle executions with actual train/test load, five-fold OOF metrics, retrospective time stress, feature importance, final fit and generated submission evidence. These are the evidence for completed experiments. |
| [`notebooks/0-initial-iteration/notebooks/build_common.py`](notebooks/0-initial-iteration/notebooks/build_common.py), [`build_01_lightgbm.py`](notebooks/0-initial-iteration/notebooks/build_01_lightgbm.py), [`build_02_xgboost.py`](notebooks/0-initial-iteration/notebooks/build_02_xgboost.py), [`build_03_catboost.py`](notebooks/0-initial-iteration/notebooks/build_03_catboost.py) | Editable source for regenerating the initial standalone notebooks. Regeneration replaces notebook content/outputs; preserve result notebooks if the saved run evidence matters. |
| [`notebooks/0-initial-iteration/eda/02_preprocess.py`](notebooks/0-initial-iteration/eda/02_preprocess.py) | Packaged copy of the original SQL feature pipeline used in the notebook generation process. |
| [`notebooks/0-initial-iteration/requirements-notebooks.txt`](notebooks/0-initial-iteration/requirements-notebooks.txt) | Pinned or bounded package requirements for model notebook generation and execution. |
| [`INITIAL_GBDT_NOTEBOOK_PLAN.md`](INITIAL_GBDT_NOTEBOOK_PLAN.md) | Historical first-iteration handoff plan. The three models have since been implemented and run; use actual notebooks and this guide for current status. |
| [`notebook-skill/SKILL.md`](notebook-skill/SKILL.md) and [`references/`](notebook-skill/references/) | Notebook authoring, structure, and cell-style conventions used for standalone notebooks. |
| [`notebook-skill/template.ipynb`](notebook-skill/template.ipynb), [`build_template.py`](notebook-skill/build_template.py) | Generic notebook starter and generator, not a model experiment. |
| [`EDA_RESEARCH_REPORT.md`](EDA_RESEARCH_REPORT.md) | Earlier first-pass EDA, feature notes and tabular-model literature. Some end-state claims were true before the initial-iteration Kaggle runs; see the results table below. |
| [`research/README.md`](research/README.md), [`papers.json`](research/papers.json), [`download_status.json`](research/download_status.json) | Index and provenance for downloaded research papers. They are methodology references, not competition feature data. |
| [`research/download_papers.py`](research/download_papers.py), [`research/papers/`](research/papers/) | Paper download/verification utility and available local PDFs. No paper data is joined to application records. |
| [`outputs/initial-iteration/`](outputs/initial-iteration/) | Local Kaggle result ZIPs with OOF predictions, fold manifests, metrics, fitted models and submission CSVs. This path is ignored by Git; the tracked result notebooks preserve the primary execution record. |
| [`requirements-eda.txt`](requirements-eda.txt) | Local EDA runtime pins. The notebooks use their separate requirements file and Kaggle environment. |

## Experiments actually run

The three saved `*-result.ipynb` notebooks report executions against the competition-mounted raw CSVs. They used the same five-fold stratified split (fold manifest SHA-256 `a233781ed128e2fde13e5ae58ee78b049b602872558fefc07c044c892dafc4f4`), 70,000 train rows, 30,000 test rows and the original 62 model-input features. The saved notebook JSON contains fold metrics and final JSON summaries. Exact package versions, per-fold scores, feature importance and submission previews are inside each result notebook.

| Model | Package version | Mean fold KS (std) | Pooled OOF KS | Retrospective Jan-Sep to Oct-Dec KS | Final notebook fit wrote |
|---|---:|---:|---:|---:|---|
| LightGBM | 4.6.0 | 0.48257 (0.01459) | 0.48172 | 0.47469 | `submission_lightgbm.csv` |
| XGBoost | 3.1.3 | 0.48237 (0.01293) | 0.47903 | 0.47783 | `submission_xgboost.csv` |
| CatBoost | 1.2.8 | 0.48184 (0.01464) | 0.47899 | 0.47953 | `submission_catboost.csv` |

These are notebook-local validation results, not Kaggle public/private leaderboard scores. Model scores are close; do not claim one is decisively superior from this comparison. The January-September/October-December check is retrospective because a 12-month outcome cannot be fully mature at an October 2025 decision cutoff. The newer extended EDA is a separate analysis and has not yet been added to or validated in these model notebooks.

The Kaggle run also wrote model artifacts, OOF scores, fold records, plots and the test submissions. Local ZIP bundles are present under `outputs/initial-iteration/` when available. They are ignored by Git and may not exist in a fresh clone; the three tracked result notebooks are the durable record of executed metrics and workflow. No successful upload or official public/private score is recorded in this repository.

## Extended EDA: strongest findings and limits

The extended work adds 124 candidate fields for inspection, not 124 approved model inputs. Highlights from the labeled data:

- Historical share of installments ever reaching 30 DPD, restricted to dues at least 30 days old, has raw-feature KS 0.435; max historical DPD has 0.371. This corrects for obvious installment exposure time, but is not an OOF model score or a reconstructed monthly delinquency transition.
- Spending at 10-30% of declared quarterly income is associated with 11.78% observed risk among score-650+ customers, versus 5.24% below 10%. Among score-650+ customers with a cash surplus, low spend burden has 5.03% observed risk versus 16.30% for spend burden above 30%. These are descriptive cross-tabs, not causal effects.
- The data has actual customer-to-loan and customer-to-category edges, but no household, employer, device or coordinate data. Category co-occurrence lift is nearly one, so a complex social or category graph currently lacks evidence.
- Zero-filling commerce before the first observed order changes apparent seasonal ratios. The extended report compares naive and post-first-order complete-month support and treats recurrence/cycle features as exploratory.
- Account balance flows reconcile exactly in the observed data; transaction availability, provider debt overlap, rejected applicant outcomes, verified salary, and complete cure/relapse histories remain limitations.

Use the report and CSV tables for denominators, subgroup support, missingness, train/test drift, source contract checks and seasonal comparisons. A raw-feature KS measures class separation for one feature over its observed support; it is not a prediction score or a causal effect. Experiment ideas labelled "plan" in the report have not been run.

## Reproducing analyses

From the repository root, with Python and dependencies installed:

```bash
python eda/01_audit.py
python eda/02_preprocess.py
python eda/03_explore.py
python eda/risk_exploration/build.py
python eda/risk_exploration/diagnose.py
python eda/risk_exploration/plot.py
python eda/risk_exploration/validate.py
```

The extended scripts also support `uv run` and document their dependencies. Each analysis writes beneath `outputs/`. The original GBDT notebooks are intended to execute independently in Kaggle with the committee input attached; use the individual notebook README and notebook settings for that workflow. Do not assume a local Parquet artifact or `outputs/initial-iteration/` ZIP is present in a clone.

## Working safely in this repository

- Keep `application_id`, `customer_id`, labels, split markers, and raw cutoffs in metadata; never train on opaque IDs or future outcomes.
- Before changing a join, window, or snapshot rule, read the dictionary definition and add evidence through the existing boundary-validation workflow. Do not treat missing history as good repayment or zero exposure.
- Keep source families separable so new model comparisons can measure incremental value on identical validation rows. The new candidate panel is not yet a model-ready train/test split; exclude metadata and fit learned transforms only within training folds.
- Distinguish four evidence levels in notes: source/data-quality checks, descriptive EDA, cross-validated model metrics, and official leaderboard results. Only the saved model notebooks currently establish the third level; none establish the fourth.
- Research PDFs explain methods and must not become feature data. Competition rules prohibit external data.
