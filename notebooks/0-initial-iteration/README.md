# LOGIKA UI 2026: Three reproducible GBDT notebooks

## Notebooks

| File | Model | Input data | Final Kaggle CSV |
|---|---|---|---|
| `01_lightgbm.ipynb` | LightGBM | All 8 committee tables + sample submission | `/kaggle/working/submission_lightgbm.csv` |
| `02_xgboost.ipynb` | XGBoost with fold-specific one-hot categories | Same | `/kaggle/working/submission_xgboost.csv` |
| `03_catboost.ipynb` | CatBoost with native categorical features | Same | `/kaggle/working/submission_catboost.csv` |

All three notebooks are standalone. They **do not** import the included builder code, the `eda/` directory, precomputed features or any external training dataset.

## Kaggle usage

1. Open [Kaggle Code](https://www.kaggle.com/code), choose **New Notebook**, and import one `.ipynb` using **File > Import Notebook** or upload it through the notebook interface.
2. Click **Add Input** and attach the LOGIKA UI 2026 competition dataset. The notebook first checks `/kaggle/input/competitions/data-science-competition-logika-ui-2026`, then discovers a unique folder with `train_applications.csv` under `/kaggle/input`, so it works with Kaggle's alternate mounted directory layout.
3. In **Notebook Settings**, set **Accelerator: None (CPU)**; enable **Internet** to install the pinned DuckDB/model libraries (if access is allowed for the competition).
4. Run **Run All**. Each notebook loads the original nine competition CSVs, creates all features from SQL, validates 5 stratified folds, runs the temporal stress test, plots results and trains one full-data model.
5. Click **Save Version / Save & Run All**. Download the model's `submission_*.csv` from the **Output** section and submit the CSV to the competition. The model-specific `/kaggle/working/<model>/` folder also contains the OOF scores, metrics, fold manifest, plots and model artifact.
6. Before submitting the designated notebook to the committee, rename it to `ParticipantNumber_TeamName_Notebook.ipynb` with your registered number and team name.

**Note:** Kaggle's `Add Input` may require joining/accepting the competition rules first. Only use the competition-organizer-provided files.

## Model protocol

- One identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across models, with a SHA-256 fold-manifest digest.
- Original audited DuckDB point-in-time SQL, reproduced inside each notebook, creates 63 features; drop the constant `term_months` to train with the same 62 features.
- Inner stratified holdout selects the number of boosting rounds by log loss. The selected number of rounds is refit on the entire outer training partition before predicting the untouched outer validation partition.
- Official KS uses `scipy.stats.ks_2samp` on out-of-fold scores from positive and negative targets. Also report AUC and average precision.
- Retrospective January–September -> October–December 2025 stress test is separate from the five-fold comparison.
- Full training on 70,000 rows with median selected iteration count, then validated 30,000-row submission in the exact `application_id,default_probability` format.

## Local validation performed

- All three `.ipynb` files passed `nbformat.validate`, and every ordinary code cell was syntactically compiled.
- All three model-specific training/prediction adapters passed a local synthetic-data training test, including early stopping, categorical handling, missing values and predicted-probability production, using LightGBM 4.6.0, XGBoost 3.1.3 and CatBoost 1.2.8.
- **Not executed on the original committee CSVs:** The supplied archive contains precomputed reference Parquet files but not the nine raw competition CSVs, and this local environment could not install DuckDB due restricted network access. Consequently no real model KS, feature-parity result or genuine test submission has been claimed. Running on Kaggle with the supplied competition CSVs is still required.

## Source files

Notebook builders live under `notebooks/build_common.py` and `notebooks/build_0*.py`. They embed the SQL features from `eda/02_preprocess.py` **at build time**. To regenerate from the packaged source, run the three builders from the repository root with `python notebooks/build_01_lightgbm.py`, etc. The generated notebook **does not** require these files at runtime.
