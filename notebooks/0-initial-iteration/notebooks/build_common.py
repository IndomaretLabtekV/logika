"""Build-time source for standalone LOGIKA UI GBDT Kaggle notebooks.

Source SQL is extracted from the original audited EDA script during generation;
each generated notebook embeds the complete SQL and has no local code dependency.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": source}


def section(title: str, anchor: str) -> dict:
    return md(f'---\n\n# {title} <a name="{anchor}"></a>\n\n---')


def intro(cells: list, model: str):
    names = {'lightgbm':'LightGBM', 'xgboost':'XGBoost', 'catboost':'CatBoost'}
    proper = names[model]
    cells.extend([
        md(f'---\n\n# LOGIKA UI 2026 | {proper} Credit Risk Baseline\n\n'
           '*Point-in-time multi-table feature engineering and reproducible five-fold KS validation.*'),
        md('---\n\n## Team / participant information\n\n'
           'Team: **[Fill in registered team name]**  \n'
           'Participant number: **[Fill in official number]**  \n'
           'Authors: **[Fill in member names]**'),
        md('---\n\n## Table of Contents\n\n'
           '1. [Introduction](#1)\n2. [Initialization](#2)\n'
           '3. [Data Audit](#3)\n4. [Point-in-Time Features](#4)\n'
           '5. [Validation Design](#5)\n6. [Model Training](#6)\n'
           '7. [Evaluation](#7)\n8. [Submission](#8)\n9. [Reproduction Notes](#9)'),
        section('1. Introduction', '1'),
        md('## Overview\n\n*DSC LOGIKA UI 2026 asks teams to rank Indonesian consumer loans by their probability of reaching 90 or more days past due within twelve months of origination.*'),
        md(f'## Aim\n\n*Establish a transparent {proper} CPU baseline without external datasets, with the exact same engineered features, validation folds and scoring code used in the other two notebooks.*'),
        md('## Metric\n\n*The official metric is the absolute two-sample Kolmogorov-Smirnov statistic between model risk scores for targets 1 and 0. Higher scores are better; no classification threshold is required.*\n\n'
           r'$$KS = \sup_s |F_{1}(s) - F_{0}(s)|$$'),
        md('## Dataset\n\n*The competition provides 70,000 labeled and 30,000 unlabeled applications, customer profiles, prior applications, repayments, account behavior, bureau reports and e-commerce records.*\n\n'
           '**Feature sources:** loan and affordability data; customer profiles; past lending; historical repayment; account monthly summaries; bureau; e-commerce.\n\n'
           '**Target:** `target` = 1 for serious repayment difficulty (90+ DPD).'),
        md('## Approach\n\n*Original DuckDB point-in-time SQL creates 63 features; constant `term_months` is excluded, leaving 62 candidate model features. Five shuffled stratified folds are evaluated without early-stopping leakage, followed by an independent retrospective time-stress analysis and full-data final fit.*'),
    ])


def initialization(cells: list, model: str):
    proper = {'lightgbm':'LightGBM', 'xgboost':'XGBoost', 'catboost':'CatBoost'}[model]
    package = {'lightgbm':'lightgbm==4.6.0', 'xgboost':'xgboost==3.1.3', 'catboost':'catboost==1.2.8'}[model]
    cells.extend([
        section('2. Initialization','2'),
        md('## Environment Setup\n\nThe competition dataset should be attached through **Add Input**. All three notebooks are CPU-first; no GPU or internet-hosted data is required.'),
        code('!nvidia-smi || true'),
        md('The following cell installs the SQL engine and the model package inside the active Kaggle kernel. Turn on Internet in Notebook Settings if these packages are not already installed.'),
        code(f'%pip -q install "duckdb==1.4.1" "{package}"'),
        md('## Import Libraries\n\nAll libraries are imported once. Their versions are recorded with model outputs.'),
        code('''import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
'''+({'lightgbm':'import lightgbm as lgb\n', 'xgboost':'import xgboost as xgb\nimport joblib\nfrom scipy import sparse\nfrom sklearn.preprocessing import OneHotEncoder\n', 'catboost':'import catboost as cb\n'}[model])),
        md('## Seed Everything\n\nA single fixed seed controls fold assignment, inner splits and training.'),
        code('''def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(42)'''),
        md('## Settings\n\nEvery input location, output location and model setting is centralized here. The competition path supplied by the user is tried first; standard Kaggle mount paths are supported automatically.'),
        code(f'''@dataclass
class Settings:
    MODEL_NAME: str = {model!r}
    SEED: int = 42
    FOLDS: int = 5
    INNER_VALID_FRACTION: float = 0.20
    MAX_ROUNDS: int = 2000
    EARLY_STOPPING_ROUNDS: int = 100
    THREADS: int = 4
    INPUT_ROOT: Path = Path("/kaggle/input")
    PREFERRED_DATA_DIR: Path = Path("/kaggle/input/competitions/data-science-competition-logika-ui-2026")
    OUTPUT_ROOT: Path = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/notebooks")
    CATEGORICALS: tuple = ("loan_purpose", "region", "employment_type", "housing_status")
    METADATA: tuple = ("application_id", "customer_id", "split", "target", "decision_datetime")
    EXPECTED_FEATURES: int = 62
    EXPECTED_TRAIN: int = 70000
    EXPECTED_TEST: int = 30000

CFG = Settings()
CFG.OUT = CFG.OUTPUT_ROOT / CFG.MODEL_NAME
CFG.OUT.mkdir(parents=True, exist_ok=True)
print("Model:", {proper!r}, "Output directory:", CFG.OUT)'''),
        md('## Load Dataset\n\nThe selected directory must contain all committee CSVs. `DATA_DICTIONARY.md` is read for documentation only, never as an additional feature source.'),
        code('''INPUT_FILES = (
    "train_applications.csv", "test_applications.csv", "customer_profiles.csv",
    "previous_applications.csv", "repayment_history.csv", "account_behavior.csv",
    "bureau_observations.csv", "ecommerce_transactions.csv", "sample_submission.csv",
)

def resolve_data_directory():
    if CFG.PREFERRED_DATA_DIR.is_dir() and (CFG.PREFERRED_DATA_DIR / "train_applications.csv").is_file():
        candidates = [CFG.PREFERRED_DATA_DIR]
    elif CFG.INPUT_ROOT.is_dir():
        candidates = sorted(set(p.parent for p in CFG.INPUT_ROOT.rglob("train_applications.csv")))
    else:
        candidates = sorted(set(p.parent for p in Path.cwd().rglob("train_applications.csv")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one committee data directory; found {len(candidates)}: {candidates}. "
            "Attach the competition data using Add Input on Kaggle."
        )
    folder = candidates[0]
    missing = [name for name in INPUT_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required committee CSVs in {folder}: {missing}")
    return folder

CFG.DATA_DIR = resolve_data_directory()
sample = pd.read_csv(CFG.DATA_DIR / "sample_submission.csv")
assert sample.columns.tolist() == ["application_id", "default_probability"], sample.columns.tolist()
assert sample.application_id.is_unique and not sample.application_id.isna().any()
print("Competition files:", CFG.DATA_DIR)
print("Sample submission:", sample.shape)
if (CFG.DATA_DIR / "DATA_DICTIONARY.md").exists():
    print("Data dictionary located:", CFG.DATA_DIR / "DATA_DICTIONARY.md")'''),
    ])


def extract_sql() -> list[str]:
    tree = ast.parse((ROOT / 'eda' / '02_preprocess.py').read_text(encoding='utf-8'))
    result = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'execute'
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            sql = n.args[0].value.strip()
            if sql.upper().startswith('CREATE TEMP TABLE'):
                result.append(sql)
    assert len(result) == 12, len(result)
    return result


def data_pipeline(cells: list):
    cells.extend([
        section('3. Data Audit','3'),
        md('The model uses only the eight input data tables, and only features observable at each selected application decision date. The train/test application IDs and labels are checked before generating features.'),
        code('''train_raw = pd.read_csv(CFG.DATA_DIR / "train_applications.csv")
test_raw = pd.read_csv(CFG.DATA_DIR / "test_applications.csv")
assert train_raw.application_id.is_unique and test_raw.application_id.is_unique
assert train_raw.customer_id.is_unique and test_raw.customer_id.is_unique
assert not set(train_raw.application_id) & set(test_raw.application_id)
assert not set(train_raw.customer_id) & set(test_raw.customer_id)
assert len(train_raw) == CFG.EXPECTED_TRAIN and len(test_raw) == CFG.EXPECTED_TEST
assert set(train_raw.target.dropna().unique()) == {0, 1}
assert set(sample.application_id) == set(test_raw.application_id)
assert len(sample) == CFG.EXPECTED_TEST
print("Train:", train_raw.shape, "Test:", test_raw.shape)
print("Positive label rate:", round(train_raw.target.mean(), 6))
print("Train dates:", train_raw.decision_datetime.min(), "to", train_raw.decision_datetime.max())'''),
        md('#### Insights\n\n> IDs, split integrity and class balance are checked against the expected competition inventory before any model is fit.'),
        section('4. Point-in-Time Features','4'),
        md('The following DuckDB transformations reproduce the existing audited 63-column feature pipeline directly from the competition CSVs. Previous applications, installment repayments, monthly account data, bureau snapshots and individual e-commerce orders are all constrained to the selected decision time.'),
        code('''TABLE_NAMES = (
    "train_applications", "test_applications", "customer_profiles",
    "previous_applications", "repayment_history", "account_behavior",
    "bureau_observations", "ecommerce_transactions",
)
feature_t0 = time.perf_counter()
con = duckdb.connect(database=":memory:")
con.execute(f"SET threads = {CFG.THREADS}")
for table in TABLE_NAMES:
    csv_path = str(CFG.DATA_DIR / f"{table}.csv").replace("'", "''")
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{csv_path}')")
con.execute("""
    CREATE TEMP TABLE applications AS
    SELECT *, 'train' AS split FROM train_applications
    UNION ALL BY NAME
    SELECT *, 'test' AS split FROM test_applications
""")
print("DuckDB views loaded:", len(TABLE_NAMES))'''),
    ])
    for sql in extract_sql():
        table = sql.split('TABLE ', 1)[1].split(' AS', 1)[0].strip()
        if table == 'features':
            cells.append(md('The final one-row-per-application join combines the independently aggregated feature families.'))
        cells.append(code(f'con.execute({sql!r})\nprint("Built SQL table: {table}")'))
    cells.extend([
        md('The feature table is materialized once. Identifiers and the selected raw timestamp are retained only for validation, not passed to the classifier.'),
        code('''features_all = con.execute("SELECT * FROM features").df()
con.close()
features_all["decision_datetime"] = pd.to_datetime(features_all["decision_datetime"])
train_df = features_all.loc[features_all.split == "train"].sort_values("application_id").reset_index(drop=True)
test_df = features_all.loc[features_all.split == "test"].sort_values("application_id").reset_index(drop=True)
feature_cols = [c for c in features_all.columns if c not in CFG.METADATA and c != "term_months"]
assert len(feature_cols) == CFG.EXPECTED_FEATURES, (len(feature_cols), feature_cols)
assert len(train_df) == CFG.EXPECTED_TRAIN and len(test_df) == CFG.EXPECTED_TEST
assert train_df.application_id.is_unique and test_df.application_id.is_unique
assert set(train_df.application_id) == set(train_raw.application_id)
assert set(test_df.application_id) == set(test_raw.application_id)
assert set(sample.application_id) == set(test_df.application_id)
assert train_df.customer_id.is_unique and test_df.customer_id.is_unique
assert not set(train_df.customer_id) & set(test_df.customer_id)
assert train_df.target.notna().all() and set(train_df.target.unique()) == {0, 1}
assert test_df.target.isna().all()
assert train_df.term_months.nunique(dropna=False) == 1
assert set(CFG.CATEGORICALS).issubset(feature_cols)
for name in feature_cols:
    if name not in CFG.CATEGORICALS:
        train_df[name] = pd.to_numeric(train_df[name], errors="raise").astype("float32")
        test_df[name] = pd.to_numeric(test_df[name], errors="raise").astype("float32")
        assert np.isfinite(train_df[name].dropna().to_numpy()).all(), name
        assert np.isfinite(test_df[name].dropna().to_numpy()).all(), name
print("Feature rows:", len(train_df), len(test_df), "model features:", len(feature_cols))
feature_elapsed = time.perf_counter() - feature_t0
print("Feature SQL elapsed seconds:", round(feature_elapsed, 1))
print("Categorical feature families:", CFG.CATEGORICALS)
print("Column inventory:", feature_cols)'''),
        md('#### Insights\n\n> Exactly 62 non-constant model features are used. Numeric missingness remains missing, and historical event counts without records are set to zero by the audited SQL. No raw ID, label or selected timestamp enters the model.'),
    ])


def validation(cells: list):
    cells.extend([
        section('5. Validation Design', '5'),
        md('One deterministic five-fold stratified split is generated after sorting IDs. Each outer fold is strictly held out from inner early stopping and any fitted categorical encoding. The fold manifest and SHA-256 hash enable direct comparison across notebooks.'),
        code('''y = train_df.target.astype("int8").to_numpy()
fold_index = np.full(len(train_df), -1, dtype=np.int8)
outer_cv = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
outer_splits = list(outer_cv.split(train_df, y))
for fold, (idx_train, idx_valid) in enumerate(outer_splits):
    fold_index[idx_valid] = fold
    assert not set(train_df.iloc[idx_train].customer_id) & set(train_df.iloc[idx_valid].customer_id)
assert (fold_index >= 0).all()
folds = train_df[["application_id"]].copy()
folds["fold"] = fold_index
fold_manifest = folds.to_csv(index=False, lineterminator="\\n")
fold_digest = hashlib.sha256(fold_manifest.encode("utf-8")).hexdigest()
folds.to_csv(CFG.OUT / "folds.csv", index=False)
print("Five-fold SHA-256 digest:", fold_digest)
print(folds.groupby("fold").size().to_string())
print(train_df.groupby(fold_index).target.agg(["size", "mean"]).to_string())

def score_metrics(y_true, probabilities):
    positives = np.asarray(probabilities)[np.asarray(y_true) == 1]
    negatives = np.asarray(probabilities)[np.asarray(y_true) == 0]
    if not len(positives) or not len(negatives):
        return {"ks": 0.0, "roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "ks": float(ks_2samp(positives, negatives).statistic),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }

print("Official KS function and fold contract ready.")'''),
        md('#### Insights\n\n> The digest is expected to be identical across LightGBM, XGBoost and CatBoost. Pooled out-of-fold KS is calculated on predictions from models that never trained on their respective validation rows.'),
    ])


def model_code(model: str) -> str:
    if model == 'lightgbm':
        return '''def model_train_predict(fit_part, pred_part, rounds, eval_part=None):
    cats = {}
    for name in CFG.CATEGORICALS:
        vocab = sorted(fit_part[name].dropna().astype(str).unique().tolist())
        cats[name] = pd.CategoricalDtype(categories=vocab)

    def transform(part):
        X = part[feature_cols].copy()
        for name, dtype in cats.items():
            X[name] = X[name].astype("string").astype(dtype)
        return X

    X_fit = transform(fit_part)
    X_pred = transform(pred_part)
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=int(rounds), learning_rate=0.03,
        num_leaves=31, min_child_samples=100, reg_lambda=3.0,
        random_state=CFG.SEED, n_jobs=CFG.THREADS, verbosity=-1,
        importance_type="gain",
    )
    kwargs = {"categorical_feature": list(CFG.CATEGORICALS)}
    if eval_part is not None:
        kwargs["eval_set"] = [(transform(eval_part), eval_part.target.astype(int))]
        kwargs["eval_metric"] = "binary_logloss"
        kwargs["callbacks"] = [lgb.early_stopping(CFG.EARLY_STOPPING_ROUNDS, verbose=False)]
    model.fit(X_fit, fit_part.target.astype(int), **kwargs)
    best_rounds = int(model.best_iteration_) if eval_part is not None else int(rounds)
    probabilities = model.predict_proba(X_pred)[:, 1]
    gain = model.booster_.feature_importance(importance_type="gain")
    return probabilities, best_rounds, model, np.asarray(gain), list(feature_cols), None
'''
    if model == 'xgboost':
        return '''def model_train_predict(fit_part, pred_part, rounds, eval_part=None):
    numeric = [c for c in feature_cols if c not in CFG.CATEGORICALS]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)

    def extract_cats(part):
        return part[list(CFG.CATEGORICALS)].astype("string").fillna("__MISSING__")

    encoder.fit(extract_cats(fit_part))

    def transform(part):
        num = sparse.csr_matrix(part[numeric].to_numpy(dtype=np.float32))
        cat = encoder.transform(extract_cats(part))
        return sparse.hstack((num, cat), format="csr", dtype=np.float32)

    X_fit = transform(fit_part)
    X_pred = transform(pred_part)
    names = numeric + encoder.get_feature_names_out(list(CFG.CATEGORICALS)).tolist()
    model_kwargs = dict(
        objective="binary:logistic", tree_method="hist", eval_metric="logloss",
        n_estimators=int(rounds), learning_rate=0.03, max_depth=5,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
        random_state=CFG.SEED, n_jobs=CFG.THREADS,
    )
    if eval_part is not None:
        model_kwargs["early_stopping_rounds"] = CFG.EARLY_STOPPING_ROUNDS
    model = xgb.XGBClassifier(**model_kwargs)
    fit_kwargs = {}
    if eval_part is not None:
        fit_kwargs["eval_set"] = [(transform(eval_part), eval_part.target.astype(int))]
        fit_kwargs["verbose"] = False
    model.fit(X_fit, fit_part.target.astype(int), **fit_kwargs)
    best_rounds = int(model.best_iteration + 1) if eval_part is not None else int(rounds)
    probabilities = model.predict_proba(X_pred)[:, 1]
    importance = np.asarray(model.feature_importances_)
    return probabilities, best_rounds, model, importance, names, encoder
'''
    if model == 'catboost':
        return '''def model_train_predict(fit_part, pred_part, rounds, eval_part=None):
    def transform(part):
        X = part[feature_cols].copy()
        for name in CFG.CATEGORICALS:
            X[name] = X[name].astype("string").fillna("__MISSING__").astype(str)
        return X

    X_fit = transform(fit_part)
    fit_pool = cb.Pool(X_fit, label=fit_part.target.astype(int), cat_features=list(CFG.CATEGORICALS))
    predict_pool = cb.Pool(transform(pred_part), cat_features=list(CFG.CATEGORICALS))
    model = cb.CatBoostClassifier(
        iterations=int(rounds), learning_rate=0.03, depth=6,
        l2_leaf_reg=5.0, loss_function="Logloss", eval_metric="Logloss",
        random_seed=CFG.SEED, thread_count=CFG.THREADS,
        verbose=False, allow_writing_files=False,
    )
    kwargs = {}
    if eval_part is not None:
        kwargs["eval_set"] = cb.Pool(transform(eval_part), label=eval_part.target.astype(int), cat_features=list(CFG.CATEGORICALS))
        kwargs["early_stopping_rounds"] = CFG.EARLY_STOPPING_ROUNDS
        kwargs["use_best_model"] = True
    model.fit(fit_pool, **kwargs)
    best_rounds = int(model.get_best_iteration() + 1) if eval_part is not None else int(rounds)
    probabilities = model.predict_proba(predict_pool)[:, 1]
    importance = model.get_feature_importance(type="FeatureImportance")
    return probabilities, best_rounds, model, np.asarray(importance), list(feature_cols), None
'''
    raise ValueError(model)


def train_section(cells: list, model: str):
    cells.extend([
        section('6. Model Training','6'),
        md('The first baseline uses a fixed tree configuration, no class weighting and binary log loss as the early-stopping surrogate. Category conversion, one-hot vocabularies or CatBoost pools are learned only from the training slice passed to each fit.'),
        code(model_code(model)),
        md('Each outer fold performs an independent stratified inner validation to choose a boosting round, then retrains on the entire outer training partition. The untouched outer validation row receives exactly one prediction.'),
        code('''train_t0 = time.perf_counter()
oof = np.full(len(train_df), np.nan, dtype=np.float64)
selected_rounds = []
fold_results = []
fold_importances = []

for fold, (outer_train_idx, outer_valid_idx) in enumerate(outer_splits):
    inner_train_idx, inner_valid_idx = train_test_split(
        outer_train_idx,
        test_size=CFG.INNER_VALID_FRACTION,
        stratify=y[outer_train_idx],
        random_state=CFG.SEED + fold,
    )
    _, best_round, _, _, _, _ = model_train_predict(
        train_df.iloc[inner_train_idx], train_df.iloc[inner_valid_idx],
        rounds=CFG.MAX_ROUNDS, eval_part=train_df.iloc[inner_valid_idx],
    )
    assert 1 <= best_round <= CFG.MAX_ROUNDS
    scores, _, fitted, importance, names, _ = model_train_predict(
        train_df.iloc[outer_train_idx], train_df.iloc[outer_valid_idx],
        rounds=best_round,
    )
    assert np.isfinite(scores).all()
    oof[outer_valid_idx] = scores
    metrics = score_metrics(y[outer_valid_idx], scores)
    metrics.update({"fold": int(fold), "n_validation": int(len(outer_valid_idx)), "best_round": int(best_round)})
    fold_results.append(metrics)
    fold_importances.append(pd.Series(importance, index=names).rename(f"fold_{fold}"))
    selected_rounds.append(int(best_round))
    print(f"Fold {fold}: KS={metrics['ks']:.5f}, AUC={metrics['roc_auc']:.5f}, AP={metrics['average_precision']:.5f}, rounds={best_round}")

assert np.isfinite(oof).all() and np.std(oof) > 0
cv_elapsed = time.perf_counter() - train_t0
fold_metrics = pd.DataFrame(fold_results)
pooled_metrics = score_metrics(y, oof)
print("\\nPer-fold mean KS:", fold_metrics.ks.mean(), "+/-", fold_metrics.ks.std(ddof=1))
print("Pooled OOF KS:", pooled_metrics["ks"])
print("Total CV seconds:", round(cv_elapsed, 1))
fold_metrics.to_csv(CFG.OUT / "fold_metrics.csv", index=False)
oof_df = train_df[["application_id", "target"]].copy()
oof_df["fold"] = fold_index
oof_df["score"] = oof
oof_df.to_csv(CFG.OUT / f"oof_{CFG.MODEL_NAME}.csv", index=False)
feature_importance = pd.concat(fold_importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
feature_importance.rename("mean_importance").to_csv(CFG.OUT / "feature_importance.csv", header=True)
print("Most important features:", feature_importance.head(12).to_dict())'''),
        md('The time-stress check fits a model using January through September decisions and scores October through December decisions. It is **retrospective** because the first nine months of loan outcomes would not all be mature by October 2025.'),
        code('''date_year = train_df.decision_datetime.dt.year
assert date_year.eq(2025).all(), "Temporal stress window was designed for 2025 decisions"
months = train_df.decision_datetime.dt.month
time_train = np.flatnonzero(months.le(9).to_numpy())
time_valid = np.flatnonzero(months.ge(10).to_numpy())
assert len(time_train) and len(time_valid) and not set(time_train) & set(time_valid)
temporal_t0 = time.perf_counter()
time_inner_train, time_inner_valid = train_test_split(
    time_train, stratify=y[time_train],
    test_size=CFG.INNER_VALID_FRACTION, random_state=CFG.SEED,
)
_, temporal_rounds, _, _, _, _ = model_train_predict(
    train_df.iloc[time_inner_train], train_df.iloc[time_inner_valid],
    rounds=CFG.MAX_ROUNDS, eval_part=train_df.iloc[time_inner_valid],
)
temporal_prob, _, _, _, _, _ = model_train_predict(
    train_df.iloc[time_train], train_df.iloc[time_valid], rounds=temporal_rounds,
)
temporal_metrics = score_metrics(y[time_valid], temporal_prob)
temporal_metrics.update({"n_train": int(len(time_train)), "n_validation": int(len(time_valid)), "best_round": temporal_rounds,
                         "elapsed_seconds": round(time.perf_counter() - temporal_t0, 2)})
print("Retrospective Jan-Sep -> Oct-Dec:", temporal_metrics)
pd.DataFrame({"application_id": train_df.iloc[time_valid].application_id.values,
              "target": y[time_valid], "score": temporal_prob}).to_csv(
    CFG.OUT / "temporal_stress_scores.csv", index=False)'''),
        md('#### Insights\n\n> The outer-fold KS values quantify held-out score separation. The time stress score tests distribution sensitivity and must not be interpreted as a prospective deployment simulation.'),
    ])


def evaluation(cells: list):
    cells.extend([
        section('7. Evaluation','7'),
        md('Diagnostics complement official KS with ROC-AUC, average precision, positive/negative score CDFs, risk-decile rates and feature attribution. All plots refer to the pooled **out-of-fold** predictions rather than in-sample fits.'),
        code('''fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].bar(fold_metrics.fold.astype(str), fold_metrics.ks)
axes[0, 0].axhline(fold_metrics.ks.mean(), color="black", linestyle="--", label="Mean KS")
axes[0, 0].set(title=f"{CFG.MODEL_NAME}: 5-fold KS", xlabel="Outer fold", ylabel="KS")
axes[0, 0].legend()

positive = np.sort(oof[y == 1])
negative = np.sort(oof[y == 0])
axes[0, 1].step(negative, np.arange(1, len(negative) + 1) / len(negative), where="post", label="Target 0")
axes[0, 1].step(positive, np.arange(1, len(positive) + 1) / len(positive), where="post", label="Target 1")
ks_at = ks_2samp(positive, negative).statistic_location
axes[0, 1].axvline(ks_at, linestyle="--", color="black", label=f"KS={pooled_metrics['ks']:.3f}")
axes[0, 1].set(title="Pooled OOF empirical CDFs", xlabel="Predicted risk", ylabel="Empirical CDF")
axes[0, 1].legend()

fpr, tpr, _ = roc_curve(y, oof)
axes[1, 0].plot(fpr, tpr, label=f"ROC-AUC {pooled_metrics['roc_auc']:.3f}")
axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1, 0].set(title="Pooled OOF ROC", xlabel="False positive rate", ylabel="True positive rate")
axes[1, 0].legend()

precision, recall, _ = precision_recall_curve(y, oof)
axes[1, 1].plot(recall, precision, label=f"AP {pooled_metrics['average_precision']:.3f}")
axes[1, 1].axhline(y.mean(), linestyle="--", color="gray", label="Class prevalence")
axes[1, 1].set(title="Pooled OOF precision-recall", xlabel="Recall", ylabel="Precision")
axes[1, 1].legend()

fig.suptitle(f"Validation diagnostics | {len(train_df):,} OOF observations")
fig.tight_layout()
fig.savefig(CFG.OUT / "validation_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()'''),
        md('Risk deciles summarize observed default rates from low-scored to high-scored applications. Decile assignment is descriptive; the competition metric does not use a classification threshold.'),
        code('''deciles = oof_df.copy()
deciles["risk_decile"] = pd.qcut(deciles.score.rank(method="first"), q=10, labels=False) + 1
decile_table = deciles.groupby("risk_decile", observed=True).agg(
    n=("target", "size"), observed_default_rate=("target", "mean"),
    mean_score=("score", "mean"), min_score=("score", "min"), max_score=("score", "max"),
).reset_index()
print(decile_table.to_string(index=False))
decile_table.to_csv(CFG.OUT / "risk_deciles.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(decile_table.risk_decile, decile_table.observed_default_rate)
ax.set(xlabel="Risk decile (10 = highest predicted risk)", ylabel="Observed serious delinquency rate",
       title=f"{CFG.MODEL_NAME} pooled OOF risk-decile separation")
fig.tight_layout()
fig.savefig(CFG.OUT / "risk_deciles.png", dpi=150, bbox_inches="tight")
plt.show()'''),
        md('To examine feature-family relevance, importance is averaged across the five outer-fold models; one-hot columns are first aggregated to their original categorical source. Importance is an association with the model, not evidence of causality or point-in-time validity.'),
        code('''importance_for_plot = feature_importance.copy()
if CFG.MODEL_NAME == "xgboost":
    group_importance = {}
    for name, value in feature_importance.items():
        src = next((c for c in CFG.CATEGORICALS if name.startswith(c + "_")), name)
        group_importance[src] = group_importance.get(src, 0.0) + float(value)
    importance_for_plot = pd.Series(group_importance).sort_values(ascending=False)
importance_for_plot.rename("mean_importance").to_csv(CFG.OUT / "feature_importance_grouped.csv", header=True)
fig, ax = plt.subplots(figsize=(10, 7))
importance_for_plot.head(22).sort_values().plot.barh(ax=ax)
ax.set(title=f"{CFG.MODEL_NAME}: outer-fold mean feature importance", xlabel="Model-specific importance")
fig.tight_layout()
fig.savefig(CFG.OUT / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

subgroup_results = []
for flag in ("previous_count", "due_installment_count", "account_complete_months"):
    for group, mask in (("none", train_df[flag].fillna(0).eq(0)), ("present", train_df[flag].fillna(0).gt(0))):
        sub_y = y[mask.to_numpy()]
        sub_score = oof[mask.to_numpy()]
        m = score_metrics(sub_y, sub_score)
        subgroup_results.append({"feature": flag, "history": group, "n": int(len(sub_y)),
                                 "positives": int(sub_y.sum()), **m})
subgroup_metrics = pd.DataFrame(subgroup_results)
print(subgroup_metrics.to_string(index=False))
subgroup_metrics.to_csv(CFG.OUT / "history_subgroups.csv", index=False)'''),
        md('#### Insights\n\n> Strong bureau, affordability, repayment or spending importance would be consistent with the prior EDA, but neither stability nor absence of leakage follows from feature importance alone. Compare temporal and no-history subgroups before choosing a final candidate.'),
    ])


def submission(cells: list, model: str):
    cells.extend([
        section('8. Submission','8'),
        md('A single final model is refit on all labeled applications. Its boosting rounds are fixed to the median selected inner-validation count from the outer folds; no test labels or extra data enter training.'),
        code('''final_rounds = max(1, int(np.median(selected_rounds)))
full_t0 = time.perf_counter()
test_scores, _, final_model, final_importance, final_names, final_encoder = model_train_predict(
    train_df, test_df, rounds=final_rounds,
)
full_elapsed = time.perf_counter() - full_t0
assert len(test_scores) == len(test_df) == len(sample)
assert np.isfinite(test_scores).all()
assert np.all((test_scores >= 0) & (test_scores <= 1))
assert np.ptp(test_scores) > 1e-8, "Predictions unexpectedly constant"

pred_df = pd.DataFrame({"application_id": test_df.application_id, "default_probability": test_scores})
submission_df = sample[["application_id"]].merge(pred_df, on="application_id", how="left", validate="one_to_one", sort=False)
assert submission_df.columns.tolist() == ["application_id", "default_probability"]
assert submission_df.application_id.tolist() == sample.application_id.tolist()
assert submission_df.application_id.is_unique
assert len(submission_df) == CFG.EXPECTED_TEST
assert submission_df.default_probability.notna().all()
assert np.isfinite(submission_df.default_probability.to_numpy()).all()
assert submission_df.default_probability.between(0, 1).all()
assert np.ptp(submission_df.default_probability.to_numpy()) > 1e-8
submission_path = CFG.OUTPUT_ROOT / f"submission_{CFG.MODEL_NAME}.csv"
submission_df.to_csv(submission_path, index=False)
submission_df.to_csv(CFG.OUT / f"submission_{CFG.MODEL_NAME}.csv", index=False)
print("Final fit + inference seconds:", round(full_elapsed, 2))
print("Final rounds:", final_rounds)
print("Created:", submission_path)
print(submission_df.head().to_string(index=False))'''),
        md('The fitted model is serialized for later comparison and inference. XGBoost also saves its full-data one-hot encoder; the other two frameworks carry their categorical representations in the trained model.'),
        code({
            'lightgbm': '''model_file = CFG.OUT / "lightgbm_final_model.txt"
final_model.booster_.save_model(str(model_file))
print("Saved model:", model_file)''',
            'xgboost': '''model_file = CFG.OUT / "xgboost_final_model.json"
final_model.save_model(str(model_file))
encoder_file = CFG.OUT / "xgboost_full_encoder.joblib"
joblib.dump(final_encoder, encoder_file)
print("Saved model:", model_file, "and encoder:", encoder_file)''',
            'catboost': '''model_file = CFG.OUT / "catboost_final_model.cbm"
final_model.save_model(str(model_file))
print("Saved model:", model_file)''',
        }[model]),
        md('The metrics JSON captures actual fold scores, the time stress result, input-feature names, execution times, and dependency versions. No fabricated local model scores are embedded in this notebook.'),
        code('''versions = {
    "python": platform.python_version(), "duckdb": duckdb.__version__,
    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "model_package": {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}[CFG.MODEL_NAME],
    "model_version": {"lightgbm": lgb.__version__ if CFG.MODEL_NAME == "lightgbm" else None,
                      "xgboost": xgb.__version__ if CFG.MODEL_NAME == "xgboost" else None,
                      "catboost": cb.__version__ if CFG.MODEL_NAME == "catboost" else None}[CFG.MODEL_NAME],
}
run_report = {
    "model": CFG.MODEL_NAME, "seed": CFG.SEED, "fold_sha256": fold_digest,
    "features": feature_cols, "feature_count": len(feature_cols), "categoricals": list(CFG.CATEGORICALS),
    "n_train": len(train_df), "n_test": len(test_df), "positives": int(y.sum()),
    "folds": fold_results,
    "mean_fold_ks": float(fold_metrics.ks.mean()),
    "std_fold_ks": float(fold_metrics.ks.std(ddof=1)),
    "pooled_oof": pooled_metrics,
    "retrospective_temporal_stress": temporal_metrics,
    "final_rounds": final_rounds, "cv_seconds": round(cv_elapsed, 2),
    "feature_engineering_seconds": round(feature_elapsed, 2),
    "final_fit_and_test_inference_seconds": round(full_elapsed, 2),
    "early_stopping_metric": "binary_logloss" if CFG.MODEL_NAME == "lightgbm" else "logloss",
    "versions": versions,
}
metrics_file = CFG.OUT / f"metrics_{CFG.MODEL_NAME}.json"
metrics_file.write_text(json.dumps(run_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: run_report[k] for k in ("model", "fold_sha256", "mean_fold_ks", "pooled_oof", "retrospective_temporal_stress", "final_rounds", "versions")}, indent=2))
print("Metrics written:", metrics_file)'''),
        md('#### Insights\n\n> The Kaggle-ready submission has exactly 30,000 unique IDs, two required columns, and finite nonconstant probabilities in the allowed range. Full-data model refitting does not change the previously computed out-of-fold diagnostics.'),
        section('9. Reproduction Notes', '9'),
        md(f'''This notebook runs independently using only committee CSVs from `/kaggle/input/competitions/data-science-competition-logika-ui-2026` or another unique attached Kaggle input directory. It uses **CPU**, four threads and seed 42; the first cell installs DuckDB and the selected model dependency.

All outputs are saved in `/kaggle/working/{model}/`, with the ready-to-upload CSV also at `/kaggle/working/submission_{model}.csv`. Use **Run All**, then **Save Version**, then download the submission from the notebook **Output**. Actual scores and installed version strings are recorded in the run-specific metrics JSON.

**Validation limitation:** The Jan-Sep / Oct-Dec split is retrospective, not a prospective operational backtest, because a twelve-month default label is not fully mature at the October 2025 cutoff.

**Competition delivery:** After selecting a single final model, rename the designated notebook file to `ParticipantNumber_TeamName_Notebook.ipynb` using the actual registered participant number and team name.'''),
    ])


def generate(model: str):
    if model not in ('lightgbm', 'xgboost', 'catboost'):
        raise ValueError(model)
    cells = []
    intro(cells, model)
    initialization(cells, model)
    data_pipeline(cells)
    validation(cells)
    train_section(cells, model)
    evaluation(cells)
    submission(cells, model)
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'cells': cells}
    path = HERE / f"{'01' if model == 'lightgbm' else '02' if model == 'xgboost' else '03'}_{model}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Wrote {len(cells)} cells -> {path}')
    return path
