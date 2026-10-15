import json
import re
import sys
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "v8.ipynb"
NN_PREP_SRC = Path(__file__).resolve().parent.parent / "eda" / "nn_prep.py"
FEATURES_SRC = Path(__file__).resolve().parent.parent / "eda" / "features.py"
HIST_SRC = Path(__file__).resolve().parent.parent / "eda" / "hist_snapshots.py"
HR  = "---"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }

def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )

def C(src: str) -> dict:
    return code(dedent(src).strip("\n"))

def M(src: str) -> dict:
    return md(dedent(src).strip("\n"))

cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# DSC LOGIKA UI 2026: Credit Risk Prediction\n\n"
    "*Predicting 90+ DPD within 12 months of origination for Indonesian consumer loans, scored by the KS statistic.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## TeamName (DSCxxxx)\n\n"
    "- Name 1 (Role)\n"
    "- Name 2\n"
    "- Name 3"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Exploratory Data Analysis**](#3)\n"
    "4. [**Feature Engineering**](#4)\n"
    "5. [**Validation Strategy**](#5)\n"
    "6. [**Gradient Boosting Models**](#6)\n"
    "7. [**Foundation and Neural Models**](#7)\n"
    "8. [**Ensemble and KS Stability**](#8)\n"
    "9. [**Submission**](#9)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(M("""
    ## Overview

    *Credit risk is a central challenge in consumer lending. A lender has to judge, at the moment of the loan decision, whether an application is likely to run into serious repayment difficulty using only the information available at that time.*

    *This competition frames the problem on a multi-table Indonesian consumer lending dataset. Each selected loan has a 12-month term, and the goal is to rank applications by their risk of reaching 90 or more days past due within 12 months of origination.*
"""))

cells.append(M("""
    ## Aim

    *Build a leak-free scoring pipeline that aggregates six behavioural tables into one row per application and produces a risk score with maximal and stable KS separation on both the public and the private leaderboard.*
"""))

cells.append(M("""
    ## Metric

    ***The Kolmogorov-Smirnov (KS) statistic** measures the largest vertical gap between the cumulative score distributions of bad loans and good loans. It depends only on the ranking of scores, so calibration does not matter and rank-averaging is a safe way to blend models.*

    $$KS = \\max_s \\left| F_1(s) - F_0(s) \\right|$$
"""))

cells.append(M("""
    ## Dataset

    *Data sourced from the Kaggle competition DSC LOGIKA UI 2026.*

    *The training set contains 70,000 selected applications (13.58% bad) and the test set contains 30,000. Every customer has exactly one selected application, and train and test customers are disjoint. About 30% of customers in both splits are thin-file, with no previous applications, repayment history or account data.*

    *The dataset is used to engineer customer-level features that respect the `decision_datetime` information cutoff.*

    **Attributes**

    1.  `train/test_applications`  : Selected loan (amount, 12-month term, rate tier, installment, purpose)
    2.  `customer_profiles`        : Age, region, employment, tenure, declared income, dependents, housing
    3.  `previous_applications`    : Historical applications with decision, approved amount and rate
    4.  `repayment_history`        : Installment-level amounts, payment dates and days past due
    5.  `account_behavior`         : Monthly balances, inflows, outflows and low-balance days
    6.  `bureau_observations`      : Up to three bureau reports per application (score, utilization, inquiries)
    7.  `ecommerce_transactions`   : Item-level orders with category, payment method and status

    **Target**
     `target`       : 1 if the loan reaches 90+ DPD within 12 months, 0 otherwise
"""))

cells.append(M("""
    ## Approach: Behavioural Features + GBDT and Tabular Foundation Model Ensemble

    *The approach aggregates every auxiliary table to the customer level with explicit cutoff filters. Window lengths are chosen from the exploratory analysis: repayment lateness is a stable trait, so full-history ratios are used, while e-commerce activity and cash-flow stress ramp up in the 3-4 months before the application, so recent-versus-baseline acceleration features are used.*

    *Version 2 adds 38 features found by a residual analysis of the version 1 out-of-fold predictions. They cover lateness severity (how many months a late installment stays unpaid), first-installment and latest-loan behaviour, non-income inflows, bureau score trend, cancellations and night-time orders. The same analysis showed that risk is close to additive in log-odds across the signal families, so every tree library is run twice: a deep variant (as in version 1) and a shallow variant.*

    *Version 6 starts from version 5 (public leaderboard 0.561), whose jump came from one new measurement: the unit price a customer pays relative to the typical price of the same category. Three independent residual sweeps of the version 5 predictions (39 per-row attributes from every table, 29 price variants, and 301 generic DFS columns) found nothing else above noise, so version 6 does not add features. It makes the pipeline stricter and widens the model side. The category price reference is now built only from purchases dated before the application's decision month, so every application is normalised with prices that were already observed. The TabM input map is fitted inside each fold. The foundation models get a fair test: TabPFN-3.5 now sees the same hand-made features as the trees, and EXAONE receives exactly its 100-column limit. A second TabM with periodic embeddings joins the blend.*

    *Version 7 kept the version 6 features and spent a full Kaggle T4 x2 session on the model side. Its blend reached 0.5268 on held-out halves (version 6: 0.5263) with a test rank correlation of 0.9995 to version 6, so it was the same submission. Causilo ran for the first time (0.5226), native TabPFN-Rel scored 0.4917 and EXAONE stopped at its 56,000-row context because the T4 could not hold it.*

    *Version 8 adds new training data instead of new columns. The previous applications contain 81,702 approved loans whose 12-month window ended before the customer's selected decision. Their outcome is observed in the repayment history (90 DPD within 12 months of origination, the same definition as the target), with a bad rate of 14.4% against 13.6% for the selected loans. Each of these loans becomes an extra training row whose features are rebuilt as of that loan's decision time with the same feature code, after resetting the repayment snapshot to that time. The offline analysis (`eda/80` to `eda/83`) checked the labels, verified that the as-of features cannot see the future, and measured how well the historical rows transfer: bureau, pricing and repayment features have the same risk curves in both populations, while the e-commerce burst and the price premium are much weaker on historical loans. The tree models and TabM are therefore trained twice, without and with the historical rows (flagged by `is_hist`, weight 0.5), on identical folds, so their effect is measured as a paired difference. The model side gains three members suggested by the version 7 review: RealMLP, TabICLv2 and EXAONE at a context the T4 can hold. Causilo now receives typed categorical columns.*

    *Three gradient boosting libraries (LightGBM, XGBoost on GPU, CatBoost on GPU) are trained with repeated stratified 5-fold CV. Out-of-fold predictions are rank-blended. The blend weights are chosen on held-out halves among an equal-weight tree blend, a fixed trees plus TabM blend, greedy selection and a bagged greedy selection, and the blend is checked for KS stability before submission.*

    ```
    7 raw tables
        |
        +-- cutoff-filtered aggregations, one row per selected application (70,000 train + 30,000 test)
        |
        +-- the same aggregations as of each earlier loan's decision (81,702 extra labelled rows, is_hist = 1)
                         |
        +----------+-----------+-----------+-------------+-----------+-----------+-----------+
        |          |           |           |             |           |           |           |
      LightGBM   XGBoost    CatBoost      TabM        RealMLP    TabPFN-3.5   TabICLv2   EXAONE,
      (deep + shallow, each without and with  (PLE, periodic,          Causilo   (12k ctx)
       the historical rows)                    PLE + history)
                         |
          rank blend (weights chosen on held-out halves)
                         |
                  default_probability
    ```

    *Train and test are a random split of the same period (PSI below 0.002 for every feature), so stratified K-fold is the correct validation scheme. The public-versus-private KS gap expected from sampling noise alone is about 0.01, so decisions are made on CV rather than on the public leaderboard.*
"""))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(M("""
    ## Environment Setup

    The notebook targets Kaggle with the **GPU T4 x2** accelerator and **Internet on**. TabPFN weights are gated: accept the licence on Hugging Face and add `HF_TOKEN` (or `TABPFN_TOKEN`) under Add-ons > Secrets. Run it with Save Version > Save & Run All; the outputs land in `/kaggle/working`. Outside Kaggle the first `data/` folder holding the CSV files is used and outputs go to `output_v8/`.

    Kaggle stops a session after 12 hours. No model has its own time budget. Instead the notebook keeps one deadline, 11 hours 15 minutes after the libraries load (`RUN_LIMIT_MIN`), and runs the models in order of priority: the trees without and with the historical rows, TabM, TabPFN-3.5, Causilo, TabICLv2, RealMLP, the periodic TabM and EXAONE. A model is dropped only when its next fold cannot finish before the deadline minus `BLEND_RESERVE_MIN`, so the notebook always reaches the blend and the submission. The expected total is about 8 to 9 hours.

    Every model after the trees is optional. If a package does not install or a model fails, its step is skipped and the notebook still completes.
"""))

cells.append(code("!nvidia-smi"))

cells.append(M("""
    The following cell installs all libraries used in this notebook. PyTorch is not reinstalled, because the Kaggle image already ships the CUDA build that matches the GPU. EXAONE and Causilo are installed without dependencies so that pip cannot replace that PyTorch build (Causilo declares PyTorch 2.13 or newer but ran on the Kaggle PyTorch 2.11 in version 7). TabICL and PyTabKit (RealMLP) are installed with their dependencies: they require PyTorch 2.2 and 2.0 or newer, which the image already satisfies, so pip keeps its build.
"""))

cells.append(C("""
    %pip install -q polars==2.0.0 lightgbm xgboost catboost tabpfn tabm rtdl_num_embeddings scikit-learn pyarrow
    %pip install -q --no-deps "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git" || echo 'EXAONE install failed'
    %pip install -q --no-deps causilo || echo 'Causilo install failed'
    %pip install -q tabicl pytabkit || echo 'TabICL / PyTabKit install failed'
"""))

cells.append(md("## Import Libraries"))

cells.append(C("""
    import copy
    import gc
    import os
    import random
    import time
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import torch
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool
    from scipy.special import ndtri
    from scipy.stats import ks_2samp, rankdata
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    try:
        from tabpfn import TabPFNClassifier
    except Exception as e:
        TabPFNClassifier = None
        print('TabPFN unavailable:', repr(e))
    try:
        from exaonetabular import EXAONETabularClassifier
    except Exception as e:
        EXAONETabularClassifier = None
        print('EXAONE unavailable:', repr(e))
    try:
        import tabm
        import rtdl_num_embeddings
    except Exception as e:
        tabm = None
        print('TabM unavailable:', repr(e))
    try:
        from tabicl import TabICLClassifier
    except Exception as e:
        TabICLClassifier = None
        print('TabICL unavailable:', repr(e))
    try:
        from pytabkit import RealMLP_TD_Classifier
    except Exception as e:
        RealMLP_TD_Classifier = None
        print('RealMLP unavailable:', repr(e))
    try:
        from causilo import CausiloClassifier
    except Exception as e:
        CausiloClassifier = None
        print('Causilo unavailable:', repr(e))
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        UserSecretsClient = None

    warnings.filterwarnings('ignore')
    print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count(), '| polars', pl.__version__)
"""))

cells.append(md("## Seed Everything"))

cells.append(C("""
    def seed_everything(seed: int = 42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(42)
"""))

cells.append(M("""
    ## Settings

    All paths, hyperparameters, and environment flags are centralised here. On Kaggle the data directory is found under `/kaggle/input`. Elsewhere the first `data/` folder holding `train_applications.csv` is used, searched next to the notebook, one level up and under `/workspace`.

    More seeds do not help: the version 5 and version 6 runs of each model (different seeds and fold splits) have rank correlation 0.998 to 0.999, and averaging them changes OOF KS by -0.0007 to +0.0006 (`eda/68`). The trees and TabM therefore keep 3 seeds, RealMLP gets 1, and TabPFN keeps 4 estimators, the setting whose key-value cache is known to fit a T4 at the full 56,000-row context.

    `HIST_WEIGHT` is the sample weight of a historical row. The offline counting test (`eda/83`) preferred 0.5 over 1.0, because the e-commerce part of their risk mapping differs from the selected loans. `EXAONE_CONTEXT` is 12,000 rows, the largest context that ran on a T4 in version 7 (`detected VRAM cannot fit` at 56,000). `DEADLINE` is the single time limit of the run, and `minutes_left()` is checked before every fold of every model.
"""))

cells.append(C("""
    def find_data_dir():
        if Path('/kaggle/input').exists():
            return next(Path('/kaggle/input').rglob('train_applications.csv')).parent
        for cand in [Path('data'), Path('../data'), Path('/workspace/data'), *Path('/workspace').glob('*/data')]:
            if (cand / 'train_applications.csv').exists():
                return cand.resolve()
        raise FileNotFoundError('put the competition CSV files in a data/ folder next to the notebook')

    class Settings:
        SEED       = 42
        GBDT_SEEDS = [42, 7, 2026]
        N_FOLDS    = 5
        _ON_KAGGLE = Path('/kaggle/input').exists()

        DATA_DIR   = find_data_dir()
        OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else DATA_DIR.parent / 'output_v8'
        FIG_DIR    = OUTPUT_DIR / 'figures'

        USE_LGB     = True
        USE_XGB     = True
        USE_CAT     = True
        USE_HIST    = True
        USE_TABPFN  = True
        USE_EXAONE  = True
        USE_TABM    = True
        USE_CAUSILO = True
        USE_TABICL  = False  # killed the kernel in the first v8 run: OOM on the 44k-row query at every context size
        USE_REALMLP = True

        HIST_WEIGHT = 0.5

        RUN_LIMIT_MIN     = 12 * 60 - 45
        DEADLINE          = time.time() + RUN_LIMIT_MIN * 60
        BLEND_RESERVE_MIN = 20

        N_GPU          = torch.cuda.device_count()
        GPUS           = [f'cuda:{i}' for i in range(N_GPU)]
        FM_N_FEATURES  = 100
        FM_MAX_CONTEXT = 60000
        FM_N_EST       = 4
        FM_CHUNK       = 10000
        EXAONE_CONTEXT = 12000
        TABICL_N_EST   = 8
        REALMLP_SEEDS  = [42]
        TABM_EPOCHS   = 60
        TABM_PATIENCE = 8
        TABM_BATCH    = 512
        TABM_BINS     = 48
        TABM_LR       = 2e-3
        TABM_WD       = 3e-4
        ENS_ITERS      = 30
        ENS_BAGS       = 20

        LGB_PARAMS = dict(
            objective='binary', metric='auc', learning_rate=0.02, num_leaves=31,
            min_child_samples=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, verbose=-1, n_jobs=max(os.cpu_count() or 4, 4),
        )
        LGB_SHALLOW = dict(LGB_PARAMS, num_leaves=7, max_depth=3, min_child_samples=200,
                           feature_fraction=0.4, lambda_l2=10.0)
        XGB_PARAMS = dict(
            objective='binary:logistic', eval_metric='auc', learning_rate=0.02, max_depth=5,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0,
            tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        XGB_SHALLOW = dict(XGB_PARAMS, max_depth=3, min_child_weight=20, colsample_bytree=0.4, reg_lambda=10.0)
        CAT_PARAMS = dict(
            iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=5, loss_function='Logloss',
            task_type='GPU' if torch.cuda.is_available() else 'CPU', od_type='Iter', od_wait=200,
            use_best_model=True, verbose=0,
        )
        CAT_SHALLOW = dict(CAT_PARAMS, depth=4, l2_leaf_reg=10)
        N_ROUNDS   = 10000
        EARLY_STOP = 200

    CFG = Settings()
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)

    def minutes_left():
        return (CFG.DEADLINE - time.time()) / 60

    print('data:', CFG.DATA_DIR, '| GPUs:', CFG.GPUS, f'| {minutes_left():.0f} minutes until the run deadline')
"""))

cells.append(M("""
    The TabPFN-3.5 weights are gated. The token comes from the `HF_TOKEN` or `TABPFN_TOKEN` environment variable, or from Kaggle secrets when the notebook runs there. A missing token is not an error, but TabPFN is then skipped.
"""))

cells.append(C("""
    if UserSecretsClient is not None and CFG._ON_KAGGLE:
        for name in ['TABPFN_TOKEN', 'HF_TOKEN']:
            try:
                os.environ[name] = UserSecretsClient().get_secret(name)
                print('loaded secret', name)
            except Exception:
                pass
    if 'TABPFN_TOKEN' not in os.environ and 'HF_TOKEN' in os.environ:
        os.environ['TABPFN_TOKEN'] = os.environ['HF_TOKEN']
"""))

cells.append(M("""
    ## Load Dataset

    All seven tables are read with Polars and dates are parsed on load. No visualisation here.
"""))

cells.append(C("""
    TABLES = ['train_applications', 'test_applications', 'customer_profiles', 'previous_applications',
              'repayment_history', 'account_behavior', 'bureau_observations', 'ecommerce_transactions']
    T = {t: pl.read_csv(CFG.DATA_DIR / f'{t}.csv', try_parse_dates=True, infer_schema_length=100000) for t in TABLES}
    sample_sub = pl.read_csv(CFG.DATA_DIR / 'sample_submission.csv')
    for t, df in T.items():
        print(f'{t:<24} {df.shape}')
"""))

# ---------------------------------------------------------------------------
# Section 3: Exploratory Data Analysis
# ---------------------------------------------------------------------------
cells.append(section("Exploratory Data Analysis", "3"))

cells.append(M("""
    The analysis below reproduces the key findings of the offline EDA that shaped the feature set. All statistics use the competition KS definition, so the helpers are defined first and reused throughout the notebook.
"""))

cells.append(M("""
    ## KS Utilities

    `ks_score` matches the official scorer (`ks_2samp` on bad versus good scores). `ks_nan` treats missing values as their own bin at either extreme, which mirrors how tree models route missing values. `psi` measures train-to-test drift on train deciles.
"""))

cells.append(C("""
    def ks_score(y, s):
        y, s = np.asarray(y), np.asarray(s, dtype=float)
        return float(ks_2samp(s[y == 1], s[y == 0]).statistic)

    def ks_nan(y, s):
        s = np.asarray(s, dtype=float)
        if np.isnan(s).all():
            return 0.0
        lo, hi = np.nanmin(s) - 1, np.nanmax(s) + 1
        return max(ks_score(y, np.where(np.isnan(s), lo, s)), ks_score(y, np.where(np.isnan(s), hi, s)))

    def psi(a, b, bins=10):
        a, b = np.asarray(a, float), np.asarray(b, float)
        qs = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1))) if (~np.isnan(a)).any() else np.array([0.0])
        def dist(x):
            idx = np.where(np.isnan(x), -1, np.searchsorted(qs[1:-1], x, side='right'))
            cnt = np.array([(idx == k).sum() for k in range(-1, len(qs))], float) + 0.5
            return cnt / cnt.sum()
        pa, pb = dist(a), dist(b)
        return float(((pb - pa) * np.log(pb / pa)).sum())
"""))

cells.append(M("""
    ## Target and Time

    The target rate and the monthly volume of both splits show whether a time-based split is required.
"""))

cells.append(C("""
    tr_raw, te_raw = T['train_applications'], T['test_applications']
    print(f"train bad rate: {tr_raw['target'].mean():.4f}  (n_bad={tr_raw['target'].sum()})")
    m = pl.concat([
        tr_raw.select(ym=pl.col('decision_datetime').dt.strftime('%Y-%m'), target='target', split=pl.lit('train')),
        te_raw.select(ym=pl.col('decision_datetime').dt.strftime('%Y-%m'), target=pl.lit(None, pl.Int64), split=pl.lit('test')),
    ])
    tm = m.group_by('ym', 'split').agg(n=pl.len(), rate=pl.col('target').mean()).sort('ym')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    p = tm.filter(pl.col('split') == 'train')
    ax[0].bar(p['ym'], p['rate']); ax[0].set_title('train bad rate by decision month'); ax[0].tick_params(axis='x', rotation=60)
    for s in ['train', 'test']:
        q = tm.filter(pl.col('split') == s)
        ax[1].plot(q['ym'], q['n'] / q['n'].sum(), marker='o', label=s)
    ax[1].legend(); ax[1].set_title('share of applications by month'); ax[1].tick_params(axis='x', rotation=60)
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'target_time.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > The bad rate is 13.6% and stays between 12% and 15% in every month of 2025. Train and test applications are interleaved over the same twelve months in the same 70/30 proportion, so the test set is a random sample of the same period rather than an out-of-time holdout.

    > Stratified K-fold therefore mimics the leaderboard split, and no time-based validation is needed.
"""))

cells.append(M("""
    ## Risk-Based Pricing

    The interest rate takes only five values. If the lender priced loans from a risk grade, the rate itself carries that grade.
"""))

cells.append(C("""
    pricing = (tr_raw.group_by('annual_interest_rate')
               .agg(n=pl.len(), bad_rate=pl.col('target').mean()).sort('annual_interest_rate'))
    print(pricing)
    print('univariate KS of annual_interest_rate:', round(ks_score(tr_raw['target'], tr_raw['annual_interest_rate']), 4))
"""))

cells.append(M("""
    #### Insights

    > The bad rate rises monotonically from 5.5% at a 16% rate to 50% at a 32% rate, a univariate KS of 0.23. The offline analysis shows that the mean bureau score falls by about 40 points per tier, so the rate is largely a coarse copy of the bureau score. Inside bureau-score deciles the rate keeps only about 0.06 KS.
"""))

cells.append(M("""
    ## Pre-Application Spending Burst

    E-commerce orders are counted per customer per week before the decision, separately for bad and good loans. The shape of this curve sets the aggregation windows.
"""))

cells.append(C("""
    o = (T['ecommerce_transactions'].join(tr_raw.select('customer_id', 'decision_datetime', 'target'), on='customer_id')
         .group_by('order_id').agg(pl.col('customer_id', 'decision_datetime', 'transaction_datetime', 'target').first())
         .with_columns(wk=((pl.col('decision_datetime') - pl.col('transaction_datetime')).dt.total_days() // 7)))
    ncls = tr_raw.group_by('target').agg(ncust=pl.len())
    wk = (o.filter(pl.col('wk') < 52).group_by('wk', 'target').agg(n=pl.len()).join(ncls, on='target')
          .with_columns(per=pl.col('n') / pl.col('ncust')).sort('wk'))
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in (0, 1):
        q = wk.filter(pl.col('target') == k)
        ax.plot(q['wk'], q['per'], label=f'target={k}')
    ax.invert_xaxis(); ax.set_xlabel('weeks before decision'); ax.set_ylabel('orders per customer'); ax.legend()
    ax.set_title('E-commerce orders per customer per week')
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'ecom_burst.png', dpi=90); plt.show()
    r = wk.pivot(on='target', index='wk', values='per').sort('wk').head(6)
    print(r.with_columns(bad_over_good=pl.col('1') / pl.col('0')))
"""))

cells.append(M("""
    #### Insights

    > Bad and good customers order at the same rate until about 16 weeks before the decision. After that, bad customers accelerate and reach 2.3 times the good rate in the final week. Lifetime totals dilute this signal, so the features use 7 to 90 day windows and an acceleration ratio against a 17-52 week baseline.

    > Account behaviour shows the same pattern. The outflow-to-inflow ratio and low-balance days of bad customers deteriorate over the last 6 months, although they look slightly healthier than good customers a year earlier. Recent-versus-baseline deltas capture this.
"""))

cells.append(M("""
    ## Lateness Severity

    Repayment lateness has two sides: how often a customer is late, and how late the payment is when it is late. Days late among paid-late installments are compared between bad and good customers.
"""))

cells.append(C("""
    rp = (T['repayment_history'].join(tr_raw.select('customer_id', 'target'), on='customer_id')
          .filter(pl.col('amount_paid') >= pl.col('amount_due'))
          .with_columns(late_d=(pl.col('last_payment_datetime') - pl.col('due_date')).dt.total_days())
          .filter(pl.col('late_d') > 0))
    print(rp.group_by('target').agg([pl.col('late_d').quantile(q).alias(f'q{int(q * 100)}') for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                                    + [pl.len().alias('n_late_installments')]).sort('target'))
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in (0, 1):
        ax.hist(rp.filter(pl.col('target') == k)['late_d'].clip(upper_bound=150), bins=75, density=True, alpha=0.5, label=f'target={k}')
    ax.set_xlabel('days late (paid-late installments)'); ax.legend(); ax.set_title('Lateness comes in whole missed months')
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'late_severity.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > Late payments cluster at about 30, 61 and 91 days. A late installment is a skipped month that is settled at a following due date. The median good customer catches up after one month and the median bad customer after two. Severity is therefore a separate measurement from the frequency features of version 1 (Spearman correlation about 0.63). `sv_maxdpd_mean_late` and `sv_share30_of_late` capture it.

    > The offline residual analysis of the version 1 predictions also found smaller orthogonal signals: the first installment of each loan, the latest loan, the share of inflows that are not income, the bureau score trend over all pulls, and cancelled and night-time orders before the application. Each one moves the cross-fitted KS of version 1 by only 0.001 to 0.003. They are added together as the version 2 features.
"""))

cells.append(M("""
    ## Price Premium

    Spending totals mix three things: how often a customer orders, how many units they buy, and how expensive each unit is. Here the unit price of every item is compared with the typical price of its category on a log scale (z-score of log unit price within the category). The customer's average z is their price premium. This exploratory view uses one pooled reference; the model features use a stricter one, built only from purchases dated before each application's decision month (see the leakage test below).
"""))

cells.append(C("""
    e = (T['ecommerce_transactions'].join(tr_raw.select('customer_id', 'decision_datetime', 'target'), on='customer_id')
         .with_columns(lu=(pl.col('item_total') / pl.col('quantity')).log(),
                       blk=((pl.col('decision_datetime') - pl.col('transaction_datetime')).dt.total_days() // 28)))
    e = e.with_columns(z=(pl.col('lu') - pl.col('lu').mean().over('product_category')) / pl.col('lu').std().over('product_category'))
    prem = e.group_by('customer_id', 'target').agg(z=pl.col('z').mean())
    print('KS of the customer price premium:', round(ks_score(prem['target'], prem['z']), 4))
    print(e.group_by('product_category').agg(sd_log_price=pl.col('lu').std(), median_unit_price=(pl.col('lu').median().exp() / 1000).round() * 1000).sort('product_category'))
    prof = e.filter(pl.col('blk') < 18).group_by('blk', 'target').agg(z=pl.col('z').mean()).sort('blk')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k in (0, 1):
        ax[0].hist(prem.filter(pl.col('target') == k)['z'], bins=60, alpha=0.5, density=True, label=f'target={k}')
        q = prof.filter(pl.col('target') == k)
        ax[1].plot(q['blk'], q['z'], marker='o', label=f'target={k}')
    ax[0].set_title('customer mean price z'); ax[0].legend()
    ax[1].invert_xaxis(); ax[1].set_xlabel('4-week blocks before decision'); ax[1].set_title('price z over time'); ax[1].legend()
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'price_premium.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > Bad customers pay more per unit than good customers in every category, and the gap is visible more than a year before the application (about 0.16 standard deviations) and widens to about 0.35 near the decision. The customer price premium alone reaches a KS of about 0.37, above the best repayment feature. Log unit prices have the same spread (sd about 0.69) in all seven categories, so one category-free z-score is meaningful.

    > The version 3 model could not see this. Its spending features add up order totals, which mixes unit price with basket size, and the residual analysis showed that version 3 even over-reads basket quantity as risk: the bad rate is flat across quantity levels while the version 3 prediction rises with them. The version 5 features therefore separate unit price (`pp_*`, overall, completed orders, recent and old windows, per category) from quantity per item (`pp_qty_item*`). The decision month is added as well, because the bad rate moves between 12.0% in May and 14.7% in February and the test set covers the same months.
"""))

cells.append(M("""
    ## Table Coverage

    Coverage of each auxiliary table is compared between splits, because a coverage mismatch would make the CV score unrepresentative.
"""))

cells.append(C("""
    split = pl.concat([tr_raw.select('customer_id', split=pl.lit('train')), te_raw.select('customer_id', split=pl.lit('test'))])
    cov = []
    for t in ['previous_applications', 'repayment_history', 'account_behavior', 'bureau_observations', 'ecommerce_transactions']:
        c = split.with_columns(has=pl.col('customer_id').is_in(T[t]['customer_id'].unique().implode()))
        cov.append(c.group_by('split').agg(pl.col('has').mean()).with_columns(table=pl.lit(t)))
    print(pl.concat(cov).pivot(on='split', index='table', values='has'))
"""))

cells.append(M("""
    #### Insights

    > Bureau and e-commerce data exist for every customer, while previous applications and account data exist for exactly 70% of customers in both splits (66% for repayment). The remaining 30% are thin-file customers, whose bad rate is lower (11.7% versus 14.4%).

    > The offline audit also confirms that every auxiliary row is already dated before its customer's `decision_datetime`. There are no exact or near-duplicate customers across splits, and the hashed IDs carry no signal (KS < 0.01). The feature code still filters on the cutoff explicitly and is tested against injected future rows below.
"""))

cells.append(M("""
    ## Offline Audit of Versions 6 and 7

    Versions 5, 6 and 7 scored the same within noise. The table summarises the offline analyses of their out-of-fold predictions (`eda/60` to `eda/72` and an independent review in `eda/v7_review`, run on a laptop CPU without training any model). Each candidate signal was tested against the residual of the version 6 blend with a noise threshold from random features. These screens are exploratory: they reuse labels that also chose the features, so a negative result means "not detected at this resolution", not "absent".

    | Question | Result |
    |---|---|
    | Did versions 6 and 7 change anything? | Public 0.5612 vs 0.5615 (v5). Paired bootstrap of OOF KS: v7 - v6 = -0.0013 [-0.0032, +0.0013]; test rank correlation v6/v7 0.9993. No real change. |
    | Joint row-level signal in any child table? | Naive-Bayes scores from per-row signatures of every table: 0 of 25 above noise. |
    | Hidden lender assessment, date or macro effects? | 0 of 6 above noise; weekly observed minus expected bad rate has sd 0.0101 against a binomial floor of 0.0095. |
    | Ratios, interactions, native DFS? | 0 of 1,138 ratios and interaction grids replicate on both halves; 0 of 297 native TabPFN-Rel DFS columns above noise. |
    | Do many weak candidates add up? | Bundles of the top 5 to 60 of 1,228 candidates gain no more held-out KS than bundles of pure noise. |
    | Do more seeds help? | v5 and v6 runs of the same model correlate 0.999; their average changes KS by -0.0007 to +0.0006. |
    | Can the price premium be measured better? | Unit prices are continuous (rounded to IDR 1,000), with no product catalogue to remove. Order-equal mean 0.375, median 0.346, shrinkage 0.372 against 0.373 for the plain mean. |
    | Is AUC a safe proxy for KS? | No: five model pairs order differently by AUC and by KS, and v7 has a higher AUC but a lower KS than v6. |
    | EXAONE? | On identical query rows, 4,000 context rows scored 0.5437 against 0.5467 for the deep LightGBM trained on 56,000 rows; 12,000 context rows scored 0.5120 against 0.5189. Competitive with far less data. The 56,000-row run failed on T4 memory. |

    #### Insights

    > The feature screens no longer find new columns at the resolution they can detect (a cross-fitted KS gain of about 0.002 per feature). Version 8 therefore looks for new training rows rather than new columns, and widens the model side with members the earlier versions did not have.
"""))

cells.append(M("""
    ## Historical Loans as Extra Labelled Data

    The target asks whether a 12-month loan reaches 90 DPD within 12 months of origination. Many customers already had loans before the selected one, and for an approved previous loan whose 12-month window ended before the selected decision, the repayment history shows whether that loan reached 90 DPD inside its window. These outcomes are dated before the cutoff, so they are legitimate information, and they exist for train and test customers alike.

    The helper code below (from `eda/hist_snapshots.py`) labels those loans and later rebuilds their features as of each loan's decision time. `repayment_history` is a snapshot taken at the selected cutoff: an installment that was due before an earlier loan's decision but paid after it still shows its final payment. `snapshot_tables` therefore resets such installments to their state at that time (unpaid, DPD = floor(T - due)), which is exactly how the dataset reports unpaid installments at the cutoff.
"""))

_hist = HIST_SRC.read_text(encoding="utf-8")
_hist = re.sub(r'^"""[\s\S]*?"""\n', "", _hist)
_hist = _hist.replace("import polars as pl\n", "").strip("\n")
cells.append(code(_hist))

cells.append(C("""
    hist_lab = hist_labels(T)
    print(f"labelled historical loans: {hist_lab.height:,} from {hist_lab['customer_id'].n_unique():,} customers, "
          f"bad rate {hist_lab['hist_target'].mean():.4f} (selected loans {tr_raw['target'].mean():.4f})")
    by_rate = (hist_lab.group_by('annual_interest_rate').agg(n_hist=pl.len(), bad_hist=pl.col('hist_target').mean())
               .join(tr_raw.group_by('annual_interest_rate').agg(n_sel=pl.len(), bad_sel=pl.col('target').mean()), on='annual_interest_rate')
               .sort('annual_interest_rate'))
    print(by_rate)
    print(hist_lab.group_by('term_months').agg(n=pl.len(), bad=pl.col('hist_target').mean()).sort('term_months'))
    link = (tr_raw.join(hist_lab.group_by('customer_id').agg(n_bad=pl.col('hist_target').sum()), on='customer_id', how='left')
            .group_by(grp=pl.col('n_bad').clip(upper_bound=2).cast(pl.String).fill_null('none')).agg(n=pl.len(), bad_rate=pl.col('target').mean()).sort('grp'))
    print(link)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(by_rate['annual_interest_rate'], by_rate['bad_hist'], 'o-', label='historical loans')
    ax[0].plot(by_rate['annual_interest_rate'], by_rate['bad_sel'], 's-', label='selected loans')
    ax[0].set_xlabel('annual interest rate tier'); ax[0].set_title('bad rate by price tier'); ax[0].legend()
    ax[1].bar(link['grp'], link['bad_rate']); ax[1].set_xlabel('bad historical loans of the customer'); ax[1].set_title('selected-loan bad rate')
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'historical_labels.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > There are 81,702 labelled historical loans, more than the 70,000 training applications, with a bad rate of 14.4%. Their bad rate rises with the price tier like the selected loans, though less steeply (8% to 36% against 5.5% to 50%), and a customer with one bad historical loan has a 34% bad rate on the selected loan against 9% with none. The two populations share most of their risk mechanism but not all of it, which is why the historical rows carry an `is_hist` flag and a lower weight.
"""))

# ---------------------------------------------------------------------------
# Section 4: Feature Engineering
# ---------------------------------------------------------------------------
cells.append(section("Feature Engineering", "4"))

cells.append(M("""
    Each auxiliary table is aggregated to one row per customer, and every builder filters on the customer's own cutoff (`decision_datetime`). Bureau reports must be *available* before the cutoff, account months must be *complete* before it, and e-commerce `order_total` is de-duplicated per order before it is summed, because it repeats on every item row.
"""))

cells.append(M("""
    ## Feature Builders

    The builders cover application and profile affordability ratios, bureau scores per provider and their change since earlier pulls, repayment lateness depth and active obligations, cash-flow levels and recent deltas, e-commerce recency and acceleration, and previous application outcomes and pricing.
"""))

_feat = FEATURES_SRC.read_text(encoding="utf-8")
_feat = re.sub(r'^"""[\s\S]*?"""\n', "", _feat)
_feat = re.sub(r'\n\s+"""[^\n]*"""', "", _feat)
_feat = _feat.replace("import polars as pl\n", "").strip("\n")
cells.append(code(_feat))

cells.append(C("""
    t0 = time.time()
    feats = build_all(T)
    assert feats['application_id'].n_unique() == feats.height == T['train_applications'].height + T['test_applications'].height
    print(f'feature matrix {feats.shape} built in {time.time() - t0:.1f}s')
"""))

cells.append(M("""
    ## Leakage Injection Test

    Fake rows dated after each customer's cutoff are appended to all five dated tables, using extreme values: a DPD of 999, a cancelled 1e9 order, a score of 300 with 99 inquiries, the incomplete decision month with zero income, and an approved 1e9 loan decided after the cutoff. The comparison allows a 1e-9 relative tolerance, because sums of floats can differ in the last bits when the row order changes.

    The price reference needs a precise rule. A purchase made after customer A's cutoff is in the past for a customer deciding in a later month, so it may legitimately enter that customer's category reference. The test therefore has two parts. First, rows after the cutoffs of 2,000 random customers must not move any non-price feature of anyone. Second, rows after the cutoffs of 2,000 December customers (which are in nobody's past) must not move any feature at all, which checks the price features' own cutoff as well.
"""))

cells.append(C("""
    apps_all = pl.concat([T['train_applications'], T['test_applications']], how='diagonal_relaxed').select('customer_id', cut='decision_datetime')
    def first_rows(t):
        return T[t].join(fut, on='customer_id').group_by('customer_id').first()
    def moved(ref, leak, cols):
        ref, leak = ref.sort('application_id'), leak.sort('application_id')
        return [c for c in cols if not np.allclose(ref[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(),
                                                   leak[c].cast(pl.Float64).fill_nan(None).fill_null(-7.0).to_numpy(), rtol=1e-9, atol=1e-9)]
    num_cols = [c for c, d in feats.schema.items() if d.is_numeric()]
    fut = apps_all.sample(2000, seed=1)
    after = pl.col('cut') + pl.duration(days=5)
    inj = {
        'repayment_history': first_rows('repayment_history').with_columns(
            due_date=after, max_days_past_due=pl.lit(999), days_past_due=pl.lit(999)),
        'ecommerce_transactions': first_rows('ecommerce_transactions').with_columns(
            transaction_datetime=pl.col('cut') + pl.duration(hours=1), order_id=pl.lit('O_FAKE') + pl.col('customer_id'),
            order_total=pl.lit(10**9), order_status=pl.lit('cancelled')),
        'bureau_observations': first_rows('bureau_observations').with_columns(
            available_datetime=after, score=pl.lit(300.0), inquiries_last_6m=pl.lit(99)),
        'account_behavior': first_rows('account_behavior').with_columns(
            month=pl.col('cut').dt.truncate('1mo').cast(pl.Date), outflows=pl.lit(10**10), income_like_credits=pl.lit(0)),
        'previous_applications': first_rows('previous_applications').with_columns(
            application_datetime=pl.col('cut') - pl.duration(hours=1), decision_datetime=after, decision=pl.lit('approved'),
            approved_amount=pl.lit(10**9), term_months=pl.lit(6),
            application_id=pl.lit('A_FAKE') + pl.col('customer_id'), loan_id=pl.lit('L_FAKE') + pl.col('customer_id')),
    }
    T_leak = dict(T)
    for t, extra in inj.items():
        T_leak[t] = pl.concat([T[t], extra.select(T[t].columns)], how='vertical_relaxed')
    changed = moved(feats, build_all(T_leak), [c for c in num_cols if not c.startswith('pp_')])
    assert not changed, f'cutoff leak: {changed}'
    print(f'part 1 passed: no non-price feature of any customer moved ({len(num_cols)} numeric features checked)')
    fut = apps_all.filter(pl.col('cut').dt.month() == 12).sample(2000, seed=2)
    ec = first_rows('ecommerce_transactions').with_columns(
        transaction_datetime=pl.col('cut') + pl.duration(hours=1), order_id=pl.lit('O_FAKE') + pl.col('customer_id'),
        item_total=pl.lit(10**9), quantity=pl.lit(1), product_category=pl.lit('groceries'))
    T_leak = dict(T)
    T_leak['ecommerce_transactions'] = pl.concat([T['ecommerce_transactions'], ec.select(T['ecommerce_transactions'].columns)], how='vertical_relaxed')
    changed = moved(feats, build_all(T_leak), num_cols)
    assert not changed, f'cutoff leak (December test): {changed}'
    print('part 2 passed: rows after December cutoffs moved no feature, including the price premium')
    del T_leak
    _ = gc.collect()
"""))

cells.append(M("""
    ## Feature Diagnostics

    Every numeric feature is scored by univariate KS against the target and by PSI between train and test. A high-KS feature that also drifts would split public and private scores, so both are checked together.
"""))

cells.append(C("""
    ID_COLS  = ['application_id', 'customer_id', 'decision_datetime', 'target', 'is_train']
    CAT_COLS = ['loan_purpose', 'region', 'employment_type', 'housing_status']
    FEATURES = sorted(c for c in feats.columns if c not in ID_COLS)
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    tr_f, te_f = feats.filter(pl.col('is_train') == 1), feats.filter(pl.col('is_train') == 0)
    y = tr_f['target'].to_numpy()
    diag = pl.DataFrame({
        'feature': NUM_COLS,
        'ks': [ks_nan(y, tr_f[c].cast(pl.Float64).to_numpy()) for c in NUM_COLS],
        'psi': [psi(tr_f[c].cast(pl.Float64).to_numpy(), te_f[c].cast(pl.Float64).to_numpy()) for c in NUM_COLS],
        'coverage': [1 - tr_f[c].null_count() / tr_f.height for c in NUM_COLS],
    }).sort('ks', descending=True)
    V2_PREFIX = ('bz_', 'ob_', 'rq_', 'sv_', 'cf_', 'ec2_')
    V5_PREFIX = ('pp_', 'dec_month')
    V2_FEATURES = [c for c in FEATURES if c.startswith(V2_PREFIX)]
    V5_FEATURES = [c for c in FEATURES if c.startswith(V5_PREFIX)]
    V3_FEATURES = [c for c in FEATURES if not c.startswith(V5_PREFIX)]
    with pl.Config(tbl_rows=25, float_precision=4):
        print(diag.head(25))
        print('version 2 features:')
        print('version 5 features:')
        print(diag.filter(pl.col('feature').is_in(V5_FEATURES)))
    print('max PSI over all features:', round(diag['psi'].max(), 5))
    print(f'{len(FEATURES)} features ({len(CAT_COLS)} categorical), {len(V5_FEATURES)} new in version 5')
"""))

cells.append(M("""
    #### Insights

    > The strongest single features are repayment lateness (`rp_frac_dpd1`, KS about 0.32), e-commerce acceleration (`ec_accel_30d`, about 0.31) and the mean bureau score (about 0.25). The first two are nearly orthogonal to the bureau score and keep about 0.30 KS inside score deciles, which is why combining sources gives a large lift. A rank average of nine such features with no training already reaches a KS of about 0.41.

    > The version 2 features have univariate KS up to about 0.27, but they were chosen for what they add on top of version 1, not for their stand-alone KS. The maximum PSI is far below the usual 0.1 alert level, so no feature needs to be dropped for stability.
"""))

cells.append(M("""
    ## Additivity of Risk Signals

    The bad rate is tabulated over quintiles of two signal families, with missing values as a sixth bin. If the lines are parallel on the log-odds scale, the families add up and deep interactions are not needed, which favours shallow trees.
"""))

cells.append(C("""
    def logit_lines(ax, a, b, q=5):
        d = tr_f.select(a, b, 'target').with_columns(
            qa=(pl.col(a).rank('ordinal') * q // (pl.col(a).count() + 1)).fill_null(q),
            qb=(pl.col(b).rank('ordinal') * q // (pl.col(b).count() + 1)).fill_null(q))
        g = (d.group_by('qa', 'qb').agg(n=pl.len(), bad=pl.col('target').sum()).filter(pl.col('n') >= 100)
              .with_columns(logit=((pl.col('bad') + 1) / (pl.col('n') - pl.col('bad') + 1)).log()).sort('qa', 'qb'))
        for k in range(q + 1):
            s = g.filter(pl.col('qb') == k)
            if s.height:
                ax.plot(s['qa'], s['logit'], 'o-', label=f'{b} q{k}' if k < q else f'{b} missing')
        ax.set_xlabel(f'{a} quintile ({q} = missing)'); ax.set_ylabel('logit bad rate'); ax.legend(fontsize=6)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for i, (a, b) in enumerate([('rp_frac_dpd1', 'bc_score_mean'), ('ec_orders_30d', 'bc_score_mean'), ('ec_orders_30d', 'rp_frac_dpd1')]):
        logit_lines(ax[i], a, b)
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'additivity.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > The lines are close to parallel, so repayment, e-commerce and bureau risk combine additively in log-odds. The offline interaction test confirms that the version 1 model already captured the remaining non-additive part: no pair of families left more than noise-level structure in its residuals. The main departure is that e-commerce is steeper for thin-file customers (the missing repayment line), and a depth-3 tree can still express that. Shallow variants with stronger regularisation are therefore added next to the deep version 1 settings.
"""))

cells.append(M("""
    ## Historical Loan Snapshots

    Every labelled historical loan becomes one feature row built by the same `build_all`, as of that loan's decision time. Customers have up to seven such loans, so `build_hist` calls `build_all` once per loan order, each time with one snapshot per customer. Two checks guard the rebuild:

    - **Identity.** Reconstructing every customer's tables at T = the selected cutoff must reproduce the production feature matrix exactly, because the repayment reset is then a no-op. This tests the reset rule against the dataset's own snapshot semantics.
    - **Future invariance.** For the second-order snapshots (the first have no earlier installments), every payment made after T becomes an early full payment, and bureau, order, account and previous-application rows dated after T are added. No feature may move. As in the production test, the extra rows are added only for snapshots decided in the last month, so that no other snapshot's category price reference may legitimately include them.
"""))

cells.append(C("""
    t0 = time.time()
    hist_feats = build_hist(T, build_all) if CFG.USE_HIST else None
    print(f'historical feature rows {hist_feats.shape} built in {time.time() - t0:.1f}s, bad rate {hist_feats["target"].mean():.4f}')
    print(hist_feats.group_by('snap_order').agg(n=pl.len(), bad=pl.col('target').mean()).sort('snap_order'))

    apps_sel = pl.concat([T['train_applications'].drop('target'), T['test_applications']], how='diagonal_relaxed')
    snap = apps_sel.select('customer_id', sel_cutoff='decision_datetime', T='decision_datetime', snap_app=pl.lit('none'))
    T_id = snapshot_tables(T, snap)
    T_id['train_applications'], T_id['test_applications'] = T['train_applications'], T['test_applications']
    changed = moved(feats, build_all(T_id), num_cols)
    assert not changed, f'identity check failed: {changed}'
    print(f'identity check passed: all {len(num_cols)} numeric features identical at the selected cutoff')

    lab2 = hist_lab.with_columns(k=pl.col('decision_datetime').rank('ordinal').over('customer_id')).filter(pl.col('k') == 2)
    s2 = lab2.select('customer_id', 'sel_cutoff', T='decision_datetime', snap_app='application_id')
    def snap_build(tables):
        tk = snapshot_tables(tables, s2)
        tk['train_applications'] = snapshot_apps(lab2, T['train_applications'].schema)
        tk['test_applications'] = T['test_applications'].clear()
        return build_all(tk)
    ref2 = snap_build(T)
    T_fut = dict(T)
    r = T['repayment_history'].join(s2.select('customer_id', 'T'), on='customer_id', how='left')
    paid_after = pl.col('T').is_not_null() & (pl.col('last_payment_datetime') > pl.col('T'))
    T_fut['repayment_history'] = r.with_columns(
        amount_paid=pl.when(paid_after).then(pl.col('amount_due')).otherwise(pl.col('amount_paid')),
        last_payment_datetime=pl.when(paid_after).then(pl.col('T') + pl.duration(days=1)).otherwise(pl.col('last_payment_datetime')),
        max_days_past_due=pl.when(paid_after).then(0).otherwise(pl.col('max_days_past_due')),
        days_past_due=pl.when(paid_after).then(0).otherwise(pl.col('days_past_due'))).drop('T')
    fut = s2.filter(pl.col('T').dt.truncate('1mo') == s2['T'].dt.truncate('1mo').max()).select('customer_id', 'T')
    def first_after(t):
        return T[t].join(fut, on='customer_id').group_by('customer_id').first()
    later = pl.col('T') + pl.duration(days=3)
    extra = {
        'bureau_observations': first_after('bureau_observations').with_columns(available_datetime=later, report_datetime=later,
                                                                               score=pl.lit(300.0), inquiries_last_6m=pl.lit(99)),
        'ecommerce_transactions': first_after('ecommerce_transactions').with_columns(
            transaction_datetime=later, item_total=pl.lit(10**9), order_id=pl.lit('O_FUT') + pl.col('customer_id')),
        'account_behavior': first_after('account_behavior').with_columns(month=pl.col('T').dt.truncate('1mo').cast(pl.Date),
                                                                         outflows=pl.lit(10**10)),
        'previous_applications': first_after('previous_applications').with_columns(
            application_datetime=later, decision_datetime=later, decision=pl.lit('rejected'),
            application_id=pl.lit('A_FUT') + pl.col('customer_id'), loan_id=pl.lit('L_FUT') + pl.col('customer_id')),
    }
    for t, ex in extra.items():
        T_fut[t] = pl.concat([T_fut[t], ex.select(T[t].columns)], how='vertical_relaxed')
    changed = moved(ref2, snap_build(T_fut), num_cols)
    assert not changed, f'historical snapshot leak: {changed}'
    print(f'future invariance passed on {s2.height:,} second-order snapshots '
          f'({r.filter(paid_after & (pl.col("due_date") <= pl.col("T"))).height:,} installments paid after T were reset)')
    del T_id, T_fut, ref2
    _ = gc.collect()
"""))

cells.append(M("""
    The next cell compares every feature's univariate KS on the selected loans with its KS on the historical loans, each on the rows where the feature is observed, so that the shorter history of the 2023 and 2024 snapshots does not drive the comparison.
"""))

cells.append(C("""
    def ks_obs(yy, x):
        ok = np.isfinite(x)
        return ks_score(yy[ok], x[ok]) if ok.sum() > 500 and 0 < yy[ok].mean() < 1 and np.unique(x[ok]).size > 1 else np.nan

    y_hist = hist_feats['target'].to_numpy()
    as_float = lambda d, c: d[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float)
    tk = pl.DataFrame({'feature': NUM_COLS,
                       'ks_selected': [ks_obs(y, as_float(tr_f, c)) for c in NUM_COLS],
                       'ks_hist': [ks_obs(y_hist, as_float(hist_feats, c)) for c in NUM_COLS],
                       'coverage_hist': [1 - hist_feats[c].null_count() / hist_feats.height for c in NUM_COLS]})
    with pl.Config(tbl_rows=15, float_precision=3):
        print(tk.sort('ks_selected', descending=True, nulls_last=True).head(15))
    fam = tk.with_columns(family=pl.col('feature').str.extract(r'^([a-z]+\\d?)_', 1)).group_by('family').agg(
        ks_selected=pl.col('ks_selected').max(), ks_hist=pl.col('ks_hist').max()).sort('ks_selected', descending=True, nulls_last=True)
    with pl.Config(tbl_rows=10, float_precision=3):
        print(fam.head(10))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(tk['ks_selected'], tk['ks_hist'], s=8); ax.plot([0, 0.45], [0, 0.45], 'k--', lw=0.5)
    ax.set_xlabel('KS on selected loans'); ax.set_ylabel('KS on historical loans'); ax.set_title('per-feature KS, observed rows')
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'hist_transfer.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > Repayment, bureau and pricing features keep most of their KS on the historical loans (offline: best repayment feature 0.44 vs 0.32, bureau 0.25 vs 0.20, rate tier 0.23 vs 0.16). The e-commerce burst and the price premium lose most of theirs (`ec_accel_30d` 0.31 vs 0.06, `pp_mean` 0.37 vs 0.14): the spending pattern before an application is tied to the selected loan, not to every loan the customer took. A model trained on both populations needs the `is_hist` flag to keep the two mappings apart, and the historical rows are down-weighted.

    > The historical rows are also data-poor: in 2023 and 2024 the tables cover a shorter past, so only about a third of the snapshots have e-commerce, account or repayment data. The offline counting test (`eda/82`, `eda/83`) found a held-out gain of about +0.002 KS when they are added at weight 0.5, consistent in 5 of 6 half splits. That is small, so the trees below measure the effect directly as a paired difference on identical folds.
"""))

# ---------------------------------------------------------------------------
# Section 5: Validation Strategy
# ---------------------------------------------------------------------------
cells.append(section("Validation Strategy", "5"))

cells.append(M("""
    Because the test set is a random sample of the training period, repeated stratified 5-fold CV is used. Every model reports per-fold KS, the out-of-fold (OOF) KS and its spread. The OOF predictions also drive a public/private split simulation that estimates how far the two leaderboards can disagree from sampling noise alone.

    The folds split only the selected applications, and validation always scores selected applications. A model with historical rows trains on the training part of the fold plus all historical rows, including those of validation and test customers. This mirrors the final situation exactly: the historical rows of test customers are available at fit time (they are dated before those customers' cutoffs and use no competition label), so they are training data for the test predictions too. The augmented frames carry two extra columns, `is_hist` (0 for selected applications) and `hist_term` (12 for selected applications, 6, 12 or 18 for historical loans).
"""))

cells.append(C("""
    X_tr = tr_f.select(FEATURES).to_pandas()
    X_te = te_f.select(FEATURES).to_pandas()
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([X_tr[c], X_te[c]])).categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)
    AUG_COLS = FEATURES + ['is_hist', 'hist_term']
    XA_TR, XA_TE = X_tr.assign(is_hist=0, hist_term=12), X_te.assign(is_hist=0, hist_term=12)
    if hist_feats is not None:
        XA_H = hist_feats.select(FEATURES).to_pandas().assign(is_hist=1, hist_term=hist_feats['hist_term'].to_numpy())
        for c in CAT_COLS:
            XA_H[c] = pd.Categorical(XA_H[c], categories=X_tr[c].cat.categories)
        y_h = hist_feats['target'].to_numpy()
        print(f'historical rows: {len(XA_H):,} ({y_h.mean():.4f} bad), weight {CFG.HIST_WEIGHT}')

    def train_rows(tri, aug, frame=None):
        if not aug:
            return (frame if frame is not None else X_tr).iloc[tri], y[tri], None
        main, hist = (frame if frame is not None else (XA_TR, XA_H))
        X = pd.concat([main.iloc[tri], hist], ignore_index=True)
        return X, np.r_[y[tri], y_h], np.r_[np.ones(len(tri)), np.full(len(y_h), CFG.HIST_WEIGHT)]

    test_ids = te_f['application_id'].to_numpy()
    np.save(CFG.OUTPUT_DIR / 'train_ids.npy', tr_f['application_id'].to_numpy().astype(str))
    np.save(CFG.OUTPUT_DIR / 'test_ids.npy', test_ids.astype(str))

    def folds(seed):
        return list(StratifiedKFold(CFG.N_FOLDS, shuffle=True, random_state=seed).split(X_tr, y))

    def gap_simulation(pred, n_test=30000, public_share=0.5, n_iter=300, seed=0):
        rng = np.random.default_rng(seed)
        gaps = []
        for _ in range(n_iter):
            idx = rng.choice(len(y), n_test, replace=False)
            cut = int(public_share * n_test)
            gaps.append(ks_score(y[idx[:cut]], pred[idx[:cut]]) - ks_score(y[idx[cut:]], pred[idx[cut:]]))
        g = np.abs(np.array(gaps))
        return float(np.median(g)), float(np.percentile(g, 95))

    RESULTS = {}

    def report(name, oof, fold_ks, test_pred):
        RESULTS[name] = dict(oof=oof, test=test_pred, fold_ks=fold_ks)
        med, p95 = gap_simulation(oof)
        print(f'[{name}] OOF KS={ks_score(y, oof):.4f}  AUC={roc_auc_score(y, oof):.4f}  '
              f'fold KS={np.mean(fold_ks):.4f}+-{np.std(fold_ks):.4f}  sim |pub-priv| p50={med:.4f} p95={p95:.4f}')
        np.save(CFG.OUTPUT_DIR / f'oof_{name}.npy', oof)
        np.save(CFG.OUTPUT_DIR / f'test_{name}.npy', test_pred)
"""))

# ---------------------------------------------------------------------------
# Section 6: Gradient Boosting Models
# ---------------------------------------------------------------------------
cells.append(section("Gradient Boosting Models", "6"))

cells.append(M("""
    The three gradient boosting libraries are run over the same repeated folds (3 seeds x 5 folds). Each one handles missing values natively, which matters because 30% of customers lack the history tables entirely. Predictions are averaged over seeds for OOF rows and over all 15 models for the test set. Every library is trained with the deep version 1 settings and with a shallow, more regularised variant (`_s`), and each of those once without and once with the historical rows (`_aug`). All four variants of a library see the same folds and seeds, so their per-fold KS values are paired.
"""))

cells.append(C("""
    def run_gbdt(name, fit_predict, seeds=None, save=True):
        seeds = seeds or CFG.GBDT_SEEDS
        oof = np.zeros(len(y))
        test_pred = np.zeros(len(X_te))
        fold_ks = []
        t0 = time.time()
        for seed in seeds:
            for k, (tri, vai) in enumerate(folds(seed)):
                pv, pt = fit_predict(tri, vai, seed)
                oof[vai] += pv / len(seeds)
                test_pred += pt / (len(seeds) * CFG.N_FOLDS)
                fold_ks.append(ks_score(y[vai], pv))
                rest_min = (time.time() - t0) / len(fold_ks) * (len(seeds) * CFG.N_FOLDS - len(fold_ks)) / 60
                if rest_min > minutes_left() - CFG.BLEND_RESERVE_MIN:
                    print(f'  {name} dropped: {rest_min:.0f} more minutes needed, {minutes_left():.0f} left before the deadline')
                    return None, None
            print(f'  {name} seed {seed}: fold KS {np.round(fold_ks[-CFG.N_FOLDS:], 4)}  ({time.time() - t0:.0f}s)')
        if save:
            report(name, oof, fold_ks, test_pred)
        return oof, fold_ks
"""))

cells.append(M("""
    ## LightGBM

    LightGBM uses native categorical splits and early stopping on validation AUC. AUC is only a proxy for KS (the version 7 review found model pairs that order differently on the two), but it is far less noisy as a stopping signal.
"""))

cells.append(C("""
    LGB_MODELS = []

    def make_lgb_fit(params, cols=None, keep=False, aug=False):
        cols = cols or (AUG_COLS if aug else FEATURES)
        cats = [c for c in CAT_COLS if c in cols]
        XV, XT = (XA_TR, XA_TE) if aug else (X_tr, X_te)
        def fit(tri, vai, seed):
            Xa, ya, wa = train_rows(tri, aug)
            dtr = lgb.Dataset(Xa[cols], ya, weight=wa, categorical_feature=cats)
            dva = lgb.Dataset(XV.iloc[vai][cols], y[vai], categorical_feature=cats, reference=dtr)
            m = lgb.train({**params, 'seed': seed}, dtr, CFG.N_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(CFG.EARLY_STOP, verbose=False)])
            if keep:
                LGB_MODELS.append(m)
            return (m.predict(XV.iloc[vai][cols], num_iteration=m.best_iteration),
                    m.predict(XT[cols], num_iteration=m.best_iteration))
        return fit

    AUG = CFG.USE_HIST and hist_feats is not None
    if CFG.USE_LGB:
        run_gbdt('lgb', make_lgb_fit(CFG.LGB_PARAMS, keep=True))
        run_gbdt('lgb_s', make_lgb_fit(CFG.LGB_SHALLOW))
        if AUG:
            run_gbdt('lgb_aug', make_lgb_fit(CFG.LGB_PARAMS, aug=True))
            run_gbdt('lgb_s_aug', make_lgb_fit(CFG.LGB_SHALLOW, aug=True))
"""))

cells.append(M("""
    ## Feature Ablation

    The version 5 features were selected offline from a residual analysis, which is only a proxy. This cell measures them directly: the shallow LightGBM is trained on one seed with the version 3 features and with all features, on identical folds. The per-fold difference is paired, so fold-to-fold noise cancels. (In version 2 the same check gave +0.0046 KS with 5 of 5 folds improved.)
"""))

cells.append(C("""
    if CFG.USE_LGB:
        ab = {}
        for tag, cols in [('v3', V3_FEATURES), ('v3+v5', FEATURES)]:
            oof_ab, fk_ab = run_gbdt(f'ablation {tag}', make_lgb_fit(CFG.LGB_SHALLOW, cols), seeds=[CFG.SEED], save=False)
            ab[tag] = (ks_score(y, oof_ab), np.array(fk_ab))
        d = ab['v3+v5'][1] - ab['v3'][1]
        print(f"OOF KS v3 features {ab['v3'][0]:.4f} -> v3+v5 features {ab['v3+v5'][0]:.4f}")
        print(f'per-fold KS gain: {np.round(d, 4)}  mean={d.mean():+.4f}  folds improved {int((d > 0).sum())}/{len(d)}')
"""))

cells.append(M("""
    The gain importance of the deep LightGBM models shows which feature families carry the ranking.
"""))

cells.append(C("""
    if LGB_MODELS:
        imp = pd.DataFrame({'feature': LGB_MODELS[0].feature_name(),
                            'gain': np.mean([m.feature_importance('gain') for m in LGB_MODELS], axis=0)})
        imp = imp.sort_values('gain', ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(imp['feature'][::-1], imp['gain'][::-1], color=['C1' if f.startswith(V5_PREFIX) else 'C0' for f in imp['feature'][::-1]])
        ax.set_title('LightGBM mean gain importance (top 30, orange = version 5)')
        plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'lgb_importance.png', dpi=90); plt.show()
"""))

cells.append(M("""
    ## XGBoost

    XGBoost runs the histogram algorithm on the GPU with native categorical support. The test matrix is built once per feature layout.
"""))

cells.append(C("""
    D_TE = {False: xgb.DMatrix(X_te, enable_categorical=True), True: xgb.DMatrix(XA_TE, enable_categorical=True)}

    def make_xgb_fit(params, aug=False):
        XV = XA_TR if aug else X_tr
        def fit(tri, vai, seed):
            Xa, ya, wa = train_rows(tri, aug)
            dtr = xgb.DMatrix(Xa, ya, weight=wa, enable_categorical=True)
            dva = xgb.DMatrix(XV.iloc[vai], y[vai], enable_categorical=True)
            m = xgb.train({**params, 'seed': seed}, dtr, CFG.N_ROUNDS, evals=[(dva, 'va')],
                          early_stopping_rounds=CFG.EARLY_STOP, verbose_eval=False)
            it = (0, m.best_iteration + 1)
            return m.predict(dva, iteration_range=it), m.predict(D_TE[aug], iteration_range=it)
        return fit

    if CFG.USE_XGB:
        run_gbdt('xgb', make_xgb_fit(CFG.XGB_PARAMS))
        run_gbdt('xgb_s', make_xgb_fit(CFG.XGB_SHALLOW))
        if AUG:
            run_gbdt('xgb_aug', make_xgb_fit(CFG.XGB_PARAMS, aug=True))
            run_gbdt('xgb_s_aug', make_xgb_fit(CFG.XGB_SHALLOW, aug=True))
"""))

cells.append(M("""
    ## CatBoost

    CatBoost trains on the GPU with ordered target statistics for the four categorical columns, which are passed as strings (the unknown purpose of a historical loan becomes the string `None`). Its symmetric trees already have an additive flavour, so the shallow variant uses depth 4.
"""))

cells.append(C("""
    def as_str(df):
        out = df.copy()
        for c in CAT_COLS:
            out[c] = out[c].astype(str)
        return out

    XC_TR, XC_TE = as_str(X_tr), as_str(X_te)
    if AUG:
        XC_A = (as_str(XA_TR), as_str(XA_H))
        XC_A_TE = as_str(XA_TE)

    CAT_DEVICE = {'task_type': CFG.CAT_PARAMS['task_type']}

    def make_cat_fit(params, aug=False):
        XV, XT = (XC_A[0], XC_A_TE) if aug else (XC_TR, XC_TE)
        def fit(tri, vai, seed):
            Xa, ya, wa = train_rows(tri, aug, XC_A if aug else XC_TR)
            pool_tr = Pool(Xa, ya, cat_features=CAT_COLS, weight=wa)
            pool_va = Pool(XV.iloc[vai], y[vai], cat_features=CAT_COLS)
            try:
                m = CatBoostClassifier(**{**params, **CAT_DEVICE}, random_seed=seed)
                m.fit(pool_tr, eval_set=pool_va)
            except Exception as e:
                if CAT_DEVICE['task_type'] != 'GPU':
                    raise
                print('  CatBoost GPU failed, falling back to CPU:', repr(e)[:120])
                CAT_DEVICE['task_type'] = 'CPU'
                m = CatBoostClassifier(**{**params, **CAT_DEVICE}, random_seed=seed)
                m.fit(pool_tr, eval_set=pool_va)
            return m.predict_proba(XV.iloc[vai])[:, 1], m.predict_proba(XT)[:, 1]
        return fit

    if CFG.USE_CAT:
        run_gbdt('cat', make_cat_fit(CFG.CAT_PARAMS))
        run_gbdt('cat_s', make_cat_fit(CFG.CAT_SHALLOW))
        if AUG:
            run_gbdt('cat_aug', make_cat_fit(CFG.CAT_PARAMS, aug=True))
            run_gbdt('cat_s_aug', make_cat_fit(CFG.CAT_SHALLOW, aug=True))
"""))

cells.append(M("""
    ## Effect of the Historical Rows

    Each `_aug` model shares folds and seeds with its base model, so the 15 per-fold KS differences are paired and fold difficulty cancels. The OOF difference is also bootstrapped over applications (300 resamples of the same predictions; this ignores retraining noise, so it is a lower bound on the uncertainty). Equal-weight blends of the six base trees and the six augmented trees are compared the same way, per segment.
"""))

cells.append(C("""
    def paired(a, b, n_boot=300):
        fa, fb = np.array(RESULTS[a]['fold_ks']), np.array(RESULTS[b]['fold_ks'])
        rng = np.random.default_rng(0)
        oa, ob = RESULTS[a]['oof'], RESULTS[b]['oof']
        boot = [ks_score(y[i], ob[i]) - ks_score(y[i], oa[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
        return ks_score(y, ob) - ks_score(y, oa), (fb - fa).mean(), int((fb > fa).sum()), len(fa), np.percentile(boot, [5, 95])

    BASE_TREES = [n for n in ('lgb', 'lgb_s', 'xgb', 'xgb_s', 'cat', 'cat_s') if n in RESULTS]
    AUG_TREES = [n + '_aug' for n in BASE_TREES if n + '_aug' in RESULTS]
    for n in BASE_TREES:
        if n + '_aug' in RESULTS:
            d, df, up, nf, ci = paired(n, n + '_aug')
            print(f'{n:<6} -> {n}_aug: OOF KS {d:+.4f} (bootstrap 90% [{ci[0]:+.4f}, {ci[1]:+.4f}]), '
                  f'per-fold {df:+.4f}, {up}/{nf} folds up')
    if AUG_TREES:
        eq = lambda ns: np.mean([rankdata(RESULTS[n]['oof']) for n in ns], axis=0)
        eb, ea = eq([n[:-4] for n in AUG_TREES]), eq(AUG_TREES)
        hist_seg = tr_f['has_history'].to_numpy() == 1
        print(f'equal-weight trees: base {ks_score(y, eb):.4f} -> with historical rows {ks_score(y, ea):.4f}')
        for lab_, m_ in [('thin-file', ~hist_seg), ('with history', hist_seg)]:
            print(f'  {lab_:<13} base {ks_score(y[m_], eb[m_]):.4f} -> aug {ks_score(y[m_], ea[m_]):.4f}')
"""))

cells.append(M("""
    #### Insights

    > Reference points from version 7 (T4, 3 seeds): deep LightGBM 0.5246, shallow LightGBM 0.5243, XGBoost 0.5224 / 0.5241, CatBoost 0.5221 / 0.5206, equal-weight trees 0.5233. The historical rows are worth keeping only if the paired difference is positive in most folds and the bootstrap interval sits mostly above zero. The offline counting test predicted about +0.002, which is below the fold standard deviation of about 0.008, so the paired view is the one to read. Either way, the blend below can use both versions of each library.
"""))

# ---------------------------------------------------------------------------
# Section 7: Foundation and Neural Models
# ---------------------------------------------------------------------------
cells.append(section("Foundation and Neural Models", "7"))

cells.append(M("""
    This section adds models that rank customers differently from the trees. TabM is a neural network trained from scratch, an efficient ensemble of 32 MLPs that share most weights; it is the strongest non-tree model so far (0.5253 PLE and 0.5251 periodic in version 7) and is also trained with the historical rows. RealMLP is a second supervised network with its own tuned defaults (`RealMLP-TD` from PyTabKit), new in version 8. TabPFN-3.5, Causilo, TabICLv2 and EXAONE are pretrained in-context learners: the training rows are the prompt, and no gradient step is taken on this data.

    Version 7 measured TabPFN-3.5 at 0.5189 and Causilo at 0.5226, with rank correlation 0.991 between them. Causilo then received categorical codes as plain floats; its documentation asks for pandas categorical columns, which it now gets. TabICLv2 is a different in-context architecture (column-then-row attention, trained on an updated synthetic prior) that has not been tried yet. EXAONE runs at 12,000 context rows, the largest size that ran on a T4 in version 7; on identical query rows it was within 0.003 to 0.007 KS of a deep LightGBM trained on 56,000 rows, so a full five-fold run is worth having even at that context. Native TabPFN-Rel (0.4917, rank correlation 0.90, no blend weight) is dropped.

    The models run in order of expected value, so that the deadline, if it is ever reached, drops the least promising ones: TabM with and without the historical rows, TabPFN-3.5, Causilo, TabICLv2, RealMLP, the periodic TabM and EXAONE.
"""))

cells.append(M("""
    ## Feature Subset and Context

    Inside each fold, numeric features are ranked by univariate KS on the fold's training part only, which avoids selection leakage. Near-duplicates (Spearman correlation above 0.95) are dropped greedily (a correlation that cannot be computed, for example between two columns that are never observed together, counts as zero), and the top numeric columns are kept with the categorical codes: 100 for TabPFN, Causilo and TabICL, 96 for EXAONE so that its total input is exactly its 100-column limit.

    `run_fm` spreads the five folds over both T4 GPUs, one worker thread per GPU (folds 0, 2, 4 on the first and 1, 3 on the second). Each fold is a separate model, so the result equals a sequential run. Before a fold starts, the longest fold time seen so far is compared with the time left before the deadline, and the model is dropped if the fold could not finish. When a fold runs out of GPU memory, its context halves and the fold is retried. Version 7 showed that EXAONE reports this as `detected VRAM cannot fit the minimum model forward` rather than as a CUDA out-of-memory error, so both messages are recognised; any other error stops the model and is printed.
"""))

cells.append(C("""
    VIEWS = {'hand': (X_tr, X_te, NUM_COLS, CAT_COLS)}
    MEMORY_ERRORS = ('out of memory', 'cannot fit')

    def select_fm_features(idx, n_num=None, view='hand'):
        n_num = n_num or CFG.FM_N_FEATURES
        XA, _, num, cats = VIEWS[view]
        sub, yy = XA.iloc[idx], y[idx]
        ranked = sorted(num, key=lambda c: -ks_nan(yy, sub[c].to_numpy(dtype=float)))[:2 * n_num]
        corr = sub[ranked].rank().corr().abs().fillna(0.0)
        keep = []
        for c in ranked:
            if all(corr.loc[c, k] < 0.95 for k in keep):
                keep.append(c)
            if len(keep) == n_num:
                break
        return keep + cats

    def fm_matrix(df, cols):
        out = df[cols].copy()
        for c in cols:
            if isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].cat.codes
        return out.to_numpy(dtype=np.float32)

    def stratified_subsample(idx, n, seed):
        if len(idx) <= n:
            return idx
        rng = np.random.default_rng(seed)
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        k = int(round(n * len(pos) / len(idx)))
        return rng.permutation(np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])

    def predict_chunks(m, X):
        if getattr(m, 'whole_query', False):
            return m.predict_proba(X)[:, 1]
        return np.concatenate([m.predict_proba(X[i:i + CFG.FM_CHUNK])[:, 1] for i in range(0, len(X), CFG.FM_CHUNK)])

    def run_fm(name, make_model, max_context, n_num=None, view='hand', devices=None, fold_min=0.0):
        XA, XB, _, cats = VIEWS[view]
        devices = devices or CFG.GPUS[:1]
        split = folds(CFG.SEED)
        oof, test_parts, fold_ks = np.zeros(len(y)), {}, {}
        state = {'fold_s': fold_min * 60, 'stop': False}

        def run_fold(k, dev):
            tri, vai = split[k]
            if state['stop'] or state['fold_s'] / 60 > minutes_left() - CFG.BLEND_RESERVE_MIN:
                state['stop'] = True
                return
            t_fold = time.time()
            cols = select_fm_features(tri, n_num, view)
            cat_idx = [cols.index(c) for c in cats]
            Xq = np.vstack([fm_matrix(XA.iloc[vai], cols), fm_matrix(XB, cols)])
            ctx = max_context
            while True:
                ci, m = stratified_subsample(tri, ctx, CFG.SEED + k), None
                try:
                    m = make_model(cat_idx, dev)
                    m.fit(fm_matrix(XA.iloc[ci], cols), y[ci])
                    pq = predict_chunks(m, Xq)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if not any(t in str(e).lower() for t in MEMORY_ERRORS):
                        raise
                    del m; gc.collect(); torch.cuda.empty_cache()
                    ctx //= 2
                    print(f'  {name} fold {k}: out of GPU memory on {dev} -> context reduced to {ctx}')
                    if ctx < 2000:
                        raise
            oof[vai], test_parts[k] = pq[:len(vai)], pq[len(vai):]
            fold_ks[k] = ks_score(y[vai], oof[vai])
            state['fold_s'] = max(state['fold_s'], time.time() - t_fold)
            print(f'  {name} fold {k} on {dev}: KS={fold_ks[k]:.4f} ctx={len(ci)} feats={len(cols)} '
                  f'({time.time() - t_fold:.0f}s, {minutes_left():.0f} min left)', flush=True)
            del m; gc.collect(); torch.cuda.empty_cache()

        def worker(i):
            with torch.cuda.device(devices[i]):
                for k in range(i, CFG.N_FOLDS, len(devices)):
                    try:
                        run_fold(k, devices[i])
                    except Exception:
                        state['stop'] = True
                        raise

        with ThreadPoolExecutor(len(devices)) as ex:
            list(ex.map(worker, range(len(devices))))
        if len(fold_ks) < CFG.N_FOLDS:
            print(f'  {name} dropped: {len(fold_ks)} of {CFG.N_FOLDS} folds finished before the deadline')
            return
        report(name, oof, [fold_ks[k] for k in range(CFG.N_FOLDS)],
               np.mean([test_parts[k] for k in range(CFG.N_FOLDS)], axis=0))
"""))

cells.append(M("""
    ## TabM

    A neural network needs inputs on a common scale, without missing values. The offline checks (`eda/32_nn_prep_verify.py`) confirmed the preparation below on the real data:

    - Every numeric column is mapped through its mid-rank empirical CDF to a standard normal. It preserves the order of values, so the univariate KS is identical to the raw column. A value shared by many rows lands in the middle of its tie block.
    - Missing values become 0, the median of the normal scale. One indicator per distinct missingness pattern tells the network which tables are absent.

    The map is fitted inside every fold on the training part only (with the historical rows when they are used), so no validation row influences its own input.

    TabM is trained over the same 3 seeds x 5 folds as the trees, with up to 60 epochs (patience 8). `tabm` uses piecewise-linear embeddings with 48 quantile bins per feature. `tabm_aug` is the same network trained with the historical rows: two numeric inputs (`is_hist`, `hist_term`) are appended, the unknown purpose of a historical loan gets its own category, and each row's binary cross-entropy is multiplied by its sample weight. `tabm_p` (periodic embeddings, learned sine and cosine features) runs later in the section. Each of the 32 members is trained on its own loss, early stopping uses validation AUC on selected applications, and the members' probabilities are averaged.
"""))

_nn = NN_PREP_SRC.read_text(encoding="utf-8")
_nn = re.sub(r'^"""[\s\S]*?"""\n', "", _nn)
_nn = re.sub(r'\n\s+"""[^\n]*"""', "", _nn)
_nn = _nn.replace("import numpy as np\n", "").replace("from scipy.special import ndtri\n", "").strip("\n")
cells.append(code(_nn))

cells.append(C("""
    A_TR, A_TE = X_tr[NUM_COLS].to_numpy(dtype=float), X_te[NUM_COLS].to_numpy(dtype=float)
    CARDS = [len(X_tr[c].cat.categories) for c in CAT_COLS]
    XCAT_TR = np.column_stack([X_tr[c].cat.codes.to_numpy() for c in CAT_COLS]).astype(np.int64)
    XCAT_TE = np.column_stack([X_te[c].cat.codes.to_numpy() for c in CAT_COLS]).astype(np.int64)
    if AUG:
        A_AUG = {'tr': np.c_[A_TR, np.zeros(len(A_TR)), np.full(len(A_TR), 12.0)],
                 'te': np.c_[A_TE, np.zeros(len(A_TE)), np.full(len(A_TE), 12.0)],
                 'h': np.c_[XA_H[NUM_COLS].to_numpy(dtype=float), np.ones(len(XA_H)), XA_H['hist_term'].to_numpy(dtype=float)]}
        XCAT_H = np.column_stack([XA_H[c].cat.codes.to_numpy() for c in CAT_COLS]).astype(np.int64)
        XCAT_H[XCAT_H < 0] = np.array(CARDS)[np.where(XCAT_H < 0)[1]]
        CARDS_AUG = [k + 1 for k in CARDS]
    tri0, vai0 = folds(CFG.SEED)[0]
    xa, xb, info0 = nn_prep(A_TR[tri0], np.vstack([A_TR[vai0], A_TE]))
    print(f'fold-local TabM input on fold 0: {xa.shape[1]} columns ({len(info0["kept"])} features + '
          f'{len(info0["indicator_of"])} missing groups), finite={np.isfinite(xa).all() and np.isfinite(xb).all()}')
    if AUG:
        xa, xb, info1 = nn_prep(np.vstack([A_AUG['tr'][tri0], A_AUG['h']]), np.vstack([A_AUG['tr'][vai0], A_AUG['te']]))
        print(f'with historical rows: {xa.shape[0]:,} training rows, {xa.shape[1]} columns '
              f'({len(info1["indicator_of"])} missing groups), finite={np.isfinite(xa).all() and np.isfinite(xb).all()}, '
              f'unknown purpose -> category {CARDS[0]}: {int((XCAT_H[:, 0] == CARDS[0]).sum()):,} rows')
    del xa, xb
"""))

cells.append(C("""
    DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def tabm_predict(model, xn, xc):
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(xn[i:i + 8192], xc[i:i + 8192]).squeeze(-1).float()).mean(1)
                              for i in range(0, len(xn), 8192)]).cpu().numpy()

    def make_tabm_fit(embedding, aug=False):
        def fit(tri, vai, seed):
            torch.manual_seed(seed)
            if aug:
                fit_x, rest_x, _ = nn_prep(np.vstack([A_AUG['tr'][tri], A_AUG['h']]), np.vstack([A_AUG['tr'][vai], A_AUG['te']]))
                xc_fit, cards = np.vstack([XCAT_TR[tri], XCAT_H]), CARDS_AUG
                y_fit, w_fit = np.r_[y[tri], y_h], np.r_[np.ones(len(tri)), np.full(len(y_h), CFG.HIST_WEIGHT)]
            else:
                fit_x, rest_x, _ = nn_prep(A_TR[tri], np.vstack([A_TR[vai], A_TE]))
                xc_fit, cards, y_fit, w_fit = XCAT_TR[tri], CARDS, y[tri], np.ones(len(tri))
            xn = torch.as_tensor(fit_x, device=DEV)
            xc = torch.as_tensor(xc_fit, device=DEV)
            vn = torch.as_tensor(rest_x[:len(vai)], device=DEV); vc = torch.as_tensor(XCAT_TR[vai], device=DEV)
            tn = torch.as_tensor(rest_x[len(vai):], device=DEV); tc = torch.as_tensor(XCAT_TE, device=DEV)
            yt = torch.as_tensor(y_fit, dtype=torch.float32, device=DEV)
            wt = torch.as_tensor(w_fit, dtype=torch.float32, device=DEV)
            if embedding == 'ple':
                emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                    rtdl_num_embeddings.compute_bins(xn, n_bins=CFG.TABM_BINS), d_embedding=16, activation=False, version='B')
            else:
                emb = rtdl_num_embeddings.PeriodicEmbeddings(xn.shape[1], lite=False)
            model = tabm.TabM.make(n_num_features=xn.shape[1], cat_cardinalities=cards, d_out=1, num_embeddings=emb).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.TABM_LR, weight_decay=CFG.TABM_WD)
            best, best_state, wait = -1.0, None, 0
            for epoch in range(CFG.TABM_EPOCHS):
                model.train()
                for idx in torch.randperm(len(xn), device=DEV).split(CFG.TABM_BATCH):
                    out = model(xn[idx], xc[idx]).squeeze(-1)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        out, yt[idx].unsqueeze(1).expand_as(out), reduction='none')
                    loss = (bce * wt[idx].unsqueeze(1)).sum() / (wt[idx].sum() * out.shape[1])
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                auc = roc_auc_score(y[vai], tabm_predict(model, vn, vc))
                if auc > best:
                    best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
                else:
                    wait += 1
                if wait >= CFG.TABM_PATIENCE:
                    break
            model.load_state_dict(best_state)
            pv, pt = tabm_predict(model, vn, vc), tabm_predict(model, tn, tc)
            del model, opt, xn, xc, vn, vc, tn, tc; gc.collect(); torch.cuda.empty_cache()
            return pv, pt
        return fit

    TABM_OK = CFG.USE_TABM and tabm is not None and torch.cuda.is_available()
    if TABM_OK:
        for name, embedding, aug in [('tabm', 'ple', False), ('tabm_aug', 'ple', True)]:
            if aug and not AUG:
                continue
            try:
                run_gbdt(name, make_tabm_fit(embedding, aug))
            except Exception as e:
                print(f'{name} skipped:', repr(e))
        if 'tabm_aug' in RESULTS:
            d, df, up, nf, ci = paired('tabm', 'tabm_aug')
            print(f'tabm -> tabm_aug: OOF KS {d:+.4f} (bootstrap 90% [{ci[0]:+.4f}, {ci[1]:+.4f}]), per-fold {df:+.4f}, {up}/{nf} folds up')
"""))

cells.append(M("""
    ## TabPFN-3.5

    TabPFN-3.5 receives the full training fold (about 56,000 rows) and the 100 strongest non-redundant hand-made columns, including the price premium. `fit_mode='fit_with_cache'` encodes the context once into a key-value cache and reuses it for all validation and test rows. Two folds run at a time, one per GPU (version 7: 23 minutes in total). A tiny fit on one GPU first downloads the weights, so the two threads never download them at the same time.
"""))

cells.append(C("""
    def make_tabpfn(cat_idx, device):
        m = TabPFNClassifier(n_estimators=CFG.FM_N_EST, device=device, fit_mode='fit_with_cache', ignore_pretraining_limits=True,
                             categorical_features_indices=cat_idx, random_state=CFG.SEED)
        m.whole_query = True
        return m

    if CFG.USE_TABPFN and TabPFNClassifier is not None and torch.cuda.is_available():
        try:
            make_tabpfn([], CFG.GPUS[0]).fit(np.random.default_rng(0).random((100, 3)), np.arange(100) % 2)
            print(f'TabPFN-3.5 on the hand-made features, folds split over {CFG.GPUS}')
            run_fm('tabpfn', make_tabpfn, CFG.FM_MAX_CONTEXT, devices=CFG.GPUS)
        except Exception as e:
            print('TabPFN skipped:', repr(e))
        gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    ## Causilo

    Causilo is a causally pretrained in-context learner. `CausiloModel` turns the float matrix back into a DataFrame and gives the four categorical columns a pandas categorical dtype, with the categories seen in the training context, as its documentation requires for numeric category codes. Unseen categories are handled by Causilo's fitted schema.
"""))

cells.append(C("""
    class CausiloModel:
        def __init__(self, cat_idx, device):
            self.cat_idx = cat_idx
            self.m = CausiloClassifier(n_estimators=CFG.FM_N_EST, random_state=CFG.SEED, device=device)

        def frame(self, X):
            d = pd.DataFrame(X, columns=[f'f{j}' for j in range(X.shape[1])])
            for j in self.cat_idx:
                d[f'f{j}'] = pd.Categorical(d[f'f{j}'], categories=self.cats[j])
            return d

        def fit(self, X, yy):
            self.cats = {j: np.unique(X[:, j][np.isfinite(X[:, j])]) for j in self.cat_idx}
            self.m.fit(self.frame(X), yy)
            return self

        def predict_proba(self, X):
            return np.asarray(self.m.predict_proba(self.frame(X)))

    if CFG.USE_CAUSILO and CausiloClassifier is not None and torch.cuda.is_available():
        try:
            run_fm('causilo', CausiloModel, CFG.FM_MAX_CONTEXT, devices=CFG.GPUS)
        except Exception as e:
            print('Causilo skipped:', repr(e))
        gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    ## Safety-Net Submission

    The first version 8 run died inside TabICL after four hours. A dead kernel leaves no traceback, but the files already written to `/kaggle/working` are kept. A provisional `submission.csv` is therefore written here from the fixed 80% trees plus 20% TabM blend, which needs no fitting. Section 9 overwrites it with the chosen blend.
"""))

cells.append(C("""
    unit_rank = lambda p: rankdata(p) / len(p)
    safe = np.mean([unit_rank(RESULTS[n]['test']) for n in ('lgb', 'lgb_s', 'xgb', 'xgb_s', 'cat', 'cat_s') if n in RESULTS], axis=0)
    if 'tabm' in RESULTS:
        safe = 0.8 * safe + 0.2 * unit_rank(RESULTS['tabm']['test'])
    pl.DataFrame({'application_id': test_ids, 'default_probability': (safe - safe.min()) / (safe.max() - safe.min())}
                 ).write_csv(CFG.OUTPUT_DIR / 'submission.csv')
    print('provisional submission written:', len(safe), 'rows')
"""))

cells.append(M("""
    ## TabICLv2

    TabICLv2 (checkpoint `tabicl-classifier-v2`) imputes a missing number with the column mean, which would make a thin-file customer look like an average customer with history. `TabICLModel` therefore appends one 0/1 column per distinct missingness group of the training context (columns that are missing on exactly the same rows form one group), the same idea as the TabM preparation. The groups are found on the context rows and applied unchanged to the query rows. All query rows are scored in one call per fold, and the library decides itself whether to offload memory to the CPU. In the first version 8 run this ran out of GPU memory at every context size, from 60,000 down to 7,500, because the 44,000 query rows, not the context, set the memory. The kernel then died, so `USE_TABICL` is off.
"""))

cells.append(C("""
    class TabICLModel:
        whole_query = True

        def __init__(self, cat_idx, device):
            self.m = TabICLClassifier(n_estimators=CFG.TABICL_N_EST, device=device, random_state=CFG.SEED)

        def with_flags(self, X):
            return np.c_[X, np.column_stack([np.isnan(X[:, g[0]]) for g in self.groups]).astype(np.float32)] if self.groups else X

        def fit(self, X, yy):
            miss = np.isnan(X)
            groups = {}
            for j in np.where(miss.any(0))[0]:
                groups.setdefault(miss[:, j].tobytes(), []).append(j)
            self.groups = list(groups.values())
            self.m.fit(self.with_flags(X), yy)
            return self

        def predict_proba(self, X):
            return np.asarray(self.m.predict_proba(self.with_flags(X)))

    if CFG.USE_TABICL and TabICLClassifier is not None and torch.cuda.is_available():
        try:
            TabICLModel([], CFG.GPUS[0]).fit(np.random.default_rng(0).random((100, 3)).astype(np.float32), np.arange(100) % 2)
            run_fm('tabicl', TabICLModel, CFG.FM_MAX_CONTEXT, devices=CFG.GPUS)
        except Exception as e:
            print('TabICL skipped:', repr(e))
        gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    ## RealMLP

    RealMLP-TD is an MLP with tuned defaults (robust scaling, numerical embeddings, a tuned learning-rate schedule) that was competitive with gradient boosting on the PyTabKit benchmarks. It does not accept missing numbers, so each fold imputes every numeric column with its median on the training part and adds a 0/1 missing flag for every column that has missing values there; observed values stay unchanged. The four categorical columns stay categorical and are passed by name. Early stopping uses the validation part with cross-entropy, as for TabM. One seed is used because RealMLP trains for up to 256 epochs.
"""))

cells.append(C("""
    def realmlp_frames(tri, rest):
        num = X_tr[NUM_COLS].iloc[tri]
        med = num.median()
        flag_cols = [c for c in NUM_COLS if num[c].isna().any()]
        def prep(df):
            out = df[NUM_COLS].fillna(med).fillna(0.0).astype(np.float32)
            flags = df[flag_cols].isna().astype(np.float32).add_suffix('_na')
            return pd.concat([out, flags, df[CAT_COLS].astype(str).astype('category')], axis=1)
        return prep(X_tr.iloc[tri]), [prep(d) for d in rest]

    def make_realmlp_fit():
        def fit(tri, vai, seed):
            Xa, (Xv, Xt) = realmlp_frames(tri, [X_tr.iloc[vai], X_te])
            m = RealMLP_TD_Classifier(device='cuda:0', random_state=seed, n_cv=1, n_refit=0, val_metric_name='cross_entropy',
                                      use_ls=False, verbosity=0)
            m.fit(Xa, y[tri], Xv, y[vai], cat_col_names=CAT_COLS)
            pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
            del m; gc.collect(); torch.cuda.empty_cache()
            return pv, pt
        return fit

    if CFG.USE_REALMLP and RealMLP_TD_Classifier is not None and torch.cuda.is_available():
        try:
            run_gbdt('realmlp', make_realmlp_fit(), seeds=CFG.REALMLP_SEEDS)
        except Exception as e:
            print('RealMLP skipped:', repr(e))
"""))

cells.append(M("""
    ## TabM with Periodic Embeddings

    The periodic TabM gave the blend a second, less correlated neural member in versions 6 and 7 (rank correlation 0.990 with the PLE TabM). It runs here, after the new models, because it was the slowest member of version 7 (52 minutes).
"""))

cells.append(C("""
    if TABM_OK:
        try:
            run_gbdt('tabm_p', make_tabm_fit('periodic'))
        except Exception as e:
            print('tabm_p skipped:', repr(e))
"""))

cells.append(M("""
    ## EXAONE Tabular

    EXAONE Tabular (21M parameters) reads at most 100 columns, so it receives exactly 96 numeric plus 4 categorical columns and its internal feature-selection pass never runs (on a T4 that pass fails with "No available kernel", as in version 5). The T4 has compute capability 7.5, where the default flash attention kernel is not available, so the attention modules are switched to the `mem_efficient` kernel. The memory cap is 85% of the free memory of the GPU that runs the fold.

    The context is a stratified sample of 12,000 rows of the training fold (`EXAONE_CONTEXT`). In version 7 the probe at this size took 346 seconds for 12,000 query rows; a fold here scores about 44,000 query rows (14,000 validation and 30,000 test), so a fold should take about 15 to 20 minutes. If a fold still runs out of memory, its context halves.
"""))

cells.append(C("""
    class EXAONEModel:
        whole_query = True

        def __init__(self, device):
            free, _ = torch.cuda.mem_get_info(torch.device(device))
            self.m = EXAONETabularClassifier.from_pretrained(device=device, seed=CFG.SEED, max_vram_bytes=int(0.85 * free))
            if torch.cuda.get_device_capability(torch.device(device))[0] < 8:
                for module in self.m.model.modules():
                    if hasattr(module, '_sdpa_backend'):
                        module._sdpa_backend = 'mem_efficient'

        def fit(self, X, yy):
            self.m.fit(np.asarray(X, dtype=np.float64), np.asarray(yy))
            return self

        def predict_proba(self, X):
            return np.asarray(self.m.predict_proba(np.asarray(X, dtype=np.float64)))

    EXAONE_NUM = 96
    if CFG.USE_EXAONE and EXAONETabularClassifier is not None and torch.cuda.is_available():
        try:
            EXAONEModel(CFG.GPUS[0])
            run_fm('exaone', lambda cat_idx, dev: EXAONEModel(dev), CFG.EXAONE_CONTEXT, n_num=EXAONE_NUM, devices=CFG.GPUS)
        except Exception as e:
            print('EXAONE skipped:', repr(e))
        gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    #### Insights

    > Reference points from version 7 (T4, 3 seeds): equal-weight trees 0.5233, TabM 0.5253, TabM periodic 0.5251, TabPFN-3.5 0.5189, Causilo 0.5226, chosen blend 0.5270 in-sample and 0.5268 on held-out halves. A non-tree model does not need to beat the trees to help. What matters is whether it ranks some customers differently and correctly, which the blend below measures on held-out halves. The paired `tabm` vs `tabm_aug` line shows whether the historical rows help the network as well as the trees.
"""))

# ---------------------------------------------------------------------------
# Section 8: Ensemble and KS Stability
# ---------------------------------------------------------------------------
cells.append(section("Ensemble and KS Stability", "8"))

cells.append(M("""
    KS depends only on ranks, so each model's predictions are converted to normalised ranks before blending. Five candidate blends are compared: the equal-weight blend of the six tree models, a fixed 80% trees plus 20% TabM blend (both need no fitting), greedy ensemble selection (Caruana et al.), which adds with replacement, `ENS_ITERS` times, the model that most raises OOF KS. In version 5 the greedy weights differed a lot between the two halves of the data (TabM at 23% versus 60%), and under nested validation they beat equal weights by only 0.0003. Every candidate is therefore scored on held-out halves: weights fitted on one half are scored on the other. Version 6 picked plain greedy (TabPFN at 20%), although its two halves chose TabPFN at only 3% and 7%. Version 7 added a bagged greedy candidate: greedy selection is repeated on `ENS_BAGS` random half-samples and the weights are averaged, which damps that instability (it won in version 7 with 0.5268). Version 8 adds a fifth fixed candidate, the equal-weight blend of all twelve tree models with and without the historical rows. The candidate with the best held-out score is used. Each non-tree model is also scored at a fixed 20% share, which shows its contribution without any fitting.
"""))

cells.append(C("""
    names = list(RESULTS)
    R_OOF = np.column_stack([rankdata(RESULTS[n]['oof']) / len(y) for n in names])
    R_TE = np.column_stack([rankdata(RESULTS[n]['test']) / len(X_te) for n in names])
    print(pd.DataFrame(R_OOF, columns=names).corr().round(3))

    def best_weights(rows):
        counts, cur = np.zeros(len(names)), np.zeros(len(rows))
        for it in range(CFG.ENS_ITERS):
            cand = [(cur * it + R_OOF[rows, j]) / (it + 1) for j in range(len(names))]
            j = int(np.argmax([ks_score(y[rows], c) for c in cand]))
            counts[j] += 1
            cur = cand[j]
        return counts / counts.sum()

    TREES = ('lgb', 'lgb_s', 'xgb', 'xgb_s', 'cat', 'cat_s')
    w_tree = np.array([1.0 if n in TREES else 0.0 for n in names]); w_tree /= w_tree.sum()
    w_fix = 0.8 * w_tree + 0.2 * np.array([1.0 if n == 'tabm' else 0.0 for n in names]) if 'tabm' in names else w_tree
    half = np.random.default_rng(0).permutation(len(y))
    a, b = half[: len(y) // 2], half[len(y) // 2:]
    w_a, w_b = best_weights(a), best_weights(b)
    def held(wa, wb=None):
        wb = wa if wb is None else wb
        return (ks_score(y[b], R_OOF[b] @ wa) + ks_score(y[a], R_OOF[a] @ wb)) / 2
    def bagged_weights(rows, seed=0):
        rng = np.random.default_rng(seed)
        return np.mean([best_weights(rng.choice(rows, len(rows) // 2, replace=False)) for _ in range(CFG.ENS_BAGS)], axis=0)

    w_tree_all = np.array([1.0 if n in TREES or (n.endswith('_aug') and n[:-4] in TREES) else 0.0 for n in names])
    w_tree_all /= w_tree_all.sum()
    cands = {'equal-weight trees': (held(w_tree), w_tree), '80% trees + 20% TabM': (held(w_fix), w_fix),
             'equal-weight trees + aug trees': (held(w_tree_all), w_tree_all),
             'greedy': (held(w_a, w_b), best_weights(np.arange(len(y)))),
             'bagged greedy': (held(bagged_weights(a), bagged_weights(b)), bagged_weights(np.arange(len(y))))}
    for k_, (sc, _) in cands.items():
        print(f'held-out halves: {k_:<22} {sc:.4f}')
    print('greedy weights fitted per half:', dict(zip(names, np.round(w_a, 2))), dict(zip(names, np.round(w_b, 2))))
    choice = max(cands, key=lambda k_: cands[k_][0])
    W = cands[choice][1]
    print('chosen:', choice, dict(zip(names, np.round(W, 2))))
    for n in [n for n in names if n not in TREES]:
        j = names.index(n)
        w_add = 0.8 * w_tree + 0.2 * np.eye(len(names))[j]
        print(f'  adding {n} at 20% to the tree blend: held-out halves {held(w_add):.4f} (trees alone {held(w_tree):.4f})')

    blend_oof, blend_te = R_OOF @ W, R_TE @ W
    for n in names:
        print(f'  {n:<10} OOF KS {ks_score(y, RESULTS[n]["oof"]):.4f}')
    print(f'  BLEND    OOF KS {ks_score(y, blend_oof):.4f}   (version 7: 0.5270 bagged greedy in-sample, 0.5268 held-out, 0.5233 equal-weight trees)')
    pd.DataFrame({'application_id': tr_f['application_id'].to_list(), 'target': y, 'blend': blend_oof,
                  **{n: RESULTS[n]['oof'] for n in names}}).to_parquet(CFG.OUTPUT_DIR / 'oof_all.parquet', index=False)
    pd.DataFrame({'application_id': test_ids, 'blend': blend_te,
                  **{n: RESULTS[n]['test'] for n in names}}).to_parquet(CFG.OUTPUT_DIR / 'test_all.parquet', index=False)
"""))

cells.append(M("""
    ## KS Stability Checks

    The blend is checked along four axes: per-fold KS, the simulated public/private gap at 30% and 50% public shares, KS inside the thin-file and history segments, and KS per decision month. A stable model shows small spreads on all four.
"""))

cells.append(C("""
    fk = [ks_score(y[v], blend_oof[v]) for _, v in folds(CFG.SEED)]
    print(f'blend fold KS: {np.round(fk, 4)}  mean={np.mean(fk):.4f} sd={np.std(fk):.4f}')
    for share in (0.3, 0.5):
        med, p95 = gap_simulation(blend_oof, public_share=share)
        print(f'public share {share:.0%}: simulated |KS_pub - KS_priv| median={med:.4f} p95={p95:.4f}')
    hist = tr_f['has_history'].to_numpy()
    for g, lab in [(0, 'thin-file'), (1, 'with history')]:
        msk = hist == g
        print(f'{lab:<13} n={msk.sum():>6} bad={y[msk].mean():.3f} KS={ks_score(y[msk], blend_oof[msk]):.4f}')
    month = tr_f['decision_datetime'].dt.month().to_numpy()
    mk = [ks_score(y[month == mth], blend_oof[month == mth]) for mth in range(1, 13)]
    print('KS by decision month:', np.round(mk, 3), f'sd={np.std(mk):.4f}')

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    s = np.sort(blend_oof)
    f1 = np.searchsorted(np.sort(blend_oof[y == 1]), s, side='right') / (y == 1).sum()
    f0 = np.searchsorted(np.sort(blend_oof[y == 0]), s, side='right') / (y == 0).sum()
    i = np.argmax(np.abs(f1 - f0))
    ax[0].plot(s, f1, label='bad (F1)'); ax[0].plot(s, f0, label='good (F0)')
    ax[0].vlines(s[i], f1[i], f0[i], color='k', ls='--', label=f'KS={abs(f1[i] - f0[i]):.4f}')
    ax[0].set_title('Blend OOF score CDFs'); ax[0].legend()
    ax[1].bar(range(1, 13), mk); ax[1].set_title('Blend OOF KS by decision month'); ax[1].set_xlabel('month')
    plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'ks_stability.png', dpi=90); plt.show()
"""))

cells.append(M("""
    #### Insights

    > A simulated public/private gap of about 0.01 is the noise floor for a 30,000-row test set with about 4,000 bad loans. A leaderboard gap of that size is expected and does not indicate overfitting. A gap well above the simulated 95th percentile would point to a distribution problem rather than chance.

    > Similar KS in the thin-file and history segments, and a flat month profile, show that the blend does not rely on a single sub-population or period. Those are the conditions for the public and private scores to stay close.
"""))

# ---------------------------------------------------------------------------
# Section 9: Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "9"))

cells.append(M("""
    The blended test ranks are min-max scaled into [0, 1], which leaves the KS unchanged. The file is validated against every rule in the competition description before it is written.
"""))

cells.append(C("""
    def to_unit(p):
        return (p - p.min()) / (p.max() - p.min())

    def write_submission(pred, path):
        sub = pl.DataFrame({'application_id': test_ids, 'default_probability': to_unit(pred)})
        assert sub.height == sample_sub.height == T['test_applications'].height
        assert sub['application_id'].n_unique() == sub.height
        assert set(sub['application_id']) == set(sample_sub['application_id'])
        assert sub['default_probability'].is_finite().all()
        assert sub['default_probability'].is_between(0, 1).all()
        sub.write_csv(path)
        print('wrote', path, sub.shape)
        return sub

    submission = write_submission(blend_te, CFG.OUTPUT_DIR / 'submission.csv')
    gbdt = [i for i, n in enumerate(names) if n in TREES]
    if gbdt:
        write_submission(R_TE[:, gbdt].mean(1), CFG.OUTPUT_DIR / 'submission_gbdt_only.csv')
    print(submission.head())
"""))

cells.append(M("""
    #### Insights

    > `submission.csv` is the final blend. `submission_gbdt_only.csv` is a fallback built only from the tree models, and it is useful as a second leaderboard data point. If the two differ on the public board by less than the simulated noise floor, choose the final submission by OOF KS.
"""))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
