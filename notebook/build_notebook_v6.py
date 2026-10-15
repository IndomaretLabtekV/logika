import json
import re
import sys
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "v6.ipynb"
NN_PREP_SRC = Path(__file__).resolve().parent.parent / "eda" / "nn_prep.py"
FEATURES_SRC = Path(__file__).resolve().parent.parent / "eda" / "features.py"
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

    *Three gradient boosting libraries (LightGBM, XGBoost on GPU, CatBoost on GPU) are trained with repeated stratified 5-fold CV. Out-of-fold predictions are rank-blended by greedy ensemble selection only when that beats an equal-weight tree blend on held-out halves, and the blend is checked for KS stability before submission.*

    ```
    7 raw tables
        |
        +-- cutoff-filtered aggregations (one row per application, ~230 features)
             |
        +----------+-----------+-----------+-------------+-----------+
        |          |           |           |             |           |
      LightGBM   XGBoost    CatBoost    TabPFN-3.5   EXAONE    TabM (PLE + periodic)
     deep+shallow deep+shallow deep+shallow     |        (probe)        |
        |          |           |           |             |
        +----------+-- rank blend (greedy selection on OOF KS) ------+
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

    The notebook targets a Kaggle GPU T4 x2 session with internet enabled (needed once to download the TabPFN-3.5 and EXAONE weights and the TabM and EXAONE packages). TabPFN weights are gated: accept the licence on Hugging Face and store a token as a Kaggle secret named `TABPFN_TOKEN` (or `HF_TOKEN`). If a foundation model cannot be loaded, its step is skipped and the notebook still completes with the gradient boosting ensemble. On Kaggle the tree models take about 35 minutes, TabPFN-3.5 about 25 minutes and each TabM variant about 30 minutes. EXAONE runs only if a small probe succeeds and its projected time fits its budget, so the whole run stays at roughly 3 hours.
"""))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(C("""
    %pip install -q polars==2.0.0 lightgbm xgboost catboost tabpfn tabm rtdl_num_embeddings
    %pip install -q --no-deps "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git" || echo 'EXAONE install failed'
"""))

cells.append(md("## Import Libraries"))

cells.append(C("""
    import copy
    import gc
    import os
    import random
    import time
    import warnings
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

    All paths, hyperparameters, and environment flags are centralised here. The data directory is located automatically under `/kaggle/input`, so the notebook works regardless of the competition slug.
"""))

cells.append(C("""
    class Settings:
        SEED       = 42
        GBDT_SEEDS = [42, 7, 2026]
        N_FOLDS    = 5
        _ON_KAGGLE = Path('/kaggle/input').exists()

        DATA_DIR   = (
            next(Path('/kaggle/input').rglob('train_applications.csv')).parent
            if _ON_KAGGLE else Path('../data')
        )
        OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../output')
        FIG_DIR    = OUTPUT_DIR / 'figures'

        USE_LGB     = True
        USE_XGB     = True
        USE_CAT     = True
        USE_TABPFN  = True
        USE_EXAONE  = True
        USE_TABM    = True

        N_GPU          = torch.cuda.device_count()
        FM_N_FEATURES  = 100
        FM_MAX_CONTEXT = 60000
        FM_N_EST       = 4
        FM_CHUNK       = 10000
        TABPFN_BUDGET_MIN = 180
        EXAONE_PROBE_ROWS = 3000
        EXAONE_BUDGET_MIN = 45
        TABM_EPOCHS   = 40
        TABM_PATIENCE = 6
        TABM_BATCH    = 512
        TABM_BINS     = 48
        TABM_LR       = 2e-3
        TABM_WD       = 3e-4
        TABM_FOLD_MAX_MIN = 4
        TABM_BUDGET_MIN   = 50
        ENS_ITERS      = 30

        LGB_PARAMS = dict(
            objective='binary', metric='auc', learning_rate=0.02, num_leaves=31,
            min_child_samples=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, verbose=-1, n_jobs=4,
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
    print('data:', CFG.DATA_DIR)
"""))

cells.append(M("""
    The TabPFN-3.5 weights are gated, so the access token is read from Kaggle secrets when it exists. Missing secrets are not an error.
"""))

cells.append(C("""
    if UserSecretsClient is not None:
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

# ---------------------------------------------------------------------------
# Section 5: Validation Strategy
# ---------------------------------------------------------------------------
cells.append(section("Validation Strategy", "5"))

cells.append(M("""
    Because the test set is a random sample of the training period, repeated stratified 5-fold CV is used. Every model reports per-fold KS, the out-of-fold (OOF) KS and its spread. The OOF predictions also drive a public/private split simulation that estimates how far the two leaderboards can disagree from sampling noise alone.
"""))

cells.append(C("""
    X_tr = tr_f.select(FEATURES).to_pandas()
    X_te = te_f.select(FEATURES).to_pandas()
    for c in CAT_COLS:
        cats = pd.Categorical(pd.concat([X_tr[c], X_te[c]])).categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)
    test_ids = te_f['application_id'].to_numpy()

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
    The three gradient boosting libraries are run over the same repeated folds (3 seeds x 5 folds). Each one handles missing values natively, which matters because 30% of customers lack the history tables entirely. Predictions are averaged over seeds for OOF rows and over all 15 models for the test set. Every library is trained twice: with the deep version 1 settings and with a shallow, more regularised variant (`_s`).
"""))

cells.append(C("""
    def run_gbdt(name, fit_predict, seeds=None, save=True, budget_min=None):
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
                projected = (time.time() - t0) / len(fold_ks) * len(seeds) * CFG.N_FOLDS
                if budget_min and len(fold_ks) == 1 and projected > budget_min * 60:
                    print(f'  {name} skipped: projected {projected / 60:.0f} min > budget {budget_min} min')
                    return None, None
            print(f'  {name} seed {seed}: fold KS {np.round(fold_ks[-CFG.N_FOLDS:], 4)}  ({time.time() - t0:.0f}s)')
        if save:
            report(name, oof, fold_ks, test_pred)
        return oof, fold_ks
"""))

cells.append(M("""
    ## LightGBM

    LightGBM uses native categorical splits and early stopping on validation AUC, which is monotonically related to KS for a fixed ranking quality.
"""))

cells.append(C("""
    LGB_MODELS = []

    def make_lgb_fit(params, cols=None, keep=False):
        cols = cols or FEATURES
        cats = [c for c in CAT_COLS if c in cols]
        def fit(tri, vai, seed):
            dtr = lgb.Dataset(X_tr.iloc[tri][cols], y[tri], categorical_feature=cats)
            dva = lgb.Dataset(X_tr.iloc[vai][cols], y[vai], categorical_feature=cats, reference=dtr)
            m = lgb.train({**params, 'seed': seed}, dtr, CFG.N_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(CFG.EARLY_STOP, verbose=False)])
            if keep:
                LGB_MODELS.append(m)
            return (m.predict(X_tr.iloc[vai][cols], num_iteration=m.best_iteration),
                    m.predict(X_te[cols], num_iteration=m.best_iteration))
        return fit

    if CFG.USE_LGB:
        run_gbdt('lgb', make_lgb_fit(CFG.LGB_PARAMS, keep=True))
        run_gbdt('lgb_s', make_lgb_fit(CFG.LGB_SHALLOW))
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

    XGBoost runs the histogram algorithm on the GPU with native categorical support.
"""))

cells.append(C("""
    D_TE = xgb.DMatrix(X_te, enable_categorical=True)

    def make_xgb_fit(params):
        def fit(tri, vai, seed):
            dtr = xgb.DMatrix(X_tr.iloc[tri], y[tri], enable_categorical=True)
            dva = xgb.DMatrix(X_tr.iloc[vai], y[vai], enable_categorical=True)
            m = xgb.train({**params, 'seed': seed}, dtr, CFG.N_ROUNDS, evals=[(dva, 'va')],
                          early_stopping_rounds=CFG.EARLY_STOP, verbose_eval=False)
            it = (0, m.best_iteration + 1)
            return m.predict(dva, iteration_range=it), m.predict(D_TE, iteration_range=it)
        return fit

    if CFG.USE_XGB:
        run_gbdt('xgb', make_xgb_fit(CFG.XGB_PARAMS))
        run_gbdt('xgb_s', make_xgb_fit(CFG.XGB_SHALLOW))
"""))

cells.append(M("""
    ## CatBoost

    CatBoost trains on the GPU with ordered target statistics for the four categorical columns, which are passed as strings. Its symmetric trees already have an additive flavour, so the shallow variant uses depth 4.
"""))

cells.append(C("""
    XC_TR, XC_TE = X_tr.copy(), X_te.copy()
    for c in CAT_COLS:
        XC_TR[c] = XC_TR[c].astype(str)
        XC_TE[c] = XC_TE[c].astype(str)

    def make_cat_fit(params):
        def fit(tri, vai, seed):
            m = CatBoostClassifier(**params, random_seed=seed)
            m.fit(Pool(XC_TR.iloc[tri], y[tri], cat_features=CAT_COLS),
                  eval_set=Pool(XC_TR.iloc[vai], y[vai], cat_features=CAT_COLS))
            return m.predict_proba(XC_TR.iloc[vai])[:, 1], m.predict_proba(XC_TE)[:, 1]
        return fit

    if CFG.USE_CAT:
        run_gbdt('cat', make_cat_fit(CFG.CAT_PARAMS))
        run_gbdt('cat_s', make_cat_fit(CFG.CAT_SHALLOW))
"""))

cells.append(M("""
    #### Insights

    > Compare each model's OOF KS against version 1 (deep LightGBM 0.5057, XGBoost 0.5059, CatBoost 0.5041, public leaderboard 0.531) and against the model-free baseline of about 0.41. The ablation isolates what the new features add, and the deep-versus-shallow pairs show whether the additive bias helps. A difference smaller than the fold standard deviation (about 0.01) should not drive a decision on its own. The blend below weighs every variant by its out-of-fold contribution.
"""))

# ---------------------------------------------------------------------------
# Section 7: Foundation and Neural Models
# ---------------------------------------------------------------------------
cells.append(section("Foundation and Neural Models", "7"))

cells.append(M("""
    TabPFN-3.5 and EXAONE Tabular are pretrained transformers that predict in-context: the training rows are the prompt, and no gradient step is taken on this data. TabM is a neural network trained from scratch, an efficient ensemble of 32 MLPs that share most weights.

    What the earlier versions measured: TabM is the only non-tree model that has helped (+0.003 KS in version 3, the strongest single model in version 5 at 0.5257). TabPFN-3.5 fed the version 3 features scored 0.4995, and fed a generic DFS view (the TabPFN-Rel recipe) it scored 0.4912; neither received blend weight. The DFS run was not a fair test of the foundation model, because its input lacked the price premium that drove version 5. Version 6 therefore gives TabPFN-3.5 the same hand-made features as the trees. EXAONE failed in version 5 for two reasons that are now fixed: it was handed 104 columns, which triggers an internal attention-based feature selection with no T4 kernel, and its probe split put every positive into the context. Causilo (rank correlation 0.994 with TabPFN in versions 1 and 2, and its current release needs PyTorch 2.13 while Kaggle has 2.11), Mitra-v2 (AutoGluon fine-tuning inside every prediction), RT-PluRel (a RelBench forecasting system, about 30 times slower than open-source TabPFN-Rel) and native TabPFN-Rel (Python 3.11 or 3.12 only) are left out.
"""))

cells.append(M("""
    ## Feature Subset and Context

    Inside each fold, numeric features are ranked by univariate KS on the fold's training part only, which avoids selection leakage. Near-duplicates (Spearman correlation above 0.95) are dropped greedily, and the top numeric columns are kept with the categorical codes: 100 for TabPFN, 96 for EXAONE so that its total input is exactly its 100-column limit. Each model gets a time budget: if the first fold projects past it, the model is skipped. On a CUDA out-of-memory error the context halves automatically.
"""))

cells.append(C("""
    def select_fm_features(idx, n_num=None):
        n_num = n_num or CFG.FM_N_FEATURES
        sub, yy = X_tr.iloc[idx], y[idx]
        ranked = sorted(NUM_COLS, key=lambda c: -ks_nan(yy, sub[c].to_numpy(dtype=float)))[:2 * n_num]
        corr = sub[ranked].rank().corr().abs()
        keep = []
        for c in ranked:
            if all(corr.loc[c, k] < 0.95 for k in keep):
                keep.append(c)
            if len(keep) == n_num:
                break
        return keep + CAT_COLS

    def fm_matrix(df, cols):
        out = df[cols].copy()
        for c in CAT_COLS:
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

    def run_fm(name, make_model, max_context, budget_min, n_num=None):
        oof, test_pred, fold_ks = np.zeros(len(y)), np.zeros(len(X_te)), []
        ctx, t0 = max_context, time.time()
        for k, (tri, vai) in enumerate(folds(CFG.SEED)):
            cols = select_fm_features(tri, n_num)
            cat_idx = [cols.index(c) for c in CAT_COLS]
            Xv, Xt = fm_matrix(X_tr.iloc[vai], cols), fm_matrix(X_te, cols)
            while True:
                ci, m = stratified_subsample(tri, ctx, CFG.SEED + k), None
                try:
                    m = make_model(cat_idx)
                    m.fit(fm_matrix(X_tr.iloc[ci], cols), y[ci])
                    pq = predict_chunks(m, np.vstack([Xv, Xt]))
                    pv, pt = pq[:len(Xv)], pq[len(Xv):]
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if 'out of memory' not in str(e).lower():
                        raise
                    del m; gc.collect(); torch.cuda.empty_cache()
                    ctx //= 2
                    print(f'  OOM -> context reduced to {ctx}')
                    if ctx < 2000:
                        raise
            oof[vai], test_pred = pv, test_pred + pt / CFG.N_FOLDS
            fold_ks.append(ks_score(y[vai], pv))
            elapsed = time.time() - t0
            print(f'  {name} fold {k}: KS={fold_ks[-1]:.4f} ctx={len(ci)} feats={len(cols)} ({elapsed:.0f}s)')
            del m; gc.collect(); torch.cuda.empty_cache()
            if k == 0 and elapsed * CFG.N_FOLDS > budget_min * 60:
                print(f'  {name} skipped: projected {elapsed * CFG.N_FOLDS / 60:.0f} min > budget {budget_min} min')
                return
        report(name, oof, fold_ks, test_pred)
"""))

cells.append(M("""
    ## TabPFN-3.5

    TabPFN-3.5 receives the full training fold (about 56,000 rows) and the 100 strongest non-redundant hand-made columns, now including the price premium. `fit_mode='fit_with_cache'` encodes the context once into a KV cache and reuses it for all validation and test rows. The same settings took about 25 minutes in versions 3 and 5.
"""))

cells.append(C("""
    def make_tabpfn(fit_mode, device):
        def make(cat_idx):
            m = TabPFNClassifier(n_estimators=CFG.FM_N_EST, device=device, fit_mode=fit_mode, ignore_pretraining_limits=True,
                                 categorical_features_indices=cat_idx, random_state=CFG.SEED)
            m.whole_query = fit_mode == 'fit_with_cache'
            return m
        return make

    if CFG.USE_TABPFN and TabPFNClassifier is not None and torch.cuda.is_available():
        attempts = [('fit_with_cache', 'cuda:0'),
                    ('fit_preprocessors', ['cuda:0', 'cuda:1'] if CFG.N_GPU > 1 else 'cuda:0')]
        for fit_mode, dev in attempts:
            try:
                print(f'TabPFN-3.5 on the hand-made features: fit_mode={fit_mode} device={dev}')
                run_fm('tabpfn', make_tabpfn(fit_mode, dev), CFG.FM_MAX_CONTEXT, CFG.TABPFN_BUDGET_MIN)
                break
            except Exception as e:
                print(f'  failed: {e!r}')
                gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    ## EXAONE Tabular

    EXAONE Tabular (21M parameters) reads at most 100 columns. Wider inputs trigger an attention-based feature selection pass, and in version 5 that pass failed on the T4 with "No available kernel" before inference started. It now receives exactly 96 numeric plus 4 categorical columns, so that pass never runs. Its model attention still pins PyTorch's flash kernel (Ampere or newer) for long sequences, so every attention module is switched to the `mem_efficient` policy, which the T4 supports. A probe on two disjoint random subsets of `EXAONE_PROBE_ROWS` rows (features chosen from the context subset only) checks that inference runs and measures speed. The context is then sized from the probe time, with cost proportional to context rows times (context + query rows), to fit `EXAONE_BUDGET_MIN`. Any failure skips EXAONE, and its KS is then simply unknown.
"""))

cells.append(C("""
    class EXAONEModel:
        whole_query = True

        def __init__(self):
            self.m = EXAONETabularClassifier.from_pretrained(device='cuda:0', seed=CFG.SEED)
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
            n = CFG.EXAONE_PROBE_ROWS
            perm = np.random.default_rng(CFG.SEED).permutation(len(y))
            ctx_idx, qry_idx = perm[:n], perm[n:2 * n]
            cols = select_fm_features(ctx_idx, EXAONE_NUM)
            assert len(cols) == 100, len(cols)
            t0 = time.time()
            probe = EXAONEModel().fit(fm_matrix(X_tr.iloc[ctx_idx], cols), y[ctx_idx])
            pp = probe.predict_proba(fm_matrix(X_tr.iloc[qry_idx], cols))[:, 1]
            t_probe = time.time() - t0
            print(f'EXAONE probe: context {n} rows ({y[ctx_idx].sum()} bad), query {n} rows ({y[qry_idx].sum()} bad), '
                  f'{t_probe:.0f}s, probe KS={ks_score(y[qry_idx], pp):.4f}')
            del probe; gc.collect(); torch.cuda.empty_cache()
            a = t_probe / (2 * n * n)
            per_fold = CFG.EXAONE_BUDGET_MIN * 60 / CFG.N_FOLDS
            query = len(y) // CFG.N_FOLDS + len(X_te)
            ctx = min(int((-query + np.sqrt(query ** 2 + 4 * per_fold / a)) / 2), CFG.FM_MAX_CONTEXT)
            print(f'  context sized to {ctx} rows for a {CFG.EXAONE_BUDGET_MIN}-minute budget')
            if ctx >= 10000:
                run_fm('exaone', lambda cat_idx: EXAONEModel(), ctx, CFG.EXAONE_BUDGET_MIN * 1.5, n_num=EXAONE_NUM)
            else:
                print('  EXAONE skipped: context would be below 10,000 rows on this GPU')
        except Exception as e:
            print('EXAONE skipped:', repr(e))
        gc.collect(); torch.cuda.empty_cache()
"""))

cells.append(M("""
    ## TabM

    A neural network needs inputs on a common scale, without missing values. The offline checks (`eda/32_nn_prep_verify.py`) confirmed the preparation below on the real data:

    - Every numeric column is mapped through its mid-rank empirical CDF to a standard normal. It preserves the order of values, so the univariate KS is identical to the raw column. A value shared by many rows lands in the middle of its tie block.
    - Missing values become 0, the median of the normal scale. One indicator per distinct missingness pattern tells the network which tables are absent.

    Version 5 fitted this map once on all training rows, so validation rows (without labels) shaped their own scale. The offline audit measured the effect as tiny (median change 0.001), but version 6 fits the map inside every fold on the training part only, so no validation row ever influences its own input.

    Two TabM variants are trained over the same 3 seeds x 5 folds as the trees. `tabm` uses piecewise-linear embeddings with 48 quantile bins per feature, as in version 5. `tabm_p` uses periodic embeddings (learned sine and cosine features), a different way to bend each numeric input that gives the blend a second, less correlated neural member. Each of the 32 members is trained on its own binary cross-entropy, early stopping uses validation AUC, and the members' probabilities are averaged. Two time guards act inside the run: each fold stops training after `TABM_FOLD_MAX_MIN` minutes and keeps its best epoch so far, and the model is dropped after its first fold if the projected total exceeds `TABM_BUDGET_MIN`.
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
    tri0, vai0 = folds(CFG.SEED)[0]
    xa, xb, info0 = nn_prep(A_TR[tri0], np.vstack([A_TR[vai0], A_TE]))
    print(f'fold-local TabM input on fold 0: {xa.shape[1]} columns ({len(info0["kept"])} features + '
          f'{len(info0["indicator_of"])} missing groups), finite={np.isfinite(xa).all() and np.isfinite(xb).all()}')
    del xa, xb
"""))

cells.append(C("""
    DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def tabm_predict(model, xn, xc):
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(model(xn[i:i + 8192], xc[i:i + 8192]).squeeze(-1).float()).mean(1)
                              for i in range(0, len(xn), 8192)]).cpu().numpy()

    def make_tabm_fit(embedding):
        def fit(tri, vai, seed):
            torch.manual_seed(seed)
            fit_x, rest_x, _ = nn_prep(A_TR[tri], np.vstack([A_TR[vai], A_TE]))
            xn = torch.as_tensor(fit_x, device=DEV)
            xc = torch.as_tensor(XCAT_TR[tri], device=DEV)
            vn = torch.as_tensor(rest_x[:len(vai)], device=DEV); vc = torch.as_tensor(XCAT_TR[vai], device=DEV)
            tn = torch.as_tensor(rest_x[len(vai):], device=DEV); tc = torch.as_tensor(XCAT_TE, device=DEV)
            yt = torch.as_tensor(y[tri], dtype=torch.float32, device=DEV)
            if embedding == 'ple':
                emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                    rtdl_num_embeddings.compute_bins(xn, n_bins=CFG.TABM_BINS), d_embedding=16, activation=False, version='B')
            else:
                emb = rtdl_num_embeddings.PeriodicEmbeddings(xn.shape[1], lite=False)
            model = tabm.TabM.make(n_num_features=xn.shape[1], cat_cardinalities=CARDS, d_out=1, num_embeddings=emb).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.TABM_LR, weight_decay=CFG.TABM_WD)
            best, best_state, wait, t_fold = -1.0, None, 0, time.time()
            for epoch in range(CFG.TABM_EPOCHS):
                model.train()
                for idx in torch.randperm(len(xn), device=DEV).split(CFG.TABM_BATCH):
                    out = model(xn[idx], xc[idx]).squeeze(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[idx].unsqueeze(1).expand_as(out))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                auc = roc_auc_score(y[vai], tabm_predict(model, vn, vc))
                if auc > best:
                    best, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
                else:
                    wait += 1
                if wait >= CFG.TABM_PATIENCE or time.time() - t_fold > CFG.TABM_FOLD_MAX_MIN * 60:
                    break
            model.load_state_dict(best_state)
            pv, pt = tabm_predict(model, vn, vc), tabm_predict(model, tn, tc)
            del model, opt, xn, xc, vn, vc, tn, tc; gc.collect(); torch.cuda.empty_cache()
            return pv, pt
        return fit

    if CFG.USE_TABM and tabm is not None and torch.cuda.is_available():
        for name, embedding in [('tabm', 'ple'), ('tabm_p', 'periodic')]:
            try:
                run_gbdt(name, make_tabm_fit(embedding), budget_min=CFG.TABM_BUDGET_MIN)
            except Exception as e:
                print(f'{name} skipped:', repr(e))
"""))

cells.append(M("""
    #### Insights

    > Reference points from version 5: equal-weight trees 0.5246, TabM 0.5257, TabPFN-3.5 on the DFS view 0.4912 (zero weight), chosen blend 0.5274 (0.525 under nested validation). Version 3 had TabPFN-3.5 on its hand-made features at 0.4995. A non-tree model does not need to beat the trees to help. What matters is whether it ranks some customers differently and correctly, which the blend below measures on held-out halves.
"""))

# ---------------------------------------------------------------------------
# Section 8: Ensemble and KS Stability
# ---------------------------------------------------------------------------
cells.append(section("Ensemble and KS Stability", "8"))

cells.append(M("""
    KS depends only on ranks, so each model's predictions are converted to normalised ranks before blending. Three candidate blends are compared: the equal-weight blend of the six tree models, a fixed 80% trees plus 20% TabM blend (both need no fitting), and greedy ensemble selection (Caruana et al.), which adds with replacement, `ENS_ITERS` times, the model that most raises OOF KS. In version 5 the greedy weights differed a lot between the two halves of the data (TabM at 23% versus 60%), and under nested validation they beat equal weights by only 0.0003. Every candidate is therefore scored on held-out halves: greedy weights fitted on one half are scored on the other. The candidate with the best held-out score is used. Each non-tree model is also scored at a fixed 20% share, which shows its contribution without any fitting.
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
    cands = {'equal-weight trees': (held(w_tree), w_tree), '80% trees + 20% TabM': (held(w_fix), w_fix),
             'greedy': (held(w_a, w_b), best_weights(np.arange(len(y))))}
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
        print(f'  {n:<8} OOF KS {ks_score(y, RESULTS[n]["oof"]):.4f}')
    print(f'  BLEND    OOF KS {ks_score(y, blend_oof):.4f}   (version 5: 0.5274 greedy in-sample, 0.5246 equal-weight trees)')
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
