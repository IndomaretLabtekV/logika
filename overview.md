# DSC LOGIKA UI 2026 — Credit Risk Prediction

## Overview

Credit risk is an important challenge in consumer lending. Financial institutions need to identify loan applications that may lead to future repayment difficulties using information available at the time of the loan decision.

In this competition, participants develop a model to **predict the probability of serious repayment difficulty** in an Indonesian consumer lending environment.

A loan experiences serious repayment difficulty if its end-of-day **Days Past Due (DPD) reaches 90 or more within 12 calendar months after origination**.

## Competition Objective

Develop a predictive model that distinguishes loans that reach serious repayment difficulty from those that do not. Participants are encouraged to explore the provided data, combine information across related tables, and engineer meaningful features reflecting financial and behavioral patterns **without data leakage**.

## Target Definition

| Target | Meaning |
|---|---|
| `1` | Loan reaches **at least 90 DPD** within 12 calendar months after origination. |
| `0` | Loan **does not reach 90 DPD** within 12 calendar months after origination. |

The target is included only in `train_applications.csv`.

## Dataset at a Glance

This is a multi-table Indonesian consumer lending dataset containing:

- **Loan applications:** selected application details and loan characteristics.
- **Customer profiles:** demographics, employment, income, dependents, and housing.
- **Previous applications:** prior loan application records.
- **Repayment history:** historical installments and repayment behavior.
- **Account behavior:** balances, inflows, outflows, and monthly account activity.
- **Credit bureau observations:** bureau scores, outstanding balances, utilization, inquiries, and delinquency signals.
- **E-commerce transactions:** historical purchase categories, payment methods, and orders.

All monetary amounts are in **Indonesian Rupiah (IDR)** and timestamps use **Asia/Jakarta local time (WIB)**. See [`data.md`](data.md) for the full file inventory and temporal restrictions. Column-level definitions and table keys are documented in the competition-provided `DATA_DICTIONARY.md` (not reproduced here).

## Evaluation Metric — Kolmogorov–Smirnov (KS) Statistic

The competition evaluates the **separation of predicted risk scores** between loans that reach serious repayment difficulty and loans that do not.

\[
\mathrm{KS} = \sup_s \left|F_1(s)-F_0(s)\right|
\]

where:

- **`F₁(s)`** is the empirical cumulative distribution function (CDF) of risk scores for positive cases (`target = 1`).
- **`F₀(s)`** is the empirical CDF of risk scores for negative cases (`target = 0`).
- **`s`** is the model's predicted probability or risk score.

**Higher KS is better.** KS measures class separation, not probability calibration. The competition scoring code uses the absolute two-sample KS statistic (`scipy.stats.ks_2samp`).

### Official Evaluation Code

```python
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Compute two-sample KS between positive and negative prediction scores."""
    target_col = "default_probability"

    sol = solution.sort_values(by=row_id_column_name).reset_index(drop=True)
    sub = submission.sort_values(by=row_id_column_name).reset_index(drop=True)

    y_true = sol[target_col].to_numpy()
    y_pred = sub[target_col].to_numpy()

    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]

    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0

    ks_result = ks_2samp(preds_bad, preds_good)
    return float(ks_result.statistic)
```

## Submission Format

Submit a CSV with **exactly two columns**:

```csv
application_id,default_probability
A_000001,0.1523
A_000002,0.7812
A_000003,0.0437
A_000004,0.3265
A_000005,0.9142
```

- `application_id` must match an application in the test set, **exactly once**.
- `default_probability` must be a **finite numeric value between 0 and 1**, inclusive; higher means greater predicted risk.
- Include every test application, with **no missing, duplicate, or extra IDs**.
- Row count must match the test set; **row order does not matter**.
- Use `sample_submission.csv` as the schema reference.

## Crucial Modeling Constraint

The selected application's `decision_datetime` is the **information cutoff**. Features may include only information that would have been available by the decision time; never use events or outcomes from the future target period. Refer to [`data.md`](data.md) and [`rules.md`](rules.md).

## Competition Timeline

The supplied competition listing indicated **“Start: 15 minutes ago”** and **“Close: 20 days to go”** at the time it was copied. These are relative, non-authoritative timestamps and should not be used as fixed calendar deadlines. The provided guidebook states that the **elimination-round notebook deadline is 30 October 2026 at 23:59 WIB**. Follow official committee announcements for any updates.

## Elimination-Round Scoring

| Component | Weight |
|---|---:|
| Kaggle public score | 45% |
| Kaggle private score | 45% |
| Notebook reproducibility and execution | 10% |
| **Total** | **100%** |

The **eight highest-scoring qualifying teams** advance to the final round, subject to competition rules.

For eligibility, prohibited actions, notebook requirements, penalties, and final-round procedures, see [`rules.md`](rules.md).
