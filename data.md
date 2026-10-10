# Dataset Guide — Credit Risk Prediction

## Purpose

Predict `default_probability`: the probability that a selected loan reaches **90 or more end-of-day Days Past Due (DPD)** within **12 calendar months after origination**.

The dataset represents Indonesian consumer lending. **Currency:** IDR. **Timestamp timezone:** Asia/Jakarta (WIB).

## Provided Files

| File | Description | Role |
|---|---|---|
| `train_applications.csv` | Selected loan applications with target labels | Training / labeled applications |
| `test_applications.csv` | Selected loan applications without target labels | Inference / submission applications |
| `customer_profiles.csv` | Demographic, employment, income, dependency, and housing information | Customer context |
| `previous_applications.csv` | Historical loan applications | Prior application behavior |
| `repayment_history.csv` | Historical installments and repayment behavior | Repayment features |
| `account_behavior.csv` | Monthly balances, cash inflows and outflows, and account activity | Financial behavior features |
| `bureau_observations.csv` | Bureau score, balances, utilization, inquiries, and delinquency indicators | Credit bureau features |
| `ecommerce_transactions.csv` | Historical orders, product categories, and payment methods | Transaction behavior features |
| `sample_submission.csv` | Example prediction submission | Output template |

The tables are interconnected primarily through identifier columns. **Do not infer exact join keys or relationships from file names**; consult the competition's `DATA_DICTIONARY.md` for authoritative column definitions, relationship keys, and table-specific rules. The dictionary was referenced in the supplied description but its contents were not provided here.

## Target

Available **only in `train_applications.csv`**:

| Value | Definition |
|---|---|
| `1` | Loan reaches at least 90 DPD within 12 calendar months after origination. |
| `0` | Loan does not reach 90 DPD within 12 calendar months after origination. |

The target describes **future repayment behavior** and must never be reconstructed using post-cutoff information as model input.

## Information Cutoff and Leakage Prevention

`decision_datetime` is the point at which a prediction for the selected application is made. **Only information available at that decision time can be used.**

Practical implications:

1. **Respect time:** As a conservative default, restrict event-based historical records to timestamps strictly earlier than `decision_datetime`, unless the authoritative data dictionary explicitly establishes an event as available at the cutoff.
2. **Respect availability:** A historical event date is not necessarily the date its associated information became known. Where supplied, use observation or availability timestamps and table-specific cutoff rules.
3. **Prevent target leakage:** Do not use repayment outcomes, future delinquency, later account activity, future bureau snapshots, or transactions unavailable at decision time.
4. **Aggregate as of each application:** For customers with multiple applications, calculate historical counts, recency, balances, and rolling-window statistics separately at each application's cutoff.
5. **Handle joins carefully:** Use documented keys and temporal conditions, verify one-row-per-application output, and prevent inadvertent duplication from one-to-many joins.
6. **Validate temporally:** Favor validation schemes that reflect deployment at loan-decision time; avoid allowing future-derived customer features or target labels to inform earlier predictions.

The above are conservative engineering practices derived from the stated information-cutoff requirement, not additional official specifications.

## Suggested Feature Families

The following are **modeling suggestions**, not confirmed dictionary columns:

| Source | Potential features, if supported by actual columns and timing rules |
|---|---|
| Loan applications | Loan amount, term, affordability ratios, product characteristics |
| Customer profiles | Income-to-obligation proxies, employment and housing stability |
| Previous applications | Prior application count, approval/rejection patterns, time since last application |
| Repayment history | Historical missed-payment frequency, maximum prior DPD, payment timeliness |
| Account behavior | Rolling inflows/outflows, balance volatility, net cash flow |
| Bureau observations | Historical score, utilization, inquiries, outstanding exposures, past delinquency |
| E-commerce transactions | Purchase frequency, amounts, merchant-category mix, payment behavior |

Each suggested feature must pass a **point-in-time availability check** before use.

## Submission Schema

The output CSV must contain **only** these fields:

```csv
application_id,default_probability
A_000001,0.1523
A_000002,0.7812
A_000003,0.0437
```

| Field | Requirement |
|---|---|
| `application_id` | Every test application ID exactly once; no missing or extra IDs |
| `default_probability` | Finite numeric probability in `[0, 1]`; higher implies greater risk |

Submission row order is irrelevant. Use `sample_submission.csv` and confirm the submission count equals the test application count.

## Metric

The evaluation metric is the two-sample **Kolmogorov–Smirnov (KS) statistic** comparing the empirical score distributions for `target = 1` and `target = 0`. **Higher is better**. See [`overview.md`](overview.md) for the formula and provided scoring function.

## Dataset Restrictions

- Use **only committee-provided datasets**; external datasets and additional external sources are prohibited.
- Data processing, feature engineering, aggregation, visualization, and modeling using the supplied datasets are allowed.
- **Python** is required for competition code.
- Do not attempt to identify real individuals, organizations, or places from the records.
- Read `DATA_DICTIONARY.md` before implementing data joins and point-in-time features.

See [`rules.md`](rules.md) for the complete competition rules.
