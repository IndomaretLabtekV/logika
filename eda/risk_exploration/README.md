# Extended credit-risk exploration

Read [the findings and preprocessing plan](../../RISK_EXPLORATION_REPORT.md). This extension leaves the original EDA and model notebooks intact. It executes directly from the raw committee CSVs and does not need the old feature Parquets.

Run from the repository root using the existing environment:

```bash
.venv/bin/python eda/risk_exploration/build.py
.venv/bin/python eda/risk_exploration/diagnose.py
.venv/bin/python eda/risk_exploration/plot.py
.venv/bin/python eda/risk_exploration/validate.py
```

Alternatively, replace `.venv/bin/python` with `uv run`; each entry point declares its dependencies. Each supports `--help`. Paths resolve relative to the script, so the working directory does not affect data discovery. The build uses four DuckDB threads and a 6 GB memory limit. Outputs overwrite only this extension's artifacts.

| Script or file | Responsibility |
|---|---|
| `build.py` | Load inputs, apply SQL, enforce source contracts and application grain, hash inputs and code, export candidate features |
| `sql/01_timelines.sql` | Complete-month account panel, prior applications, mature repayment denominators, same-provider bureau changes |
| `sql/02_commerce.sql` | Deduplicated orders, cadence, category edges, aligned cash/spending, interactions and cyclic encodings |
| `sql/03_contracts.sql` | Source keys, ownership, monetary reconciliation, eligibility and relationship inventory |
| `diagnose.py`, `diagnostics.sql` | Label associations, train/test feature drift, period checks, Wilson intervals, interaction cells, graph and seasonality screens |
| `plot.py` | Five aggregate charts |
| `validate.py` | Hand-calculable fixtures passed through the actual SQL plus full-data grain checks |

All results live in [`outputs/04_risk_exploration`](../../outputs/04_risk_exploration). Aggregate CSV/JSON/PNG files are eligible for version control. The four larger Parquet files remain in the repository workspace but are ignored by Git:

| Artifact | Grain / meaning |
|---|---|
| `features.parquet` | One selected application; 124 candidate fields plus six metadata fields, including target and identifiers |
| `joint_months.parquet` | Application x observed complete account month; includes commerce support flag |
| `category_edges.parquet` | Application x completed-purchase category, with spend and order weights |
| `split_manifest.parquet` | Application/customer IDs, split, cutoff, retrospective period and provisional label-maturity date |

**This is an exploration panel, not a ready-to-train matrix.** Exclude metadata, diagnostics and constants, apply the feature-family decisions in the report, and fit any data-dependent transformations inside training folds. The current data has one selected application per customer; the build rejects a changed grain because the repayment snapshots cannot safely be reused at arbitrary earlier cutoffs.

For exact formulas use the SQL. For types, missing fractions and cardinalities use `feature_inventory.csv`. For data/code identity use `build_manifest.json`; for boundary-check results use `validation.json`. The analysis is deterministic apart from irrelevant physical ordering among equal application keys in multirow Parquets.
