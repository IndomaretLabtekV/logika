# DSC LOGIKA UI 2026 — Competition Rules

> This document reorganizes the competition rules provided in the prompt. Official committee announcements and the competition guidebook govern any updates or conflicts.

## 1. Eligibility

- Open to active undergraduate students (`S1/D4/D3/D2/D1`) enrolled at an Indonesian university during the **odd semester of academic year 2026/2027**.
- Teams must have **2–3 members**; members may attend different universities.
- Each participant may register with **only one team**.
- Participants must complete official registration.
- The Kaggle team name must match the registered team name.

## 2. Participation Obligations

- Follow all DSC LOGIKA UI 2026 committee procedures, regulations, and official announcements.
- Provide accurate information throughout registration and participation.
- Finalists must complete required re-registration. Failure to do so may be treated as withdrawal.

## 3. Data and Modeling Rules

- **Only datasets supplied by the committee may be used.** External datasets, external sources, or additional data are prohibited.
- Permitted: preprocessing, feature engineering, aggregation, visualization, and machine learning using authorized data.
- Follow the documented table relationships and consult `DATA_DICTIONARY.md`.
- Treat the selected loan application's **`decision_datetime` as its information cutoff**.
- Do not use information that would not have been available at decision time, including future target-period information or other data leakage.
- No external data is required.

## 4. Programming Language

- **Python is mandatory.** Appropriate Python libraries may be used for data processing, feature engineering, visualization, and machine learning.

## 5. Kaggle Submission Rules

Submit a CSV with **exactly**:

```csv
application_id,default_probability
```

Requirements:

- Exactly one prediction per test application.
- Every `application_id` must belong to the test data; none may be missing, duplicated, or added.
- Prediction row count must equal the number of test applications.
- `default_probability` must be a finite numerical value in **`[0, 1]`**.
- Higher values denote higher predicted serious-repayment-difficulty risk.
- Row order does not matter.
- Teams must check submission validity before uploading through Kaggle.

## 6. Elimination-Round Scoring

The elimination round comprises Kaggle evaluation and notebook execution assessment.

| Component | Weight |
|---|---:|
| Kaggle **public** leaderboard score | 45% |
| Kaggle **private** leaderboard score | 45% |
| Notebook reproducibility and execution | 10% |
| **Total** | **100%** |

The Kaggle evaluation metric is the **Kolmogorov–Smirnov (KS) statistic**, measuring separation between predicted scores for loans reaching at least 90 DPD within 12 months and those not reaching that threshold. **Higher KS is better.** The notebook component is assessed **solely for reproducibility and successful code execution**.

## 7. Notebook Submission

Each team must submit a notebook that:

- Contains the competition code used for the team's approach and submissions.
- Is reproducible and executes in the designated environment.
- Does not depend on unauthorized external data.
- Uses correctly configured dependencies, input files, and paths.
- Is uploaded through the committee's designated submission channel.

**Required filename pattern:**

```text
ParticipantNumber_TeamName_Notebook
```

**Example:**

```text
DSC0001_RandomSeed_Notebook
```

**Elimination-round notebook deadline (as stated in the supplied guidebook): 30 October 2026, 23:59 WIB.** Check official announcements for amendments.

## 8. Late Notebook Submission

- A late elimination-round notebook submission incurs a **1-point deduction per minute** after the deadline, as described by the competition guidelines.
- Teams must confirm successful upload before the deadline.

## 9. Advancement to Final Round

- The **eight teams with the highest qualifying elimination scores** advance.
- Finalists must re-register, attend the Final Technical Meeting, and submit the required report and presentation materials on the committee's schedule.
- The final round takes place **offline at Universitas Indonesia, Depok**.
- Additional technical requirements are provided at the Final Technical Meeting.

## 10. Final Round

- Finalists present their work and submit a final report.
- Presentation time and format are determined by the committee.
- Final-round submission details and deadlines are communicated by the committee.
- Final decisions of judges and the committee are final and cannot be contested.

## 11. Prohibited Actions

Prohibited conduct includes:

- Registering in more than one team.
- Giving false or misleading registration information.
- Using unauthorized external datasets, data sources, or additional information.
- Introducing information leakage from the future target period.
- Manipulating or attempting to manipulate scoring or the leaderboard.
- Using unauthorized methods or sources to produce predictions.
- Copying or plagiarizing other teams' work without appropriate attribution.
- Cheating, fraud, or other unfair conduct.
- Disregarding committee regulations or directions.
- Any other proven action that provides an unfair advantage.

## 12. Sanctions and Disqualification

Violations can include administrative noncompliance, multiple-team registration, fraud, false information, unauthorized data, rule-breaking, or unfair advantage.

Depending on severity, committee sanctions may include **score deduction, submission invalidation, leaderboard removal, disqualification, loss of finalist status, or revocation of winner status and/or awards**. The committee determines sanctions appropriate to the violation.

## 13. Data Privacy

- Do not attempt to re-identify real persons, organizations, or locations represented in the data.
- Do not use data obtained through unauthorized means.

## 14. Rule Changes

The committee may revise or update rules. Updates are communicated through **official DSC LOGIKA UI 2026 channels**, which participants must monitor.

## 15. Acceptance of Rules

Participation constitutes agreement to follow applicable rules, technical requirements, and committee decisions.

For dataset details consult [`data.md`](data.md); for the task, metric, and scoring code see [`overview.md`](overview.md). The authoritative column-level reference is the competition-supplied `DATA_DICTIONARY.md`.
