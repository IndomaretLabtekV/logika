import json

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from common import OUT


DEST = OUT / "03_explore"
DEST.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
train = con.execute(
    f"SELECT * FROM read_parquet('{OUT / '02_preprocess/train_features.parquet'}')"
).df()
test = con.execute(
    f"SELECT * FROM read_parquet('{OUT / '02_preprocess/test_features.parquet'}')"
).df()
target = train["target"].astype(int)
excluded = {"application_id", "target"}
numeric = [c for c in train.select_dtypes(include="number").columns if c not in excluded]
categorical = [c for c in train.columns if c not in excluded and c not in numeric]

numeric_rows = []
for name in numeric:
    pos = train.loc[target == 1, name].dropna().to_numpy(dtype=float)
    neg = train.loc[target == 0, name].dropna().to_numpy(dtype=float)
    if len(pos) and len(neg) and len(np.unique(np.r_[pos, neg])) > 1:
        statistic = float(ks_2samp(pos, neg).statistic)
        q1 = float(np.median(pos))
        q0 = float(np.median(neg))
    else:
        statistic, q1, q0 = np.nan, np.nan, np.nan
    numeric_rows.append({
        "feature": name, "nonmissing_train": int(train[name].notna().sum()),
        "unique_train": int(train[name].nunique()),
        "ks_raw_feature": statistic, "median_positive": q1,
        "median_negative": q0,
        "missing_positive_pct": 100 * train.loc[target == 1, name].isna().mean(),
        "missing_negative_pct": 100 * train.loc[target == 0, name].isna().mean(),
    })
numeric_df = pd.DataFrame(numeric_rows).sort_values("ks_raw_feature", ascending=False)
numeric_df.to_csv(DEST / "numeric_signal.csv", index=False)

category_rows = []
category_rates = []
for name in categorical:
    labeled = train[name].fillna("<missing>").astype(str)
    counts = pd.crosstab(labeled, target)
    if counts.shape[1] == 2 and counts.shape[0] > 1:
        chi2 = chi2_contingency(counts, correction=False).statistic
        cramer_v = float(np.sqrt(chi2 / (len(train) * (min(counts.shape) - 1))))
    else:
        cramer_v = np.nan
    category_rows.append({"feature": name, "levels": len(counts), "cramers_v": cramer_v})
    grouped = train.groupby(labeled, dropna=False)["target"].agg(["size", "mean"])
    for level, row in grouped.iterrows():
        category_rates.append({"feature": name, "level": level,
                               "count": int(row["size"]),
                               "target_rate": float(row["mean"])})
pd.DataFrame(category_rows).to_csv(DEST / "categorical_signal.csv", index=False)
pd.DataFrame(category_rates).to_csv(DEST / "categorical_rates.csv", index=False)

selected_numeric = [
    "bureau_score_mean", "bureau_delinquent_accounts_max",
    "installment_income_ratio", "loan_amount", "account_low_balance_days_mean",
    "repayment_max_historical_dpd", "ecommerce_90d_spend",
    "age", "declared_monthly_income",
]
deciles = []
fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout="constrained")
for name, ax in zip(selected_numeric, axes.flat):
    s = train[name]
    if s.nunique(dropna=True) <= 12:
        bins = s
        grouped = train.groupby(bins, observed=True)["target"].agg(["size", "sum", "mean"])
        labels = [str(x) for x in grouped.index]
    else:
        bins = pd.qcut(s, 10, duplicates="drop")
        grouped = train.groupby(bins, observed=True)["target"].agg(["size", "sum", "mean"])
        labels = [f"Q{i + 1}" for i in range(len(grouped))]
    if s.isna().any():
        missing_group = train.loc[s.isna(), "target"]
        grouped = pd.concat([grouped, pd.DataFrame({
            "size": [len(missing_group)], "sum": [missing_group.sum()],
            "mean": [missing_group.mean()]}, index=["Missing"])])
        labels.append("Missing")
    for level, row in grouped.iterrows():
        deciles.append({"feature": name, "bin": str(level), "count": int(row["size"]),
                        "positives": int(row["sum"]), "target_rate": float(row["mean"])})
    ax.bar(range(len(grouped)), 100 * grouped["mean"], color="#0072b2")
    ax.set(title=name.replace("_", " "), ylabel="90-DPD rate, %")
    ax.set_xticks(range(len(grouped)), labels=labels, rotation=45, ha="right", fontsize=8)
pd.DataFrame(deciles).to_csv(DEST / "numeric_bins.csv", index=False)
fig.savefig(DEST / "target_rate_numeric_bins.png", dpi=160)
plt.close(fig)

selected_categorical = ["loan_purpose", "employment_type", "region", "housing_status"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
for name, ax in zip(selected_categorical, axes.flat):
    grouped = train.groupby(name, dropna=False)["target"].agg(["size", "mean"])
    grouped = grouped.sort_values("mean")
    ax.barh(grouped.index.astype(str), 100 * grouped["mean"], color="#e69f00")
    ax.set(title=name.replace("_", " "), xlabel="90-DPD rate, %")
    for i, row in enumerate(grouped.itertuples()):
        ax.text(100 * row.mean, i, f"  n={row.size:,}", va="center", fontsize=8)
fig.savefig(DEST / "target_rate_categories.png", dpi=160)
plt.close(fig)

top = numeric_df.dropna(subset=["ks_raw_feature"]).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
ax.barh(top["feature"], top["ks_raw_feature"], color="#009e73")
ax.set(title="Raw numeric feature separation between target classes",
       xlabel="Two-sample KS of feature values (descriptive)")
fig.savefig(DEST / "numeric_separation.png", dpi=160)
plt.close(fig)

missing_rows = []
for name in numeric + categorical:
    if train[name].isna().any():
        for is_missing, group in train.groupby(train[name].isna()):
            missing_rows.append({"feature": name, "missing": bool(is_missing),
                                 "count": len(group), "target_rate": group.target.mean()})
missing_df = pd.DataFrame(missing_rows)
missing_df.to_csv(DEST / "missingness_target_rates.csv", index=False)
with_missing = missing_df[missing_df.missing].sort_values("count", ascending=False).head(18)
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
ax.barh(with_missing.feature.iloc[::-1], 100 * with_missing.target_rate.iloc[::-1])
ax.axvline(100 * target.mean(), color="black", linestyle="--", label="Overall train rate")
ax.set(title="Target rate when source value is missing", xlabel="90-DPD rate, %")
ax.legend()
fig.savefig(DEST / "missingness_risk.png", dpi=160)
plt.close(fig)

drift_rows = []
for name in numeric:
    tr = train[name].dropna().to_numpy(dtype=float)
    te = test[name].dropna().to_numpy(dtype=float)
    if len(tr) and len(te) and np.unique(tr).size > 1:
        drift_ks = float(ks_2samp(tr, te).statistic)
    else:
        drift_ks = np.nan
    drift_rows.append({"feature": name, "train_test_ks": drift_ks,
                       "train_missing_pct": 100 * train[name].isna().mean(),
                       "test_missing_pct": 100 * test[name].isna().mean()})
drift_df = pd.DataFrame(drift_rows).sort_values("train_test_ks", ascending=False)
drift_df.to_csv(DEST / "train_test_drift.csv", index=False)
top = drift_df.dropna(subset=["train_test_ks"]).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
ax.barh(top.feature, top.train_test_ks, color="#cc79a7")
ax.set(title="Largest train to test changes in feature distribution",
       xlabel="Two-sample KS of feature values")
fig.savefig(DEST / "train_test_drift.png", dpi=160)
plt.close(fig)

source_flags = {
    "previous": "previous_count", "repayment": "due_installment_count",
    "account": "account_complete_months", "bureau": "bureau_provider_count",
    "ecommerce": "ecommerce_order_count",
}
source_rates = []
for source, name in source_flags.items():
    for present, group in train.groupby(train[name] > 0):
        source_rates.append({"source": source, "history_present": bool(present),
                             "count": len(group), "target_rate": group.target.mean()})
source_df = pd.DataFrame(source_rates)
source_df.to_csv(DEST / "source_coverage_target_rates.csv", index=False)

score_bin = pd.qcut(train.bureau_score_mean, 5, duplicates="drop")
ratio_bin = pd.qcut(train.installment_income_ratio, 5, duplicates="drop")
joint_rate = train.pivot_table(index=score_bin, columns=ratio_bin, values="target",
                               aggfunc="mean", observed=True)
joint_count = train.pivot_table(index=score_bin, columns=ratio_bin, values="target",
                                aggfunc="size", observed=True)
joint_rate.to_csv(DEST / "bureau_score_x_affordability_rate.csv")
joint_count.to_csv(DEST / "bureau_score_x_affordability_count.csv")
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
im = ax.imshow(100 * joint_rate.to_numpy(), cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(joint_rate.columns)), labels=joint_rate.columns.astype(str), rotation=45,
              ha="right", fontsize=8)
ax.set_yticks(range(len(joint_rate.index)), labels=joint_rate.index.astype(str), fontsize=8)
ax.set(xlabel="Installment / monthly income quintile",
       ylabel="Bureau score quintile", title="90-DPD rate across two risk dimensions")
fig.colorbar(im, ax=ax, label="90-DPD rate, %")
fig.savefig(DEST / "score_affordability_heatmap.png", dpi=160)
plt.close(fig)

summary = {
    "train_rows": len(train), "test_rows": len(test),
    "positive_count": int(target.sum()), "positive_rate": float(target.mean()),
    "top_numeric_separators": numeric_df[["feature", "ks_raw_feature"]].head(12).to_dict("records"),
    "top_train_test_drift": drift_df[["feature", "train_test_ks"]].head(12).to_dict("records"),
    "history_presence_rates": source_rates,
    "note": "These are descriptive associations and distribution comparisons, not fitted model performance.",
}
with (DEST / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
