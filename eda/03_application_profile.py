"""Application + customer profile: pricing tier, affordability, demographics."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from common import OUT, analyze, features

f = features()
cols = ["loan_amount", "annual_interest_rate", "proposed_monthly_installment", "loan_purpose", "age", "region",
        "employment_type", "employment_tenure_months", "declared_monthly_income", "dependents", "housing_status",
        "decision_lag_h", "app_hour", "app_dow", "dti_declared", "lti_declared", "tenure_over_age", "income_per_head"]
analyze(f, cols, "03_app")
tr = f.filter(pl.col("is_train") == 1)
# is the rate tier explained by profile? bad rate within each rate tier by DTI tercile
t = tr.with_columns(dti_q=pl.col("dti_declared").qcut(3, labels=["lo", "mid", "hi"]))
print(t.group_by("annual_interest_rate", "dti_q").agg(n=pl.len(), r=pl.col("target").mean())
      .pivot(on="dti_q", index="annual_interest_rate", values="r").sort("annual_interest_rate"))
print(t.group_by("annual_interest_rate", "has_history").agg(n=pl.len(), r=pl.col("target").mean()).sort("annual_interest_rate", "has_history"))
print("rate tier mix train vs test:\n", f.group_by("is_train", "annual_interest_rate").len()
      .with_columns(p=pl.col("len") / pl.col("len").sum().over("is_train")).pivot(on="is_train", index="annual_interest_rate", values="p").sort("annual_interest_rate"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for v in sorted(tr["annual_interest_rate"].unique()):
    ax[0].hist(tr.filter(pl.col("annual_interest_rate") == v)["dti_declared"].log(), bins=60, histtype="step", density=True, label=str(v))
ax[0].legend(title="rate"); ax[0].set_title("log DTI by rate tier")
ax[1].hist([f.filter(pl.col("is_train") == k)["loan_amount"].log10() for k in (1, 0)], bins=50, density=True, label=["train", "test"])
ax[1].legend(); ax[1].set_title("log10 loan_amount train vs test")
plt.tight_layout(); plt.savefig(OUT / "03_app_extra.png", dpi=80)
