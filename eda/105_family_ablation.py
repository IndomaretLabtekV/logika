"""Where is the information? Logistic regression on rank-gaussian inputs (valid2.lr_q_fit, as good as TabM/GBDT on
these features) per feature family: KS of the family alone and the KS lost when the family is dropped, 5 outer folds
(seed 42, C picked on the inner split). Also the within-class correlation of the family score components, which shows
whether the families measure one shared risk factor or separate ones. Prints numbers, saves eda/out/105_*.png/csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import valid2 as v
from common import OUT, features

X, Xt, y, ids, tids = v.frames(features())
FAM = {
    "app": lambda c: c in {"loan_amount", "annual_interest_rate", "proposed_monthly_installment", "age",
                           "employment_tenure_months", "declared_monthly_income", "dependents", "decision_lag_h",
                           "app_hour", "app_dow", "dti_declared", "lti_declared", "tenure_over_age", "income_per_head",
                           "dec_month"} or c in v.CAT_COLS,
    "bureau": lambda c: c.startswith(("bc_", "bp_", "bh_", "bz_")),
    "repay": lambda c: c.startswith(("rp_", "rq_", "sv_")),
    "account": lambda c: c.startswith(("ac_", "cf_")),
    "prev": lambda c: c.startswith(("pv_", "ob_")),
    "ec_behav": lambda c: c.startswith(("ec_", "ec2_")),
    "ec_price": lambda c: c.startswith("pp_"),
}
fam_of = {c: next((f for f, t in FAM.items() if t(c)), "cross") for c in X.columns}
print(pd.Series(fam_of).value_counts().to_dict())
fit = v.lr_q_fit()
folds = v.outer(y, 42)


def cv(cols):
    Xs = X[cols]
    cats = [c for c in v.CAT_COLS if c in cols]
    for c in v.CAT_COLS:  # lr_q expects all four categoricals; a constant column when the family excludes them
        if c not in cols:
            Xs = Xs.assign(**{c: pd.Categorical(np.zeros(len(Xs), int), categories=[0])})
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        fi, st = v.inner(y[tr], 42, k)
        (p,), _ = fit(Xs, y, tr, fi, st, [Xs.iloc[va]], 42)
        oof[va] = p
    return oof


allc = list(X.columns)
full = cv(allc)
print(f"all features: LR OOF KS {v.ks(y, full):.5f}")
rows, logit = [], {}
for fam in list(FAM) + ["cross"]:
    inc = [c for c in allc if fam_of[c] == fam]
    exc = [c for c in allc if fam_of[c] != fam]
    alone, drop = cv(inc), cv(exc)
    logit[fam] = np.log(alone / (1 - alone))
    rows.append(dict(family=fam, n=len(inc), ks_alone=v.ks(y, alone), ks_drop=v.ks(y, drop), loss=v.ks(y, full) - v.ks(y, drop)))
    print(rows[-1], flush=True)
d = pd.DataFrame(rows)
d.to_csv(OUT / "105_family_ablation.csv", index=False)
L = pd.DataFrame(logit)
for cls in (0, 1):
    print(f"\nwithin-class correlation of family scores, target={cls}:")
    print(L[y == cls].corr().round(2))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].barh(d.family, d.ks_alone)
ax[0].set(title="KS of family alone (LR)")
ax[1].barh(d.family, d.loss)
ax[1].set(title="KS lost when family is dropped")
fig.tight_layout()
fig.savefig(OUT / "105_family_ablation.png", dpi=110)
