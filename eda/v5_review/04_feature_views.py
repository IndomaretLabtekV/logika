"""Audit DFS versus handcrafted inputs; export one ID-bound, fold-local raw feature view. No model."""
from shared import *
from common import ks_nan

CORE = ["safe_price_mean", "safe_price_done", "price_0_30", "price_90_182", "safe_qty_mean",
        "ec2_ewm28_exc", "ec2_ewm7_rel", "has_history", "bc_score_mean", "bc_delinq_max",
        "rp_maxdpd_max", "ac_lowdays_mean", "dti_total", "declared_monthly_income", "dec_month"]

def select(df, y, budget=100, core=()):
    ranked = sorted(df.columns, key=lambda c: -ks_nan(y, df[c].to_numpy(float)))
    pool = list(dict.fromkeys(list(core) + ranked[:2*budget]))
    # Same rank-then-Pearson redundancy rule as v5; pairwise re-ranking would be substantially slower.
    corr = df[pool].rank().corr(min_periods=100).abs()
    keep = list(core)
    for c in pool:
        if c in keep:
            continue
        # Undefined correlation is not evidence of redundancy; keep the feature for model ablation.
        if all(not np.isfinite(corr.loc[c, k]) or corr.loc[c, k] < .95 for k in keep):
            keep.append(c)
        if len(keep) >= budget:
            break
    assert len(set(keep)) == len(keep) <= budget
    return keep

def main():
    f, tr, te = data(); y = tr["target"].to_numpy(); fd = folds(y, seed=20261014); fit = fd != 0
    dfs = f.select("application_id").join(table("dfs"), on="application_id", validate="1:1", maintain_order="left")
    dp = dfs.to_pandas()
    dnums = [c for c in dfs.columns if c not in IDS and c not in CATS]
    dtrain = dp.loc[f["is_train"].to_numpy() == 1, dnums]
    keep_dfs = select(dtrain.iloc[np.flatnonzero(fit)], y[fit])
    prices = pl.read_parquet(OUT / "01_price_fold0.parquet").drop("customer_id", "is_train")
    base = f.drop([c for c in f.columns if c.startswith("pp_")])
    joined = base.join(prices, on="application_id", validate="1:1", maintain_order="left")
    extra = [c for c in dfs.columns if c.startswith("dfs_")]
    joined = joined.join(dfs.select("application_id", *extra), on="application_id", validate="1:1", maintain_order="left")
    nums = [c for c in joined.columns if c not in IDS and c not in CATS]
    jt = joined.filter(pl.col("is_train") == 1).select(nums).to_pandas()
    keep_joint = select(jt.iloc[np.flatnonzero(fit)], y[fit], core=CORE)
    assert len(keep_joint[:96]+CATS) == 100
    view = joined.select("application_id", "customer_id", "is_train", *keep_joint, *CATS)
    assert "target" not in view.columns and all(c not in IDS for c in keep_joint)
    view.write_parquet(OUT / "04_joint_raw_fold0.parquet")
    rows = []
    hist = tr["has_history"].to_numpy()
    for name, X, cols in [("v5_DFS_selection_reproduced_on_diagnostic_fold", dtrain, keep_dfs),
                           ("proposed_hand_plus_DFS", jt, keep_joint)]:
        for h in [0, 1]:
            rows.append(dict(view=name, history=h, numeric_features=len(cols),
                categorical_features=len(CATS), mean_numeric_coverage=float(X.loc[hist == h, cols].notna().to_numpy().mean()),
                price_features=sum("price" in c for c in cols),
                recent_burst_features=sum(c.startswith("ec2_ewm") for c in cols)))
    summary = csv(rows, "04_view_coverage")
    js(dict(dfs_input_predictors=len(dnums)+len(CATS), actual_v5_FM_predictors=104,
        handcrafted_input_predictors=f.width-5,
        old_DFS_has_unit_price_premium=any(c.startswith("pp_") for c in dfs.columns),
        selected_dfs=keep_dfs, selected_joint=keep_joint, categories=CATS, mandatory_core=CORE,
        exaone_columns_total100=keep_joint[:96]+CATS,
        data_artifact="04_joint_raw_fold0.parquet; raw NaNs and categorical strings preserved",
        caveat="Selection uses training part of diagnostic fold0 only. Refit feature selection and price references in every actual model fold; this is not a proven improvement."),
        "04_feature_view_manifest")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for h, shift in [(0, -.15), (1, .15)]:
        subset = summary[summary.history == h]
        ax[0].bar(np.arange(2)+shift, subset.mean_numeric_coverage, .3, label=f"history={h}")
    ax[0].set_xticks(range(2), ["DFS100", "Hand+DFS100"]); ax[0].set_ylim(0, 1)
    ax[0].set_title("Observed feature coverage"); ax[0].legend()
    families = ["price", "ec2", "bureau", "repayment", "account", "other"]
    def family(c):
        if "price" in c or c.startswith("safe_"): return "price"
        if c.startswith("ec2"): return "ec2"
        if c.startswith(("bc_", "bp_", "bh_", "bz_", "dfs_bureau", "dfs_pull")): return "bureau"
        if c.startswith(("rp_", "rq_", "sv_", "dfs_inst", "dfs_loan")): return "repayment"
        if c.startswith(("ac_", "cf_", "dfs_acct")): return "account"
        return "other"
    for cols, shift, label in [(keep_dfs, -.18, "DFS100"), (keep_joint, .18, "Hand+DFS100")]:
        ax[1].bar(np.arange(len(families))+shift, [sum(family(c)==g for c in cols) for g in families], .36, label=label)
    ax[1].set_xticks(range(len(families)), families, rotation=20); ax[1].set_title("Input families; counts are not importance")
    ax[1].legend(); figsave(fig, "04_feature_views")

if __name__ == "__main__":
    main()
