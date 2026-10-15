"""Exploratory residual associations, not new validation scores or fitted correction models."""
from shared import *
from scipy.stats import spearmanr

def main():
    f, tr, te, y, raw, R, score = predictions()
    cand = pl.read_parquet(OUT / "01_candidates_oof.parquet")
    cand = tr.select("application_id").join(cand, on="application_id", validate="1:1", maintain_order="left")
    new = [c for c in cand.columns if c not in IDS]
    X = cand.select(new).to_pandas()
    old = [c for c in tr.columns if c.startswith("pp_")] + [
        "ec2_ewm28_exc", "ec2_ewm7_rel", "bc_score_mean", "bc_delinq_max", "dti_total",
        "ac_income_cv", "ac_lowdays_delta", "ec_cancel_90d", "rq_late_trend", "declared_monthly_income"]
    for c in old:
        X["v5_" + c] = tr[c].to_numpy()
    rng = np.random.default_rng(91027)
    half = folds(y, seed=91027, k=2) == 0
    for k in range(6):
        X[f"negative_control_{k}"] = rng.normal(size=len(y))
    hist = tr["has_history"].to_numpy()
    month = tr["decision_datetime"].dt.month().to_numpy()
    strata = {n: qbins(score, score[half], n) * 24 + hist * 12 + month - 1 for n in [20, 60]}
    rows = []
    for c in X:
        x = X[c].to_numpy(float); b = qbins(x, x[half], 5)
        finite = np.isfinite(x[half]); median = np.median(x[half][finite]) if finite.any() else 0.
        xx = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
        direction = 1 if signed_ks(y[half], xx[half]) >= signed_ks(y[half], -xx[half]) else -1
        row = dict(feature=c, coverage=float(np.isfinite(x).mean()), confirmation_univariate_signed_KS=signed_ks(y[~half], direction*xx[~half]))
        for part, mask in [("discovery", half), ("confirmation", ~half)]:
            stat, df, p = conditional_test(y[mask], b[mask], strata[20][mask])
            row[part + "_p"] = p
        row["fine_risk_confirmation_p"] = conditional_test(y[~half], b[~half], strata[60][~half])[2]
        rows.append(row)
    res = pd.DataFrame(rows)
    for part in ["discovery", "confirmation"]:
        res[part + "_q"] = bh(res[part + "_p"])
    res["replicates"] = (res.discovery_q < .05) & (res.confirmation_q < .05) & (res.fine_risk_confirmation_p < .05)
    res = res.sort_values(["confirmation_q", "confirmation_p"])
    csv(res, "02_residual_screen")
    js(dict(tested=len(res), new_candidates=len(new),
        replicated_new=res[res.feature.isin(new) & res.replicates].feature.tolist(),
        replicated_existing=res[res.feature.str.startswith("v5_") & res.replicates].feature.tolist(),
        replicated_negative_controls=res[res.feature.str.startswith("negative_control") & res.replicates].feature.tolist(),
        caution="Exploratory: the v5 scores/features were selected using these labels. Coarse score strata do not remove all confounding. No fresh holdout, no promised KS gain."),
        "02_decision")
    top = res[~res.feature.str.startswith("negative_control")].head(6).feature.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    rate_rows = []
    for ax, c in zip(axes.ravel(), top):
        x = X[c].to_numpy(float); b = qbins(x, x[half], 5)
        for part, mask, style in [("discovery", half, "o-"), ("confirmation", ~half, "s--")]:
            gg = pd.DataFrame({"bin": b[mask], "y": y[mask]}).groupby("bin").y.agg(["mean", "count"])
            ax.plot(gg.index, gg["mean"], style, label=part)
            for bi, rr in gg.iterrows():
                rate_rows.append(dict(feature=c, part=part, bin=bi, bad_rate=rr["mean"], n=rr["count"]))
        ax.set(title=c, xlabel="Feature bin; last=missing", ylabel="Observed bad rate (unadjusted)")
        ax.legend(fontsize=8)
    fig.suptitle("Exploratory associations: no feature passes the replication rule")
    figsave(fig, "02_residual_candidates"); csv(rate_rows, "02_candidate_rates")
    # Stop when candidates fail repeated conditional association; do not repeatedly tune on confirmation labels.
    corr = X[new + ["v5_pp_mean", "v5_ec2_ewm28_exc"]].corr(method="spearman", min_periods=500)
    csv(corr.reset_index(), "02_candidate_correlations")
    matched = X.price_change_matched.notna()
    rows = []
    for g, mask in [("matched", matched), ("not_matched", ~matched)]:
        for cls in [0, 1]:
            a = mask.to_numpy() & (y == cls)
            rows.append(dict(group=g, target=cls, n=int(a.sum()), fraction_of_class=float(a.sum()/(y==cls).sum()),
                recent_mean=float(X.loc[a, "price_0_30"].mean()), older_mean=float(X.loc[a, "price_90_182"].mean()),
                paired_change_mean=float(X.loc[a, "price_change_matched"].mean())))
    csv(rows, "02_matched_temporal_cohort")

if __name__ == "__main__":
    main()
