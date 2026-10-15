"""Saved OOF KS, sampling uncertainty, monthly/segment robustness. No model execution."""
from deep_common import *

def main():
    f, y, raw, R, score = v3()
    hist = f["has_history"].to_numpy()
    month = f["decision_datetime"].dt.strftime("%Y-%m").to_numpy()
    scores = {"v3_blend": score, "trees_equal": R[:, :6].mean(1), **raw}
    rows = []
    for name, p in scores.items():
        for seg, m in [("all", np.ones(len(y), bool)), ("thin", hist == 0), ("history", hist == 1)]:
            rows.append(dict(model=name, segment=seg, n=int(m.sum()), positives=int(y[m].sum()),
                             KS=ks(y[m], p[m]), signed_KS=signed_ks(y[m], p[m])))
    save_table(rows, "01_oof_metrics")
    periods = [dict(month=str(m), n=int((month == m).sum()), bad_rate=float(y[month == m].mean()),
                    KS=ks(y[month == m], score[month == m])) for m in np.unique(month)]
    save_table(periods, "01_monthly_oof")
    manifest = f.select("application_id", "customer_id", "decision_datetime", "target", "has_history").with_columns(
        audit_fold=pl.Series(folds(y)), discovery=pl.Series(folds(y, 20261013, 2) == 0))
    manifest.write_parquet(OUT / "audit_split_manifest.parquet")
    print("Audit folds are NEW diagnostic splits, not original model-training fold IDs.")
    # Sort once, then random subset counts compute exact KS including ties.
    order = np.argsort(score, kind="stable")
    ys = y[order]
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
    def weighted(w):
        a, b = w * ys, w * (1 - ys)
        return float(np.abs(np.cumsum(a / a.sum() - b / b.sum())[ends]).max())
    rng = np.random.default_rng(20261013)
    simrows, distributions = [], {}
    for share in (.1, .3, .5):
        pub, priv = [], []
        for _ in range(1000):
            idx = rng.choice(len(y), 30000, replace=False)
            k = int(share * 30000)
            w = np.zeros(len(y)); w[idx[:k]] = 1
            z = np.zeros(len(y)); z[idx[k:]] = 1
            pub.append(weighted(w)); priv.append(weighted(z))
        pub, priv = np.array(pub), np.array(priv)
        distributions[share] = pub
        simrows.append(dict(public_share=share, public_n=int(share*30000), public_median=float(np.median(pub)),
                            public_p025=float(np.quantile(pub, .025)), public_p975=float(np.quantile(pub, .975)),
                            fraction_ge_reported_054545=float(np.mean(pub >= .54545)),
                            pub_private_gap_p95=float(np.quantile(np.abs(pub-priv), .95))))
    save_table(simrows, "01_sampling_simulation")
    print("0.54545 is user-reported public KS. Actual public sample size and labels are UNKNOWN.")
    print("Simulation assumes exchangeable rows and fixed OOF scores; not a forecast or guarantee of private KS.")
    # Compare selected v3 and v2 without recalibrating on labels.
    names2 = TREES + ["tabpfn", "causilo"]
    R2 = np.column_stack([rankdata(np.load(ROOT / "results/v2" / ("oof_"+n+".npy")))/len(y) for n in names2])
    p2 = R2 @ (np.array([17,4,5,0,2,2,0,0])/30)
    o2 = np.argsort(p2, kind="stable")
    e2 = np.r_[np.flatnonzero(np.diff(p2[o2])!=0), len(y)-1]
    diffs = []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(1000):
        w = np.bincount(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))], minlength=len(y))
        cum = np.cumsum(w[o2]*(y[o2]/len(pos)-(1-y[o2])/len(neg)))
        diffs.append(weighted(w[order])-np.abs(cum[e2]).max())
    save_json(dict(v3_KS=ks(y,score), v2_KS=ks(y,p2), observed_delta=ks(y,score)-ks(y,p2),
                   paired_bootstrap_CI95=np.quantile(diffs,[.025,.975]).tolist(),
                   limitation="Fixed selected predictions; does not account for retraining or adaptive feature/model selection."), "01_paired_bootstrap")
    # Missingness/drift and rank correlations are descriptive, not model importance.
    te = features().filter(pl.col("is_train") == 0)
    drift = []
    for c in sorted(set(f.columns)-IDS-set(CATS)):
        x, z = f[c].cast(pl.Float64).to_numpy(), te[c].cast(pl.Float64).to_numpy()
        drift.append(dict(feature=c, PSI=psi(x,z), missing_train=float(np.mean(~np.isfinite(x))),
                          missing_test=float(np.mean(~np.isfinite(z)))))
    save_table(pl.DataFrame(drift).sort("PSI", descending=True), "01_feature_drift")
    fig, ax = plt.subplots(1,3,figsize=(16,4.5))
    for share,pub in distributions.items():
        ax[0].hist(pub,bins=35,density=True,alpha=.35,label=f"public {share:.0%}")
    ax[0].axvline(.54545,color="black",ls="--",label="reported public")
    ax[0].set_title("Sampling uncertainty: public KS"); ax[0].legend(fontsize=8)
    ax[1].plot([r["month"][-2:] for r in periods],[r["KS"] for r in periods],"o-")
    ax[1].axhline(ks(y,score),color="grey",ls="--"); ax[1].set_title("Monthly OOF KS (not forward-time CV)")
    names=TREES+["tabpfn","tabm"]
    im=ax[2].imshow(np.corrcoef(R.T),vmin=.94,vmax=1,cmap="viridis")
    ax[2].set_xticks(range(8),names,rotation=75); ax[2].set_yticks(range(8),names)
    ax[2].set_title("OOF Spearman correlation"); fig.colorbar(im,ax=ax[2],shrink=.7)
    figsave(fig,"01_validation")
    fig,ax=plt.subplots(figsize=(9,4))
    for label,m in [("good",y==0),("bad",y==1)]:
        ss=np.sort(score[m]); ax.plot(ss,np.arange(1,len(ss)+1)/len(ss),label=label)
    ax.set_title("V3 empirical CDF: KS is maximum vertical separation"); ax.legend()
    figsave(fig,"01_ks_cdf")

if __name__ == "__main__":
    self_check()
    main()
