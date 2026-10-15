"""Audit existing OOF; bootstrap uncertainty is conditional on these saved predictions, not retraining."""
from shared import *
from deep_common import v3

def main():
    self_check()
    f, tr, te, y, raw, R, s = predictions()
    oldtr, oldy, _, _, old = v3()
    assert np.array_equal(oldtr["application_id"], tr["application_id"]) and np.array_equal(y, oldy)
    tree = R[:, :6].mean(1)
    fixed = .8 * tree + .2 * R[:, -1]
    scores = {"v3_saved_blend": old, **raw, "v5_equal_trees": tree, "v5_fixed_20pct_tabm": fixed, "v5_selected_blend": s}
    csv([{"model": n, "KS": ks(y, a), "signed_KS": signed_ks(y, a)} for n, a in scores.items()], "00_scores")
    groups = {"all": np.ones(len(y), bool)}
    hist, month = tr["has_history"].to_numpy(), tr["decision_datetime"].dt.month().to_numpy()
    groups.update({f"history_{h}": hist == h for h in [0, 1]})
    groups.update({f"month_{m:02}": month == m for m in range(1, 13)})
    rows = []
    for g, ix in groups.items():
        rows.append(dict(group=g, n=int(ix.sum()), positives=int(y[ix].sum()), v3_KS=ks(y[ix], old[ix]),
                         v5_KS=ks(y[ix], s[ix]), tree_KS=ks(y[ix], tree[ix])))
    seg = csv(rows, "00_segments")
    rng = np.random.default_rng(20261014)
    cls = [np.flatnonzero(y == k) for k in [0, 1]]
    boot = []
    for _ in range(400):
        ix = np.concatenate([rng.choice(c, len(c), replace=True) for c in cls])
        boot.append([ks(y[ix], s[ix]) - ks(y[ix], old[ix]), ks(y[ix], s[ix]) - ks(y[ix], fixed[ix])])
    boot = np.asarray(boot)
    gaps = []
    for share in [.3, .5]:
        delta = []
        for _ in range(400):
            ix = rng.choice(len(y), 30000, replace=False)
            k = int(share * 30000)
            delta.append(ks(y[ix[:k]], s[ix[:k]]) - ks(y[ix[k:]], s[ix[k:]]))
        gaps.append({"public_share_scenario": share, "abs_gap_median": np.median(np.abs(delta)),
                     "abs_gap_p95": np.quantile(np.abs(delta), .95)})
    csv(gaps, "00_gap_scenarios")
    audit = dict(n_train=len(y), positives=int(y.sum()), n_test=te.height,
                 delta_v5_v3=ks(y, s)-ks(y, old), conditional_paired_bootstrap_95= np.quantile(boot[:, 0], [.025, .975]).tolist(),
                 selected_minus_fixed20_conditional_95=np.quantile(boot[:, 1], [.025, .975]).tolist(),
                 exaone_original_probe_context_bad=int(round(6000*y.mean())),
                 exaone_original_probe_query_bad=max(0, int(round(6000*y.mean()))-3000),
                 cache_features=f.width-5, cache_has_pp_median="pp_median" in f.columns,
                 note="Blend weights and features already selected on these labels; bootstrap excludes selection/retraining uncertainty.")
    js(audit, "00_audit")
    tr.select("application_id", "customer_id", "target").with_columns(
        v5_score=pl.Series(s), v3_score=pl.Series(old), diagnostic_fold=pl.Series(folds(y, seed=20261014))
    ).write_parquet(OUT / "00_bound_oof.parquet")
    paths = [ROOT / "notebook/v5.ipynb", ROOT / "data_pq/features.parquet", ROOT / "results/v5/submission.csv"]
    paths += sorted((ROOT / "results/v5").glob("oof_*.npy"))
    js({str(p.relative_to(ROOT)): digest(p) for p in paths}, "00_input_sha256")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    m = seg[seg.group.str.startswith("month")]
    ax[0].plot(range(1, 13), m.v3_KS, "o-", label="v3")
    ax[0].plot(range(1, 13), m.v5_KS, "o-", label="v5")
    ax[0].set(xlabel="Decision month", ylabel="Saved OOF KS", title="Random OOF sliced by month (not time holdout)")
    ax[0].legend()
    ax[1].hist(boot[:, 0], bins=30); ax[1].axvline(0, color="r")
    ax[1].set(title="Paired bootstrap: v5 minus v3", xlabel="KS delta; conditional on selected predictions")
    corr = pd.DataFrame(R, columns=list(raw)).corr()
    im = ax[2].imshow(corr, vmin=.85, vmax=1)
    ax[2].set_xticks(range(len(raw)), raw, rotation=90); ax[2].set_yticks(range(len(raw)), raw)
    ax[2].set_title("OOF rank correlation")
    fig.colorbar(im, ax=ax[2]); figsave(fig, "00_validation")
    csv(corr.reset_index(), "00_prediction_correlation")

if __name__ == "__main__":
    main()
