"""Compare the actual v5 transform with fold-local step ECDF, then verify interpolated ECDF plus explicit masks."""
from shared import *
from preprocess import FoldNumeric, category_fit_transform, stratified_probe, self_check as prep_check
sys.path.insert(0, str(HERE.parent))
from nn_prep import nn_prep

def main():
    prep_check()
    f, tr, te = data(); y = tr["target"].to_numpy()
    fd = folds(y, seed=20261014); fit, valid = fd != 0, fd == 0
    nums = [c for c in f.columns if c not in IDS and c not in CATS]
    a = f.select(nums).to_numpy().astype(float)
    train = tr.select(nums).to_numpy().astype(float)
    _, old, old_info = nn_prep(train, a)
    _, step, step_info = nn_prep(train[fit], a)
    row_train = np.flatnonzero(f["is_train"].to_numpy() == 1)
    vi = row_train[valid]
    common = sorted(set(old_info["kept"]) & set(step_info["kept"]))
    oc = [old_info["kept"].index(c) for c in common]
    sc = [step_info["kept"].index(c) for c in common]
    diff = abs(old[vi][:, oc] - step[vi][:, sc])
    numeric = FoldNumeric().fit(train[fit]); safe = numeric.transform(a)
    assert np.isfinite(safe).all()
    assert np.array_equal(safe[:, len(nums):].astype(bool), ~np.isfinite(a[:, numeric.indicators]))
    assert np.array_equal(numeric.transform(a[:100]), safe[:100])
    _, cat, mappings = category_fit_transform(tr.filter(pl.Series(fit)), f, CATS)
    context, query = stratified_probe(y)
    js(dict(global_shape=old.shape, fold_step_shape=step.shape, corrected_shape=safe.shape,
        numeric_validation_abs_difference_median=float(np.median(diff)),
        numeric_validation_abs_difference_p99=float(np.quantile(diff, .99)),
        original_probe_context_positive=815, original_probe_query_positive=0,
        corrected_probe_context_positive=int(y[context].sum()), corrected_probe_query_positive=int(y[query].sum()),
        unknown_category_values=int((cat == 0).sum()),
        preprocessing_only="Numbers in this audit isolate the numeric-transform change on v5 inputs. The NPZ export additionally replaces global pp columns with fold0 past-only price features; see 03_integrated_manifest."),
        "03_preprocessing_audit")
    rows = []
    for name in ["declared_monthly_income", "pp_mean", "ec_n_orders", "rp_maxdpd_max", "ac_minbal_min"]:
        j = nums.index(name); raw = train[valid, j]; z = safe[vi, j]
        median = np.nanmedian(train[fit, j]); filled = np.where(np.isfinite(raw), raw, median)
        rows.append(dict(feature=name, missing_fraction=float((~np.isfinite(raw)).mean()),
            raw_median_imputed_KS=ks(y[valid], filled), corrected_numeric_KS=ks(y[valid], z),
            numeric_spearman=pd.Series(raw).corr(pd.Series(z), method="spearman")))
    csv(rows, "03_transform_metrics")
    # Combine both fixes for the exported artifact, while keeping the isolated numeric comparison above.
    price = pl.read_parquet(OUT / "01_price_fold0.parquet").drop("customer_id", "is_train")
    full = f.drop([c for c in f.columns if c.startswith("pp_")]).join(
        price, on="application_id", validate="1:1", maintain_order="left")
    final_cols = [c for c in full.columns if c not in IDS and c not in CATS]
    final_raw = full.select(final_cols).to_numpy().astype(float)
    final_prep = FoldNumeric().fit(final_raw[row_train[fit]])
    final_encoded = final_prep.transform(final_raw)
    assert np.isfinite(final_encoded).all()
    assert np.array_equal(final_encoded[:,len(final_cols):].astype(bool), ~np.isfinite(final_raw[:,final_prep.indicators]))
    assert np.array_equal(final_prep.transform(final_raw[vi[:100]]), final_encoded[vi[:100]])
    assert np.array_equal(full["application_id"], f["application_id"])
    js(dict(raw_numeric_columns=len(final_cols), exported_numeric_shape=final_encoded.shape,
        masks=len(final_prep.indicators), categories=len(CATS),
        replaced_global_price_columns=[c for c in f.columns if c.startswith("pp_")],
        reference="01_price_fold0.parquet", fold_seed=20261014, fit_fold_rule="diagnostic_fold != 0; training applications only",
        model_validation_KS=None, status="Integrated price + numeric preprocessing, tested but no model evaluated"),
        "03_integrated_manifest")
    # Diagnostic artifacts are ID-bound; they are not a universal matrix for all CV folds.
    np.savez_compressed(OUT / "03_fold0_numeric_diagnostic.npz", numeric=final_encoded, categorical=cat,
        application_id=f["application_id"].to_numpy().astype(str), is_train=f["is_train"].to_numpy(),
        fit_application_id=tr.filter(pl.Series(fit))["application_id"].to_numpy().astype(str),
        columns=np.array(final_cols, dtype=str), mask_columns=np.array([final_cols[j] for j in final_prep.indicators], dtype=str))
    js(mappings, "03_category_mapping")
    f.select("application_id", "customer_id", "is_train").with_columns(
        diagnostic_fold=pl.Series(np.r_[fd, np.full(te.height, -1)] if np.all(f["is_train"].to_numpy()[:len(y)] == 1) else [
            dict(zip(tr["application_id"], fd)).get(v, -1) for v in f["application_id"]])
    ).write_parquet(OUT / "03_fold_assignment.parquet")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for col, name in enumerate(["declared_monthly_income", "pp_mean", "rp_maxdpd_max"]):
        j = nums.index(name)
        for label in [0, 1]:
            ix = vi[y[valid] == label]
            values = a[ix, j]; finite = values[np.isfinite(values)]
            if len(finite):
                lo, hi = np.quantile(train[fit, j][np.isfinite(train[fit, j])], [.01, .99])
                axes[0, col].hist(np.clip(finite, lo, hi), bins=40, density=True, histtype="step", label=str(label))
            axes[1, col].hist(safe[ix, j], bins=40, density=True, histtype="step", label=str(label))
        axes[0, col].set_title(name + " raw (display clipped)")
        axes[1, col].set_title("Fold-local transform; missing=0 + mask")
        axes[0, col].legend(); axes[1, col].legend()
    figsave(fig, "03_numeric_before_after")
    print("No transformed model KS claimed: only univariate checks and stored v5 OOF are available.")

if __name__ == "__main__":
    main()
