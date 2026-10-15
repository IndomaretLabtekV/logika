"""Fold-local, point-in-time category references; retain item records and deduplicate only order aggregates."""
from shared import *

WINDOWS = [(0, 30), (30, 90), (90, 182), (182, 365), (365, 10000)]

def reference(e, fit_ids):
    r = e.select("customer_id", "product_category", "event_month", "lu").join(fit_ids.select("customer_id"), on="customer_id", how="semi").group_by(
        "product_category", "event_month").agg(n=pl.len(), s=pl.col("lu").sum(), ss=pl.col("lu").pow(2).sum())
    r = r.sort(["product_category", "event_month"]).with_columns(
        *[pl.col(c).cum_sum().over("product_category").alias(c) for c in ["n", "s", "ss"]])
    return r.with_columns(mu=pl.col("s") / pl.col("n"),
        sd=((pl.col("ss") - pl.col("s").pow(2) / pl.col("n")) / (pl.col("n") - 1)).clip(lower_bound=0).sqrt())

def apply_reference(e, ref):
    z = e.sort("query_month").join_asof(ref.select("product_category", "event_month", "mu", "sd").sort("event_month"), left_on="query_month",
        right_on="event_month", by="product_category", strategy="backward",
        allow_exact_matches=False, suffix="_ref", check_sortedness=False)
    assert z.filter(pl.col("event_month_ref") >= pl.col("query_month")).height == 0
    assert z["sd"].null_count() == 0 and z["sd"].min() > 0
    return z.with_columns(z=(pl.col("lu") - pl.col("mu")) / pl.col("sd"))

def aggregate(z):
    expr = [pl.col("z").mean().alias("safe_price_mean"), pl.col("z").std().alias("safe_price_sd"),
        pl.col("z").filter(pl.col("order_status") == "completed").mean().alias("safe_price_done"),
        pl.len().alias("safe_item_count"), pl.col("order_id").n_unique().alias("safe_order_count"),
        pl.col("quantity").mean().alias("safe_qty_mean")]
    for lo, hi in WINDOWS:
        mask = (pl.col("age") > lo) & (pl.col("age") <= hi) if lo else pl.col("age") <= hi
        expr += [pl.col("z").filter(mask).mean().alias(f"price_{lo}_{hi}"),
                 mask.sum().alias(f"items_{lo}_{hi}")]
    expr += [pl.col("z").filter(pl.col("product_category") == c).mean().alias("safe_price_cat_" + c) for c in sorted(z["product_category"].unique().to_list())]
    a = z.group_by("customer_id").agg(expr)
    order = z.group_by("customer_id", "order_id").agg(pl.col("z").mean()).group_by("customer_id").agg(
        pl.col("z").mean().alias("safe_price_order_equal"))
    cat = z.group_by("customer_id", "product_category").agg(pl.col("z").mean(), n=pl.len())
    cat = cat.with_columns(p=pl.col("n") / pl.col("n").sum().over("customer_id"))
    cat = cat.group_by("customer_id").agg(pl.col("z").mean().alias("safe_price_category_equal"),
        pl.col("z").std().alias("safe_price_category_sd"), (-(pl.col("p") * pl.col("p").log()).sum()).alias("category_entropy"))
    a = a.join(order, on="customer_id", validate="1:1").join(cat, on="customer_id", validate="1:1")
    return a.with_columns(
        pl.when((pl.col("items_0_30") >= 3) & (pl.col("items_90_182") >= 3))
          .then(pl.col("price_0_30") - pl.col("price_90_182")).alias("price_change_matched"),
        (pl.col("safe_price_mean") - pl.col("safe_price_order_equal")).alias("item_order_weight_difference"))

def main():
    f, tr, te = data(); y = tr["target"].to_numpy()
    fd = folds(y, seed=20261014); e = items(f)
    assert e["order_item_id"].n_unique() == e.height, "Duplicate item primary keys require source investigation"
    order = e.group_by("order_id").agg(pl.col("customer_id").n_unique().alias("customers"),
        pl.col("order_total").n_unique().alias("totals"), pl.col("order_total").first().alias("order_total"),
        pl.col("item_total").sum().alias("items_sum"))
    assert order["customers"].max() == 1 and order["totals"].max() == 1
    content = ["order_id", "product_category", "quantity", "item_total"]
    repeated = e.filter(e.select(content).is_duplicated()).height
    duplicate_audit = dict(item_rows=e.height, orders=order.height, repeated_item_content_rows=repeated,
        order_total_inflation_if_summed_over_items=e["order_total"].sum()/order["order_total"].sum(),
        order_total_vs_item_sum_mismatch=order.filter(pl.col("items_sum") != pl.col("order_total")).height,
        profile_exact_duplicates_excluding_id=table("customer_profiles").drop("customer_id").is_duplicated().sum(),
        feature_exact_duplicates_excluding_ids=f.select([c for c in f.columns if c not in IDS]).is_duplicated().sum(),
        train_test_customer_overlap=tr.select("customer_id").join(te.select("customer_id"), on="customer_id").height)
    js(duplicate_audit, "01_duplicates")
    del order
    e = e.drop("order_item_id", "order_total", "item_total", "payment_method")
    # Item-content repetition is not sufficient to delete legitimate repeated purchases.
    out_oof, out_test, comparisons = [], [], []
    for k in range(5):
        fit_ids = tr.filter(pl.Series(fd != k)).select("customer_id")
        ref = reference(e, fit_ids)
        z = apply_reference(e, ref)
        a = f.select("application_id", "customer_id", "is_train").join(aggregate(z), on="customer_id", how="left", maintain_order="left")
        assert a.height == f.height and a["application_id"].n_unique() == a.height
        a.write_parquet(OUT / f"01_price_fold{k}.parquet")
        ref.write_csv(OUT / f"01_reference_fold{k}.csv")
        valid_ids = tr.filter(pl.Series(fd == k)).select("application_id")
        out_oof.append(a.join(valid_ids, on="application_id", how="semi"))
        out_test.append(a.filter(pl.col("is_train") == 0).drop("is_train"))
        x = a.filter(pl.col("is_train") == 1)["safe_price_mean"].to_numpy()
        valid = fd == k; old = tr["pp_mean"].to_numpy()
        comparisons.append(dict(fold=k, n_fit=len(y)-int(valid.sum()), n_valid=int(valid.sum()),
            global_price_KS=ks(y[valid], old[valid]), safe_price_KS=ks(y[valid], x[valid]),
            median_abs_change=np.nanmedian(abs(x[valid]-old[valid])),
            p99_abs_change=np.nanquantile(abs(x[valid]-old[valid]), .99),
            rank_correlation=pd.Series(x[valid]).corr(pd.Series(old[valid]), method="spearman")))
        if k == 0:
            # Independent check: one query-month/category uses only training items before that month.
            sample = z.row(0, named=True)
            eligible = e.join(fit_ids, on="customer_id", how="semi").filter(
                (pl.col("product_category") == sample["product_category"]) &
                (pl.col("transaction_datetime") < sample["query_month"]))
            assert np.isclose(eligible["lu"].mean(), sample["mu"], rtol=1e-10)
            assert np.isclose(eligible["lu"].std(), sample["sd"], rtol=1e-10)
            # Perturb validation data: training reference must not move.
            val_ids = tr.filter(pl.Series(fd == k)).select("customer_id")
            extra = e.join(val_ids, on="customer_id", how="semi").head(10).with_columns(lu=pl.lit(50.))
            perturbed = reference(pl.concat([e, extra]), fit_ids)
            assert ref.equals(perturbed)
            # Later events from a training customer cannot change earlier reference snapshots.
            early_month = e["query_month"].min()
            later = e.join(fit_ids, on="customer_id", how="semi").filter(
                pl.col("cutoff") > pl.lit(early_month).dt.offset_by("6mo")).tail(10).with_columns(
                event_month=pl.lit(early_month).dt.offset_by("1mo"),
                transaction_datetime=pl.lit(early_month).dt.offset_by("1mo"), lu=pl.lit(float(np.log(1e9))))
            assert later.height == 10 and later.filter(pl.col("transaction_datetime") > pl.col("cutoff")).height == 0
            perturbed_items = pl.concat([e, later])
            future_ref = reference(perturbed_items, fit_ids)
            early_items = e.filter(pl.col("query_month") == early_month).head(1000)
            early = apply_reference(early_items, future_ref)
            original = apply_reference(early_items, ref)
            assert np.allclose(early["z"], original["z"])
            global_before = e.group_by("product_category").agg(pl.col("lu").mean()).sort("product_category")
            global_after = perturbed_items.group_by("product_category").agg(pl.col("lu").mean()).sort("product_category")
            global_shift = float((global_after["lu"] - global_before["lu"]).abs().max())
            assert global_shift > 0
            js(dict(injected_rows=10, all_before_own_customer_cutoff=True,
                old_global_reference_max_mean_change=global_shift, early_safe_reference_max_z_change=float((early["z"]-original["z"]).abs().max())),
                "01_cross_customer_future_injection")
            del perturbed_items
            sample_ids = tr.group_by("target").head(3).select("customer_id", "target")
            csv(z.join(sample_ids, on="customer_id").select("customer_id", "target", "product_category",
                "transaction_datetime", "cutoff", "event_month_ref", "lu", "mu", "sd", "z").group_by("target", maintain_order=True).head(30).to_pandas(), "01_transformed_samples")
        print("Finished fold", k, flush=True)
    oof = pl.concat(out_oof)
    # Test mean is for descriptive EDA only; a training fold must consume its matching 01_price_foldK file.
    nums = [c for c in out_test[0].columns if c not in ["application_id", "customer_id"]]
    test_mean = pl.concat(out_test).group_by("application_id", "customer_id").agg(*[pl.col(c).mean() for c in nums]).with_columns(is_train=pl.lit(0))
    combined = pl.concat([oof, test_mean.select(oof.columns)], how="vertical_relaxed")
    combined = f.select("application_id").join(combined, on="application_id", validate="1:1", maintain_order="left")
    combined.write_parquet(OUT / "01_candidates_oof.parquet")
    cmp = csv(comparisons, "01_reference_comparison")
    ct = combined.filter(pl.col("is_train") == 1).to_pandas()
    old = tr["pp_mean"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ix = np.random.default_rng(7).choice(len(y), 3000, replace=False)
    ax[0].scatter(old[ix], ct.safe_price_mean.to_numpy()[ix], s=3, alpha=.3)
    ax[0].set(xlabel="v5 global price premium", ylabel="Fold-local past-only premium", title="Actual before/after transformation")
    for label in [0, 1]:
        ax[1].hist(ct.safe_price_mean.to_numpy()[y == label], bins=60, density=True, histtype="step", label=f"target={label}")
    ax[1].legend(); ax[1].set(title="Transformed class distributions", xlabel="Safe price mean")
    cor = ct[["safe_price_mean", "safe_price_order_equal", "safe_price_category_equal", "safe_qty_mean",
              "price_change_matched", "safe_price_sd"]].corr(method="spearman", min_periods=500)
    im = ax[2].imshow(cor, vmin=-1, vmax=1, cmap="coolwarm")
    ax[2].set_xticks(range(len(cor)), cor.columns, rotation=90); ax[2].set_yticks(range(len(cor)), cor.columns)
    ax[2].set_title("Pairwise-complete Spearman"); fig.colorbar(im, ax=ax[2])
    figsave(fig, "01_price_before_after"); csv(cor.reset_index(), "01_candidate_correlation")
    js(dict(reference_fit="Diagnostic training customers only; cumulative observations strictly before query calendar month",
        diagnostic_seed=20261014, folds=5, self_checks="reference recomputation, validation perturbation, future perturbation passed",
        caveat="Standalone diagnostic folds are not sklearn folds used by v5. Refit references inside actual future model folds."),
       "01_preprocessing_manifest")

if __name__ == "__main__":
    main()
