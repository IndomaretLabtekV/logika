"""Keys, cutoff, duplicate semantics, table grain and cross-split memorization audit."""
from deep_common import *
import gc

def main():
    A = apps()
    assert A["customer_id"].n_unique() == A.height == A["application_id"].n_unique()
    cut = A.select("customer_id", cutoff="decision_datetime")
    audit = []
    keys = {"train_applications": ["application_id"], "test_applications": ["application_id"],
            "customer_profiles": ["customer_id"], "previous_applications": ["application_id"],
            "repayment_history": ["installment_id"], "account_behavior": ["customer_id", "month"],
            "bureau_observations": ["bureau_observation_id"], "ecommerce_transactions": ["order_item_id"]}
    for name in TABLES:
        t = table(name)
        raw_n = pl.scan_csv(ROOT / "data" / (name + ".csv")).select(pl.len()).collect().item()
        assert raw_n == t.height, "raw/cache row-count mismatch"
        dup = t.height - t.select(keys[name]).unique().height
        orphan = t.select("customer_id").unique().join(cut.select("customer_id"), on="customer_id", how="anti").height
        future = 0
        if name not in ("train_applications", "test_applications", "customer_profiles"):
            tc = t.join(cut, on="customer_id", validate="m:1")
            expr = {"previous_applications": pl.col("decision_datetime"),
                    "repayment_history": pl.col("due_date"),
                    "account_behavior": pl.col("month").dt.offset_by("1mo").cast(pl.Datetime),
                    "bureau_observations": pl.col("available_datetime"),
                    "ecommerce_transactions": pl.col("transaction_datetime")}[name]
            future = tc.filter(expr > pl.col("cutoff")).height
            if name == "repayment_history":
                assert tc.filter(pl.col("last_payment_datetime") >= pl.col("cutoff")).height == 0
            del tc
        audit.append(dict(table=name, rows=t.height, columns=t.width, duplicate_keys=dup,
                          orphan_customers=orphan, post_cutoff_rows=future))
        assert dup == orphan == future == 0
        del t; gc.collect()
    save_table(audit, "00_integrity")
    ec = table("ecommerce_transactions")
    orders = ec.group_by("order_id").agg(
        pl.col("customer_id", "order_total", "order_status").first(),
        item_sum=pl.col("item_total").sum(), n_items=pl.len(),
        consistent=pl.struct("customer_id", "transaction_datetime", "order_total", "order_status", "payment_method").n_unique())
    assert orders.filter((pl.col("item_sum") != pl.col("order_total")) | (pl.col("consistent") != 1)).height == 0
    sig = [c for c in ec.columns if c != "order_item_id"]
    apparent = ec.filter(ec.select(sig).is_duplicated())
    collapsed = ec.unique(subset=sig).group_by("order_id").agg(item_sum_dedup=pl.col("item_total").sum())
    changed = orders.join(collapsed, on="order_id").filter(pl.col("item_sum_dedup") != pl.col("order_total"))
    save_table(changed.head(50), "00_do_not_deduplicate_items_examples")
    completed = orders.filter(pl.col("order_status") == "completed")
    naive = ec.filter(pl.col("order_status") == "completed")["order_total"].sum()
    correct = completed["order_total"].sum()
    prev = table("previous_applications")
    rep = table("repayment_history")
    assert rep.select("loan_id").unique().join(prev.select("loan_id").drop_nulls().unique(), on="loan_id", how="anti").height == 0
    selected = A.select("application_id")
    assert prev.select("application_id").join(selected, on="application_id", how="inner").height == 0
    valid_apps = pl.concat([selected, prev.select("application_id")]).unique()
    assert table("bureau_observations").select("application_id").unique().join(valid_apps, on="application_id", how="anti").height == 0
    f = features()
    featcols = [c for c in f.columns if c not in IDS]
    full = f.group_by(featcols).agg(n=pl.len(), n_splits=pl.col("is_train").n_unique())
    prof = table("customer_profiles").join(f.select("customer_id", "is_train"), on="customer_id", validate="1:1")
    twins = prof.group_by([c for c in prof.columns if c not in ("customer_id", "is_train")]).agg(n=pl.len(), n_splits=pl.col("is_train").n_unique())
    result = dict(orders=orders.height, completed_spend_correct=correct, naive_spend=naive,
                  naive_overcount_factor=naive / correct, identical_non_id_item_rows=apparent.height,
                  damaged_orders_if_content_dedup=changed.height,
                  full_feature_duplicate_groups=full.filter(pl.col("n") > 1).height,
                  full_feature_cross_split_groups=full.filter(pl.col("n_splits") > 1).height,
                  profile_cross_split_groups=twins.filter(pl.col("n_splits") > 1).height,
                  action="Deduplicate repeated order_total at order grain; keep distinct order_item_id rows.")
    save_json(result, "00_duplicate_decision")
    # IDs are opaque; sampled near-duplicates in the previous EDA are not proof of identity.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].barh([r["table"] for r in audit], [r["rows"] for r in audit]); ax[0].set_xscale("log")
    ax[0].set_title("Raw table sizes; all primary keys unique")
    ax[1].hist(orders["n_items"].to_numpy(), bins=np.arange(1, 9)-.5)
    ax[1].set_title("Items per order"); ax[1].set_xlabel("Count")
    ax[2].bar(["Correct\norder grain", "Wrong\nitem grain"], [correct / 1e12, naive / 1e12])
    ax[2].set_ylabel("Completed spend, trillion IDR"); ax[2].set_title("Repeated order_total must not be summed")
    figsave(fig, "00_grain_and_duplicates")

if __name__ == "__main__":
    self_check()
    main()
