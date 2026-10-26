"""Deterministic CPU-only regression checks; no real-data build or model runs.

Run from the repository root with:
    .venv/bin/python eda/strict_feature_pool_checks.py
"""
from datetime import datetime
import unittest

import polars as pl
from polars.testing import assert_frame_equal

try:
    from . import strict_feature_pool as pools
except ImportError:
    import strict_feature_pool as pools


def fixture():
    apps = pl.DataFrame({
        "application_id": ["A0", "A1", "A2", "A3"],
        "customer_id": ["C0", "C1", "C2", "C3"],
        "application_datetime": [datetime(2025, 1, 1)] * 4,
        "decision_datetime": [datetime(2025, 4, 1)] * 4,
        "loan_purpose": ["household"] * 4,
        "target": [0, 1, 1, None], "is_train": [1, 1, 1, 0],
    }).with_columns(pl.col("target").cast(pl.Int64))
    ecommerce = pl.DataFrame({
        "customer_id": [f"C{i}" for i in range(4) for _ in range(2)],
        "transaction_datetime": [datetime(2025, 1, 10), datetime(2025, 1, 20)] * 4,
        "item_total": [10, 20, 30, 40, 50, 60, 70, 80],
        "quantity": [1] * 8, "product_category": ["groceries"] * 8,
        "order_status": ["completed"] * 8,
    })
    legacy = pools.features.v5_features(apps, {"ecommerce_transactions": ecommerce})
    main = apps.join(legacy, on="customer_id", validate="1:1").with_columns(
        pl.lit("Region_A").alias("region"), pl.lit("salaried").alias("employment_type"),
        pl.lit("owned").alias("housing_status"), pl.lit(2.0).alias("local_numeric"),
    )
    return apps, ecommerce, main.sort("application_id")


class StrictPoolChecks(unittest.TestCase):
    def test_query_only_covariates_cannot_change_fit_or_other_query_references(self):
        apps, ecommerce, main = fixture()
        before = pools.replace_price_features(main, apps, ecommerce, ["A0", "A1"])
        changed = ecommerce.with_columns(
            pl.when(pl.col("customer_id") == "C2").then(pl.col("item_total") * 10000)
            .otherwise(pl.col("item_total")).alias("item_total"))
        after = pools.replace_price_features(main, apps, changed, ["A0", "A1"])
        # A2's own covariates may change A2, never another customer's reference.
        assert_frame_equal(before.filter(pl.col("application_id") != "A2").sort("application_id"),
                           after.filter(pl.col("application_id") != "A2").sort("application_id"))
        self.assertNotEqual(before.filter(pl.col("application_id") == "A2")["pp_mean"][0],
                            after.filter(pl.col("application_id") == "A2")["pp_mean"][0])

    def test_query_labels_are_irrelevant_to_all_price_features(self):
        apps, ecommerce, main = fixture()
        changed = apps.with_columns(pl.when(pl.col("application_id").is_in(["A2", "A3"]))
                                    .then(0).otherwise(pl.col("target")).alias("target"))
        before = pools.replace_price_features(main, apps, ecommerce, ["A0", "A1"])
        after = pools.replace_price_features(main, changed, ecommerce, ["A0", "A1"])
        assert_frame_equal(before.sort("application_id"), after.sort("application_id"))

    def test_reference_events_after_query_month_do_not_change_earlier_query(self):
        apps, ecommerce, main = fixture()
        apps = apps.with_columns(pl.when(pl.col("application_id") == "A2")
                                 .then(pl.lit(datetime(2025, 2, 1)))
                                 .otherwise(pl.col("decision_datetime")).alias("decision_datetime"))
        # Fitting customers have later selected decisions, but their later events
        # must not enter February's reference mean or variance.
        late = ecommerce.filter(pl.col("customer_id") == "C0").with_columns(
            pl.lit(datetime(2025, 3, 1)).alias("transaction_datetime"),
            (pl.col("item_total") * 1000000).alias("item_total"))
        before = pools.replace_price_features(main, apps, ecommerce, ["A0", "A1"])
        after = pools.replace_price_features(main, apps, pl.concat([ecommerce, late]), ["A0", "A1"])
        assert_frame_equal(before.filter(pl.col("application_id") == "A2"),
                           after.filter(pl.col("application_id") == "A2"))

    def test_target_erasure_preserves_original_split_and_original_frame(self):
        apps, _, _ = fixture()
        erased = pools.erase_nonfit_targets(apps, ["A1", "A0"])
        self.assertEqual(erased["target"].to_list(), [0, 1, None, None])
        self.assertEqual(erased["is_train"].to_list(), [1, 1, 1, 0])
        self.assertEqual(apps["target"].to_list(), [0, 1, 1, None])
        self.assertEqual(pools.fitting_customers(apps, ["A1", "A0"]),
                         (["A0", "A1"], ["C0", "C1"]))

    def test_invalid_fitting_membership_is_rejected(self):
        apps, _, _ = fixture()
        for ids in ([], ["A0", "A0"], ["A3"], ["unknown"], [1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                pools.erase_nonfit_targets(apps, ids)

    def test_unknown_dictionary_category_is_rejected_not_learned(self):
        with self.assertRaises(ValueError):
            pools.validate_domains(pl.DataFrame({"product_category": ["query_only_category"]}))
        pools.validate_domains(pl.DataFrame({"product_category": ["groceries"]}))
        with self.assertRaises(ValueError):
            pools.validate_domains(pl.DataFrame({"bureau_provider": ["Bureau_A", None]}))

    def test_pivot_columns_use_only_fixed_dictionary_domains(self):
        columns = [f"bp_{stat}_{provider.removeprefix('Bureau_').lower()}"
                   for stat in ("score", "util", "inq", "delinq", "out")
                   for provider in pools.DOMAINS["bureau_provider"]]
        columns += [f"ec_cat_{category}" for category in pools.DOMAINS["product_category"]]
        pools.validate_pivot_columns(columns)
        with self.assertRaises(ValueError):
            pools.validate_pivot_columns(columns + ["bp_score_query_only"])
        with self.assertRaises(ValueError):
            pools.validate_pivot_columns(columns[:-1])

    def test_canonical_labels_and_ids_are_checked_against_cache(self):
        apps, _, main = fixture()
        columns = ["application_id", "customer_id", "target", "is_train"]
        pools._aligned(main.reverse(), apps, columns)
        wrong = main.with_columns(pl.when(pl.col("application_id") == "A2").then(0)
                                  .otherwise(pl.col("target")).alias("target"))
        with self.assertRaises(ValueError):
            pools._aligned(wrong, apps, columns)
        duplicate = pl.concat([main.slice(0, 3), main.slice(0, 1)])
        with self.assertRaises(ValueError):
            pools._aligned(duplicate, apps, columns)

    def test_prefix_only_adds_namespaced_numeric_nonoverlaps(self):
        apps, _, main = fixture()
        prefix = main.select("application_id", "customer_id", "target", "decision_datetime",
                             "application_datetime", *pools.APPLICATION_CATEGORIES, "local_numeric").with_columns(
            pl.Series("split", ["train", "train", "train", "test"]),
            pl.Series("extra_numeric", [10., 20., 30., 40.]),
            pl.lit("approved").alias("v2_prev_last_decision"),
            pl.lit("COD").alias("v2_ecom_last_payment_method"))
        joined, audit = pools.add_prefix(main, prefix.reverse(), apps)
        self.assertEqual(audit["added_columns"], ["t__extra_numeric"])
        self.assertEqual(audit["overlap_excluded"], ["local_numeric"])
        self.assertEqual(joined["t__extra_numeric"].to_list(), [10., 20., 30., 40.])
        self.assertNotIn("t__v2_prev_last_decision", joined.columns)
        self.assertNotIn("t__loan_purpose", joined.columns)
        with self.assertRaises(ValueError):
            pools.add_prefix(main, prefix.with_columns(pl.lit("surprise").alias("unknown_string")), apps)
        with self.assertRaises(ValueError):
            pools.add_prefix(main, prefix.with_columns(pl.lit(1.).alias("v3_reference")), apps)

    def test_feature_allowlist_rejects_dates_ids_and_labels(self):
        _, _, main = fixture()
        columns = pools._feature_columns(main)
        self.assertFalse(set(columns) & pools.METADATA)
        for name, value in (("numeric_id", 1), ("event_datetime", datetime(2025, 1, 1))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pools._feature_columns(main.with_columns(pl.lit(value).alias(name)))

    def test_repayment_calendar_day_reconstruction_and_invalid_snapshots(self):
        apps, _, _ = fixture()
        apps = apps.with_columns(pl.lit(datetime(2025, 1, 20, 10)).alias("decision_datetime"))
        rep = pl.DataFrame({
            "customer_id": ["C0"] * 4,
            "due_date": [datetime(2025, 1, 10)] * 4,
            "last_payment_datetime": [datetime(2025, 1, 10, 18), datetime(2025, 1, 15, 9),
                                      None, datetime(2025, 1, 12, 12)],
            "amount_due": [100] * 4, "amount_paid": [100, 100, 0, 50],
            "days_past_due": [0, 0, 10, 10], "max_days_past_due": [0, 4, 10, 10],
        })
        self.assertEqual(pools.audit_repayment(rep, apps), {
            "rows": 4, "current_mismatches": 0, "maximum_mismatches": 0, "future_payment_rows": 0})
        for changed in (
            rep.with_columns((pl.col("max_days_past_due") + 1).alias("max_days_past_due")),
            rep.with_columns(pl.lit(datetime(2025, 1, 21)).alias("last_payment_datetime")),
            rep.with_columns(pl.lit(None, dtype=pl.Datetime).alias("last_payment_datetime")),
        ):
            with self.assertRaises(ValueError):
                pools.audit_repayment(changed, apps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
