"""Strict fitting-partition feature pools, without rebuilding frozen builders.

``build_pool(fit_application_ids, output_dir, *, include_teammate=True)`` returns
``{"parquet": Path, "manifest_path": Path, "manifest": dict}``. The parquet has
100,000 canonical applications, sorted by application_id. ``main_columns`` and
``union_columns`` in the manifest are the only model-input allowlists. Targets
outside the supplied official-training membership are null; is_train still
means the ORIGINAL competition split. Official query labels are never exported.

The audited main cache contributes only customer-local/nonprice predictors.
Every pp_* column is replaced by v5_features with fitting-customer references
and its expanding month-before-decision rule. Teammate inputs are exclusively
the verified analyze/extend prefix: these stages use customer/loan/order-local
reductions, not population numerical references. V3/V4/V5 are never loaded.
Their dynamic pivots are constrained to the published dictionary domains.

Raw CSVs are audited one at a time; only e-commerce is retained for price
construction. Estimated practical peak is 2-4 GiB (not a measured guarantee):
~0.2 GiB main matrix, ~0.6 GiB prefix, and e-commerce/join temporaries dominate.
The manifest records actual output logical bytes, not an invented peak RSS.
No model training, feature selection, category fitting, or label file occurs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import polars as pl

try:
    from . import features
except ImportError:
    import features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
MAIN_CACHE = ROOT / "data_pq/features.parquet"
PREFIX_ROOT = ROOT / "results/v56/teammate_features/outer_s42_f0"
PREFIX_CACHE = PREFIX_ROOT / "data/processed/prefix_features_extended.parquet"
RAW_TABLES = (
    "train_applications", "test_applications", "customer_profiles",
    "previous_applications", "repayment_history", "account_behavior",
    "bureau_observations", "ecommerce_transactions", "sample_submission",
)
APPLICATION_CATEGORIES = ("loan_purpose", "region", "employment_type", "housing_status")
METADATA = {"application_id", "customer_id", "target", "is_train", "split",
            "decision_datetime", "application_datetime", "decision_date"}
EXCLUDED_PREFIX_STRINGS = ("v2_prev_last_decision", "v2_ecom_last_payment_method")
# Hashes pin the exact caches and source bodies reviewed for this adapter. A new
# artifact requires a new audit, not an automatic blessing of its current bytes.
MAIN_SHA256 = "bd7ca1d8d88c53f22cd1589e60ddf9bbb5686d1fb5f6e214ff251505f8b9b034"
PREFIX_SHA256 = "1f60d75eae3cfae92e603d3795c0673dbd5343200dc5360c53279f5aae08fef3"
PREFIX_SOURCE_SHA256 = {
    "scripts/analyze.py": "7048832de175936c2f9a87899e54229777261f96f4607a8eb6071477664cad75",
    "scripts/extend_features.py": "f963f9749b478d587cc583753e006e47bbc340b2e4214955357c32723a6d3880",
}
RAW_SHA256 = {
    "train_applications": "b60df5704bcd3713314a31c5382db22f41d38971bd76906b15f45a38edff509b",
    "test_applications": "113f6ee4ff15e4c8bf14fb9a85fc399b906d1c569f6a715675a5814da2e02dfe",
    "customer_profiles": "bcffa8c5558e81d9b359d2785345af218c459704f1c416b1ecbbe3a0ccf73f5d",
    "previous_applications": "faa0bc4552afee73cf58481234a94834bf0b786a195cbb979883c15642c39560",
    "repayment_history": "b0b00e18de2c05233653b3627bc7f9d679644e84c56680284766d71c5e279877",
    "account_behavior": "2f46be03496583b0231591194e13bea7593e87323056dfead6b9b19c48871924",
    "bureau_observations": "e37b8a76fe4f97ee6f1fd78ab26ffb0e78e464714f77d961062e03d757d057a0",
    "ecommerce_transactions": "495b3859f2146f87e7f8a383b308770af3d48342e3e1acdb45d7f1bed2bab742",
    "sample_submission": "e1d1c105de6bc0906025366f9839e5f97e19f8efb2167e6cb5fef710ad91679d",
}
DOMAINS = {
    "loan_purpose": ("education", "home_improvement", "household", "working_capital", "medical"),
    "region": ("Region_A", "Region_B", "Region_C", "Region_D", "Region_E"),
    "employment_type": ("salaried", "self_employed", "variable_income"),
    "housing_status": ("owned", "rented", "family"),
    "decision": ("approved", "rejected", "cancelled"),
    "bureau_provider": ("Bureau_A", "Bureau_B", "Bureau_C"),
    "product_category": ("groceries", "fashion", "electronics", "home", "health", "education", "personal_care"),
    "payment_method": ("COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"),
    "order_status": ("completed", "cancelled"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_raw(name):
    return pl.read_csv(RAW / f"{name}.csv", try_parse_dates=True, infer_schema_length=100000)


def validate_domains(frame):
    """Reject unknown/null dictionary categories; never discover a vocabulary."""
    for column, values in DOMAINS.items():
        if column in frame.columns:
            bad = frame.filter(pl.col(column).is_null() | ~pl.col(column).is_in(values))
            _require(not bad.height, f"Unknown/null dictionary category: {column}")


def _dictionary_contract():
    text = (RAW / "DATA_DICTIONARY.md").read_text()
    for column, expected in DOMAINS.items():
        definitions = re.findall(r"^\| `" + re.escape(column) + r"` \|.*?Values: ([^.|]+)", text, re.M)
        _require(bool(definitions), f"Missing dictionary domain: {column}")
        for definition in definitions:
            _require(tuple(v.strip() for v in definition.split(",")) == expected,
                     f"Dictionary domain changed: {column}")
    _require(tuple(features.PP_CATS) == DOMAINS["product_category"], "Price category domain changed")


def canonical_applications(train, test):
    """Assemble authoritative IDs/labels without consulting any feature cache."""
    _require("target" not in test.columns, "Official test data unexpectedly contains target")
    apps = pl.concat([
        train.with_columns(pl.lit(1, dtype=pl.Int8).alias("is_train")),
        test.with_columns(pl.lit(None, dtype=train.schema["target"]).alias("target"),
                          pl.lit(0, dtype=pl.Int8).alias("is_train")),
    ], how="diagonal_relaxed").sort("application_id")
    for column in ("application_id", "customer_id"):
        _require(apps[column].null_count() == 0 and apps[column].n_unique() == apps.height,
                 f"Canonical {column} must be nonnull and unique")
    _require(train["target"].null_count() == 0 and set(train["target"].to_list()) <= {0, 1},
             "Official training labels must be binary and nonnull")
    _require(not apps.filter(pl.col("application_datetime").is_null() |
                             pl.col("decision_datetime").is_null() |
                             (pl.col("application_datetime") > pl.col("decision_datetime"))).height,
             "Invalid selected application timestamps")
    validate_domains(apps)
    return apps


def fitting_customers(apps, fit_application_ids):
    """Validate nonempty unique official-train membership and return sorted IDs."""
    ids = list(fit_application_ids)
    _require(bool(ids) and all(isinstance(value, str) for value in ids),
             "Fitting application IDs must be nonempty strings")
    _require(len(ids) == len(set(ids)), "Fitting application IDs must be unique")
    official = set(apps.filter(pl.col("is_train") == 1)["application_id"].to_list())
    _require(set(ids) <= official, "Fitting membership must be official training applications")
    return sorted(ids), sorted(apps.filter(pl.col("application_id").is_in(ids))["customer_id"].to_list())


def erase_nonfit_targets(frame, fit_application_ids):
    """Erase labels, not original train/test alignment; return a new frame."""
    ids, _ = fitting_customers(frame, fit_application_ids)
    return frame.with_columns(pl.when(pl.col("application_id").is_in(ids))
                              .then(pl.col("target")).otherwise(None).alias("target"))


def _aligned(frame, apps, columns):
    """Exact ID-keyed equality, allowing only physical dtype differences."""
    _require(frame.height == apps.height, "Cache row count differs from canonical applications")
    for key in ("application_id", "customer_id"):
        _require(frame[key].null_count() == 0 and frame[key].n_unique() == frame.height,
                 f"Cache {key} is null or duplicated")
    frame = frame.sort("application_id")
    expected = apps.sort("application_id").select(columns)
    actual = frame.select([pl.col(c).cast(expected.schema[c]) for c in columns])
    _require(actual.equals(expected, null_equal=True), f"Cache/raw mismatch: {columns}")
    return frame


def audit_repayment(rep, apps):
    """Reconstruct both snapshot DPD fields from dates and amounts, exactly.

    Current DPD is date(cutoff)-date(due) for an outstanding installment, zero
    otherwise. Fully-paid max DPD is max(date(last_payment)-date(due)-1, 0);
    unpaid max DPD equals current. Null/future dates or mismatches are fatal.
    """
    r = _cutoff_join(rep, apps)
    _before_cutoff(r, "due_date", nullable=False)
    _before_cutoff(r, "last_payment_datetime", nullable=True)
    _require(not r.filter(pl.col("amount_due").is_null() | pl.col("amount_paid").is_null() |
                          (pl.col("amount_due") <= 0) | (pl.col("amount_paid") < 0) |
                          (pl.col("amount_paid") > pl.col("amount_due"))).height,
             "Invalid repayment snapshot amounts")
    paid = pl.col("amount_paid") >= pl.col("amount_due")
    _require(not r.filter(paid & pl.col("last_payment_datetime").is_null()).height,
             "Fully paid installment lacks observed payment time")
    current = pl.when(~paid).then((pl.col("cutoff").dt.date() - pl.col("due_date").dt.date())
                                 .dt.total_days()).otherwise(0)
    maximum = pl.when(paid).then(((pl.col("last_payment_datetime").dt.date() -
                                  pl.col("due_date").dt.date()).dt.total_days() - 1)
                                .clip(lower_bound=0)).otherwise(current)
    r = r.with_columns(current.alias("_current"), maximum.alias("_maximum"))
    counts = r.select(
        (pl.col("days_past_due").is_null() | (pl.col("days_past_due") != pl.col("_current")))
        .sum().alias("current_mismatches"),
        (pl.col("max_days_past_due").is_null() | (pl.col("max_days_past_due") != pl.col("_maximum")))
        .sum().alias("maximum_mismatches"),
    ).row(0, named=True)
    _require(not any(counts.values()), f"Repayment DPD reconstruction failed: {counts}")
    return {"rows": rep.height, **counts, "future_payment_rows": 0}


def _cutoff_join(frame, apps):
    joined = frame.join(apps.select("customer_id", pl.col("decision_datetime").alias("cutoff")),
                        on="customer_id", how="left", validate="m:1")
    _require(joined["cutoff"].null_count() == 0, "Raw table contains unknown/null customer IDs")
    return joined


def _before_cutoff(frame, column, *, nullable=False):
    bad = pl.col(column) >= pl.col("cutoff")
    if not nullable:
        bad = bad | pl.col(column).is_null()
    _require(not frame.filter(bad).height, f"Unobserved/future raw timestamp: {column}")


def _audit_raw(apps, main):
    audits = {}
    ecommerce = None
    for name in RAW_TABLES:
        if name in {"train_applications", "test_applications", "sample_submission"}:
            continue
        frame = _read_raw(name)
        validate_domains(frame)
        if name == "customer_profiles":
            _require(frame.height == apps.height and frame["customer_id"].n_unique() == apps.height,
                     "Profiles must cover each canonical customer exactly once")
            profile_apps = apps.select("application_id", "customer_id").join(frame, on="customer_id",
                                                                              how="left", validate="1:1")
            _require(set(frame["customer_id"].to_list()) == set(apps["customer_id"].to_list()),
                     "Profile customer IDs differ from canonical customers")
            direct = [c for c in frame.columns if c in main.columns and c != "customer_id"]
            _aligned(main, profile_apps, ["application_id", "customer_id", *direct])
            audits[name] = {"rows": frame.height}
            continue
        if name == "repayment_history":
            audits[name] = audit_repayment(frame, apps)
            _require(frame.height == 1361341, "Repayment cache audit expects all 1,361,341 rows")
            continue
        joined = _cutoff_join(frame, apps)
        if name == "account_behavior":
            _require(not joined.filter(pl.col("month").is_null() |
                         (pl.col("month").dt.day() != 1) |
                         (pl.col("month").dt.offset_by("1mo").cast(pl.Datetime) > pl.col("cutoff"))).height,
                     "Account snapshot contains an incomplete calendar month")
        else:
            dates = {"previous_applications": ("application_datetime", "decision_datetime"),
                     "bureau_observations": ("report_datetime", "available_datetime"),
                     "ecommerce_transactions": ("transaction_datetime",)}[name]
            for column in dates:
                _before_cutoff(joined, column)
            if name == "previous_applications":
                _require(not joined.filter(pl.col("decision_datetime") < pl.col("application_datetime")).height,
                         "Historical decision precedes its application")
            if name == "bureau_observations":
                _require(not joined.filter(pl.col("available_datetime") < pl.col("report_datetime")).height,
                         "Bureau availability precedes report")
        audits[name] = {"rows": frame.height, "future_rows": 0}
        if name == "ecommerce_transactions":
            ecommerce = frame
        del joined, frame
    return audits, ecommerce


def _verify_prefix(raw_hashes, main_hash):
    mp = PREFIX_ROOT / "manifest.json"
    _require(PREFIX_CACHE.is_file() and mp.is_file(),
             f"Verified prefix cache required at {PREFIX_CACHE}; no implicit rebuild is permitted")
    manifest = json.loads(mp.read_text())
    inputs = manifest["inputs"]
    _require(manifest["status"] == "complete" and manifest["inputs_sha256"] == _digest(inputs),
             "Prefix metadata is incomplete or its input fingerprint is inconsistent")
    _require(inputs["raw_sha256"] == raw_hashes, "Prefix raw hashes differ from canonical CSVs")
    _require(inputs["id_split_cache_sha256"] == main_hash, "Prefix canonical ID cache hash differs")
    # Verify all recorded source bytes, even unused later stages; only the two
    # pinned prefix source bodies confer the no-population-reference guarantee.
    for relative, expected in inputs["source_sha256"].items():
        path = (ROOT / relative).resolve()
        _require(path.is_relative_to(ROOT), "Prefix source path escapes repository")
        _require(sha256(path) == expected, f"Prefix source provenance changed: {relative}")
    for relative, expected in PREFIX_SOURCE_SHA256.items():
        _require(inputs["source_sha256"].get(relative) == expected,
                 f"Unaudited prefix source version: {relative}")
    actual = sha256(PREFIX_CACHE)
    _require(actual == PREFIX_SHA256 == manifest["stages"]["extended"]["sha256"],
             "Prefix stage bytes do not match the audited prefix")
    return {"path": str(PREFIX_CACHE.relative_to(ROOT)), "sha256": actual,
            "metadata_sha256": sha256(mp), "sources": PREFIX_SOURCE_SHA256}


def replace_price_features(main, apps, ecommerce, fit_application_ids):
    """Customer-local main cache plus strict expanding fitting-only pp_* values."""
    _, customers = fitting_customers(apps, fit_application_ids)
    prices = features.v5_features(apps, {"ecommerce_transactions": ecommerce},
                                  reference_customers=customers)
    price_columns = [c for c in prices.columns if c.startswith("pp_")]
    old_columns = [c for c in main.columns if c.startswith("pp_")]
    _require(set(price_columns) == set(old_columns), "Strict price schema differs from canonical main232")
    return main.drop(old_columns).join(prices.select("customer_id", *price_columns),
                                       on="customer_id", how="left", validate="1:1", maintain_order="left")


def _feature_columns(frame):
    result = []
    for name, dtype in frame.schema.items():
        if name in METADATA:
            continue
        _require(not name.endswith(("_id", "_date", "_datetime")), f"Opaque ID/date predictor: {name}")
        _require(dtype.is_numeric() or dtype == pl.Boolean or
                 (name in APPLICATION_CATEGORIES and dtype == pl.String),
                 f"Unsupported main predictor: {name} ({dtype})")
        result.append(name)
    _require(set(APPLICATION_CATEGORIES) <= set(result), "Missing shared application categories")
    return sorted(result)


def validate_pivot_columns(columns, *, teammate=False):
    """Validate pivot names against dictionary domains, not observed query values."""
    providers, categories, methods = (DOMAINS[c] for c in
                                      ("bureau_provider", "product_category", "payment_method"))
    if not teammate:
        groups = {
            "bp_": {f"bp_{stat}_{p.removeprefix('Bureau_').lower()}"
                    for stat in ("score", "util", "inq", "delinq", "out") for p in providers},
            "ec_cat_": {f"ec_cat_{c}" for c in categories},
        }
    else:
        groups = {
            "bureau_Bureau_": {f"bureau_{p}_{stat}" for p in providers
                               for stat in ("score", "utilization", "outstanding_balance", "report_age_days")},
            "v2_Bureau_": {f"v2_{p}_latest_{stat}" for p in providers for stat in
                           ("score", "active_accounts", "outstanding_balance", "revolving_credit_limit",
                            "revolving_outstanding_balance", "utilization", "inquiries_last_6m",
                            "delinquent_accounts_last_12m", "age_days", "lag_days")},
            "ecom_category_share_": {f"ecom_category_share_{c}" for c in categories},
            "ecom_payment_share_": {f"ecom_payment_share_{m}" for m in methods},
        }
        for days in (90, 180):
            groups[f"v2_ecom_{days}d_category_"] = {f"v2_ecom_{days}d_category_{c}" for c in categories}
            groups[f"v2_ecom_{days}d_payment_spend_share_"] = {
                f"v2_ecom_{days}d_payment_spend_share_{m}" for m in methods}
    for prefix, expected in groups.items():
        actual = {c for c in columns if c.startswith(prefix)}
        _require(actual == expected, f"Pivot domain mismatch: {prefix}; missing={sorted(expected - actual)}, "
                                    f"unknown={sorted(actual - expected)}")


def add_prefix(main, prefix, apps):
    """Add numeric non-overlaps with t__ names; retain main's four categories."""
    prefix = prefix.with_columns(pl.col("split").replace_strict({"train": 1, "test": 0},
                                                               return_dtype=pl.Int8).alias("is_train"))
    prefix = _aligned(prefix, apps, ["application_id", "customer_id", "target", "is_train",
                                     "decision_datetime", "application_datetime"])
    validate_domains(prefix)
    shared = [c for c in APPLICATION_CATEGORIES]
    _aligned(prefix, main, ["application_id", "customer_id", *shared])
    additions, overlaps, excluded = [], [], []
    for name, dtype in prefix.schema.items():
        if name in METADATA or name in APPLICATION_CATEGORIES:
            excluded.append(name)
        elif name in EXCLUDED_PREFIX_STRINGS:
            _require(dtype == pl.String, f"Expected excluded text predictor: {name}")
            excluded.append(name)
        elif dtype.is_numeric() or dtype == pl.Boolean:
            _require(not name.endswith(("_id", "_date", "_datetime")), f"Prefix ID/date predictor: {name}")
            if name in main.columns:
                overlaps.append(name)
            else:
                additions.append(name)
        else:
            raise ValueError(f"Unsupported prefix predictor: {name} ({dtype})")
    _require(not any(c.startswith(("v3_", "v4_", "v5_")) for c in prefix.columns),
             "Reference-stage columns are forbidden in prefix input")
    renamed = {c: "t__" + c for c in additions}
    _require(not set(renamed.values()) & set(main.columns), "Namespaced prefix collision")
    joined = main.join(prefix.select("application_id", *additions).rename(renamed),
                       on="application_id", how="left", validate="1:1", maintain_order="left")
    return joined, {"added_columns": sorted(renamed.values()), "overlap_excluded": sorted(overlaps),
                    "metadata_and_strings_excluded": sorted(excluded)}


def _validate_output(frame, apps, fit_ids, main_columns, union_columns):
    expected = erase_nonfit_targets(apps, fit_ids)
    _aligned(frame, expected, ["application_id", "customer_id", "target", "is_train", "decision_datetime"])
    _require(set(frame.columns) == METADATA.intersection(frame.columns) | set(union_columns),
             "Output contains unlisted predictors")
    _require(set(main_columns) <= set(union_columns) and not set(union_columns) & METADATA,
             "Invalid feature allowlists")
    _require(len(union_columns) == len(set(union_columns)), "Duplicate union feature names")


def build_pool(fit_application_ids, output_dir, *, include_teammate=True):
    """Build or hash-validate one immutable, fitting-specific local feature pool.

    Returns Path-valued ``parquet``/``manifest_path`` and the parsed ``manifest``.
    Existing mismatched/partial outputs fail instead of being overwritten. Pass
    include_teammate=False explicitly for main232 alone. This call intentionally
    performs raw audits and price aggregation; importing the module does not.
    """
    output = Path(output_dir).resolve()
    protected = (ROOT, RAW, ROOT / "data_pq", PREFIX_ROOT)
    _require(output != ROOT and not any(output.is_relative_to(p) for p in protected[1:]),
             "Output directory must not overwrite canonical inputs or historical prefix")
    parquet, mp = output / "strict_features.parquet", output / "manifest.json"
    _require(not output.exists() or mp.is_file() or not any(output.iterdir()),
             "Refusing to adopt a nonempty output directory without a strict manifest")
    apps = canonical_applications(_read_raw("train_applications"), _read_raw("test_applications"))
    _require(apps.height == 100000 and apps.filter(pl.col("is_train") == 1).height == 70000,
             "Expected canonical 70,000 train and 30,000 test applications")
    fit_ids, fit_customers = fitting_customers(apps, fit_application_ids)
    _dictionary_contract()
    raw_hashes = {name: sha256(RAW / f"{name}.csv") for name in RAW_TABLES}
    _require(raw_hashes == RAW_SHA256, "Canonical raw bytes changed since the frozen main/prefix audit")
    main_hash = sha256(MAIN_CACHE)
    _require(main_hash == MAIN_SHA256, "Main cache is not the audited canonical main232 artifact")
    prefix_info = _verify_prefix(raw_hashes, main_hash) if include_teammate else None
    inputs = {
        "version": 1, "fit_application_ids": fit_ids, "fit_ids_sha256": _digest(fit_ids),
        "fit_customer_ids_sha256": _digest(fit_customers), "include_teammate": bool(include_teammate),
        "raw_sha256": raw_hashes, "main_cache_sha256": main_hash, "prefix": prefix_info,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                          (Path(__file__).resolve(), Path(features.__file__).resolve(), RAW / "DATA_DICTIONARY.md")},
        "polars_version": pl.__version__,
    }
    fingerprint = _digest(inputs)
    if mp.exists():
        manifest = json.loads(mp.read_text())
        _require(manifest.get("inputs_sha256") == fingerprint and manifest.get("inputs") == inputs,
                 "Pool fitting membership/input/source changed; use a new output directory")
        _require(manifest.get("status") == "complete" and parquet.is_file(), "Incomplete strict feature pool")
        _require(sha256(parquet) == manifest["output_sha256"], "Strict output hash mismatch")
        frame = pl.read_parquet(parquet)
        _validate_output(frame, apps, fit_ids, manifest["main_columns"], manifest["union_columns"])
        return {"parquet": parquet, "manifest_path": mp, "manifest": manifest}
    main = pl.read_parquet(MAIN_CACHE)
    direct = [c for c in apps.columns if c in main.columns]
    main = _aligned(main, apps, direct)
    main_columns = _feature_columns(main)
    validate_pivot_columns(main.columns)
    _require(len(main_columns) == 232, "Canonical main feature pool must contain 232 predictors")
    # Fixed cache hashes plus raw-domain checks freeze historical pivot columns;
    # no query-specific category can add a dimension to either model family.
    audits, ecommerce = _audit_raw(apps, main)
    main = replace_price_features(main, apps, ecommerce, fit_ids)
    del ecommerce
    prefix_audit = None
    if include_teammate:
        prefix = pl.read_parquet(PREFIX_CACHE)
        validate_pivot_columns(prefix.columns, teammate=True)
        main, prefix_audit = add_prefix(main, prefix, apps)
        del prefix
    union_columns = _feature_columns(main)
    frame = erase_nonfit_targets(main, fit_ids).select(
        "application_id", "customer_id", "decision_datetime", "target", "is_train", *union_columns,
    ).sort("application_id")
    _validate_output(frame, apps, fit_ids, main_columns, union_columns)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "strict_features.parquet.tmp"
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(parquet)
    manifest = {
        "status": "complete", "inputs": inputs, "inputs_sha256": fingerprint,
        "output": parquet.name, "output_sha256": sha256(parquet), "rows": frame.height,
        "fitting_rows": len(fit_ids), "erased_target_rows": frame.height - len(fit_ids),
        "main_columns": main_columns, "union_columns": union_columns,
        "application_categories": list(APPLICATION_CATEGORIES), "dictionary_domains": DOMAINS,
        "raw_audit": audits, "prefix_audit": prefix_audit,
        "reference_policy": "fitting customers only; reference event month < each query decision month",
        "label_policy": "nonfit target null; original is_train preserved; no official-label export",
        "prefix_policy": "customer-local analyze/extend only; numerical v3/v4/v5 references forbidden",
        "logical_output_bytes": frame.estimated_size(), "model_executed": False,
    }
    temp_manifest = output / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temp_manifest.replace(mp)
    # JSON normalization gives identical return types on first build and reuse.
    manifest = json.loads(mp.read_text())
    return {"parquet": parquet, "manifest_path": mp, "manifest": manifest}
