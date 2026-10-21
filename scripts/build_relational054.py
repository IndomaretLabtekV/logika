"""Prepare capped, past-only child tables for a Kumo Relational experiment.

Opaque IDs are retained solely as graph keys. No target enters a related table.
Sampling is deterministic and uses timestamps/row position, never target labels.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pandas as pd

import research_goal054 as r

OUT = r.ROOT / "data/processed/relational_goal054"


def spans_and_recent(frame, time_column, spans=16, recent=4):
    frame = frame.sort_values(["customer_id", time_column], kind="mergesort").copy()
    grouped = frame.groupby("customer_id", sort=False, observed=True)
    position = grouped.cumcount()
    count = grouped[time_column].transform("size")
    frame["_history_bucket"] = (position * spans // count).astype(np.int16)
    take = ~frame.duplicated(["customer_id", "_history_bucket"]) | position.ge(
        count - recent
    )
    result = frame.loc[take].drop(columns="_history_bucket")
    assert result.groupby("customer_id", observed=True).size().max() <= spans + recent
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = r.ROOT / "data"
    manifest_file = OUT / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        assert manifest["builder_sha256"] == r.checksum(
            r.ROOT / "scripts/build_relational054.py"
        )
        for name, digest in manifest["output_sha256"].items():
            assert r.checksum(OUT / name) == digest
        print("RELATIONAL FEATURES CACHED", flush=True)
        return
    applications = pd.concat(
        [
            pd.read_csv(
                raw / f"{split}_applications.csv",
                usecols=[
                    "application_id",
                    "customer_id",
                    "application_datetime",
                    "decision_datetime",
                ],
                parse_dates=["application_datetime", "decision_datetime"],
            )
            for split in ["train", "test"]
        ],
        ignore_index=True,
    )
    assert (
        len(applications) == 100000
        and applications.customer_id.is_unique
        and applications.application_id.is_unique
    )
    assert applications.application_datetime.le(applications.decision_datetime).all()
    cutoffs = applications.set_index("customer_id").decision_datetime
    users = pd.read_csv(raw / "customer_profiles.csv")
    assert users.customer_id.is_unique and set(users.customer_id) == set(cutoffs.index)
    profile_columns = [c for c in users if c != "customer_id"]
    metadata = dict(
        rows=100000,
        customer_ids_unique=True,
        targets_in_related_tables=False,
        ids_are_graph_links_only=True,
        builder_sha256=r.checksum(r.ROOT / "scripts/build_relational054.py"),
        tables={},
        raw_sha256={},
        output_sha256={},
    )
    paths = ["customer_profiles.csv", "train_applications.csv", "test_applications.csv"]

    def save(name, frame, source, raw_rows, eligibility):
        assert "target" not in frame and "split" not in frame
        assert set(frame.customer_id).issubset(cutoffs.index)
        path = OUT / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        metadata["tables"][name] = dict(
            raw_rows=raw_rows,
            selected_rows=len(frame),
            customers=int(frame.customer_id.nunique()),
            columns=frame.columns.tolist(),
            eligibility=eligibility,
            maximum_rows_per_customer=int(
                frame.groupby("customer_id", observed=True).size().max()
            ),
        )
        metadata["output_sha256"][path.name] = r.checksum(path)
        paths.append(source)
        print("RELATED TABLE", name, len(frame), flush=True)

    account = pd.read_csv(
        raw / "account_behavior.csv",
        parse_dates=["month"],
        dtype={"customer_id": "category"},
    )
    cutoff = account.customer_id.map(cutoffs).astype("datetime64[ns]")
    account["month_end"] = account.month + pd.offsets.MonthEnd(0)
    eligible = account.month_end.lt(cutoff)
    assert eligible.all(), "Account data contains an incomplete/future month."
    counts = account.groupby("customer_id", observed=True).size()
    users["history_account_rows"] = (
        users.customer_id.map(counts).fillna(0).astype(np.int32)
    )
    n = len(account)
    account = (
        account.sort_values(["customer_id", "month"], kind="mergesort")
        .groupby("customer_id", observed=True, sort=False)
        .tail(12)
        .copy()
    )
    account["net_flow"] = account.inflows - account.outflows
    account["outflow_income_ratio"] = (
        account.outflows
        / account.income_like_credits.where(account.income_like_credits.gt(0))
    )
    account = account.drop(columns="month").replace([np.inf, -np.inf], np.nan)
    save(
        "accounts",
        account,
        "account_behavior.csv",
        n,
        "All months end strictly before decision; retain last 12",
    )
    del account
    gc.collect()

    bureau = pd.read_csv(
        raw / "bureau_observations.csv",
        parse_dates=["report_datetime", "available_datetime"],
        dtype={"customer_id": "category", "bureau_provider": "category"},
    )
    cutoff = bureau.customer_id.map(cutoffs).astype("datetime64[ns]")
    assert (
        bureau.report_datetime.lt(cutoff).all()
        and bureau.available_datetime.lt(cutoff).all()
    )
    counts = bureau.groupby("customer_id", observed=True).size()
    users["history_bureau_rows"] = (
        users.customer_id.map(counts).fillna(0).astype(np.int32)
    )
    n = len(bureau)
    bureau = (
        bureau.sort_values(
            ["customer_id", "bureau_provider", "report_datetime"], kind="mergesort"
        )
        .groupby(["customer_id", "bureau_provider"], observed=True, sort=False)
        .tail(4)
        .copy()
    )
    bureau["availability_lag_days"] = (
        bureau.available_datetime - bureau.report_datetime
    ).dt.total_seconds() / 86400
    bureau = bureau.drop(
        columns=["bureau_observation_id", "application_id", "available_datetime"]
    )
    save(
        "bureau",
        bureau,
        "bureau_observations.csv",
        n,
        "Report and availability strictly before decision; retain last 4 per provider",
    )
    del bureau
    gc.collect()

    events = pd.read_csv(
        raw / "ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
        dtype={
            "customer_id": "category",
            "product_category": "category",
            "payment_method": "category",
            "order_status": "category",
        },
    )
    cutoff = events.customer_id.map(cutoffs).astype("datetime64[ns]")
    assert events.transaction_datetime.lt(cutoff).all() and events.quantity.gt(0).all()
    counts = events.groupby("customer_id", observed=True).size()
    users["history_ecommerce_rows"] = (
        users.customer_id.map(counts).fillna(0).astype(np.int32)
    )
    n = len(events)
    events = spans_and_recent(events, "transaction_datetime")
    events["unit_price"] = events.item_total / events.quantity
    events["log_unit_price"] = np.log1p(events.unit_price)
    events = events.drop(columns=["order_id", "order_item_id"])
    save(
        "ecommerce",
        events,
        "ecommerce_transactions.csv",
        n,
        "Event strictly before decision; up to 16 history-spanning item rows plus last 4; cancelled baskets remain known preferences",
    )
    del events
    gc.collect()

    repayment = pd.read_csv(
        raw / "repayment_history.csv",
        parse_dates=["due_date", "last_payment_datetime"],
        dtype={"customer_id": "category"},
    )
    cutoff = repayment.customer_id.map(cutoffs).astype("datetime64[ns]")
    known = repayment.last_payment_datetime.isna() | repayment.last_payment_datetime.lt(
        cutoff
    )
    assert known.all(), "A repayment allocation occurs after cutoff."
    n = len(repayment)
    future_due = int(repayment.due_date.ge(cutoff).sum())
    repayment = repayment[repayment.due_date.lt(cutoff)].copy()
    counts = repayment.groupby("customer_id", observed=True).size()
    users["history_repayment_rows"] = (
        users.customer_id.map(counts).fillna(0).astype(np.int32)
    )
    repayment = (
        repayment.sort_values(["customer_id", "due_date"], kind="mergesort")
        .groupby("customer_id", observed=True, sort=False)
        .tail(16)
        .copy()
    )
    repayment["payment_fraction"] = repayment.amount_paid / repayment.amount_due.where(
        repayment.amount_due.gt(0)
    )
    repayment["payment_delay_days"] = (
        repayment.last_payment_datetime - repayment.due_date
    ).dt.total_seconds() / 86400
    repayment = repayment.drop(
        columns=["loan_id", "installment_id", "last_payment_datetime"]
    ).replace([np.inf, -np.inf], np.nan)
    save(
        "repayment",
        repayment,
        "repayment_history.csv",
        n,
        f"Last allocation before cutoff or missing; {future_due} future scheduled due rows excluded; retain last 16 past-due rows",
    )
    del repayment
    gc.collect()

    save(
        "users",
        users,
        "customer_profiles.csv",
        len(users),
        "Profile at selected decision; history row counts computed without targets",
    )
    metadata["profile_feature_columns"] = profile_columns
    metadata["cutoff_range"] = [str(cutoffs.min()), str(cutoffs.max())]
    for name in sorted(set(paths)):
        metadata["raw_sha256"][name] = r.checksum(raw / name)
    r.write_json(manifest_file, metadata)
    r.write_json(r.DEST / "relational_feature_manifest.json", metadata)
    print(
        "RELATIONAL FEATURES READY",
        json.dumps({k: v["selected_rows"] for k, v in metadata["tables"].items()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
