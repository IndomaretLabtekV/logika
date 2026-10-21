"""Use pre-decision basket preferences, including cancelled baskets, as price signals."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from optimize_ks import ROOT, OUT


def main():
    base = pd.read_parquet(
        ROOT / "data/processed/application_features_v4.parquet"
    ).set_index("customer_id")
    events = pd.read_csv(
        ROOT / "data/ecommerce_transactions.csv",
        parse_dates=["transaction_datetime"],
    )
    events = events.merge(
        base[["decision_datetime"]],
        left_on="customer_id",
        right_index=True,
        validate="many_to_one",
    )
    future_rows = int(events.transaction_datetime.ge(events.decision_datetime).sum())
    assert future_rows == 0
    events = events[events.quantity.gt(0)].copy()
    events["age"] = (
        events.decision_datetime - events.transaction_datetime
    ).dt.total_seconds() / 86400
    events["log_price"] = np.log1p(events.item_total / events.quantity)
    train = events.customer_id.isin(base.index[base.split.eq("train")])
    reference = events[train].groupby("product_category").log_price.median()
    events["relative_price"] = events.log_price - events.product_category.map(reference)
    blocks = []
    for status, mask in [
        ("pooled", np.ones(len(events), dtype=bool)),
        ("cancelled", events.order_status.eq("cancelled")),
    ]:
        for window in [None, 180]:
            part = (
                events[mask] if window is None else events[mask & events.age.le(window)]
            )
            tag = f'{status}_{"all" if window is None else "180d"}'
            grouped = part.groupby("customer_id").relative_price
            summary = grouped.agg(["mean", "median", "std", "count"])
            summary.columns = [f"v5_price_{tag}_{c}" for c in summary]
            orders = (
                part.groupby(["customer_id", "order_id"])
                .relative_price.mean()
                .groupby(level=0)
            )
            summary[f"v5_price_{tag}_equal_order_mean"] = orders.mean()
            category = (
                part.groupby(["customer_id", "product_category"])
                .relative_price.mean()
                .unstack()
            )
            summary[f"v5_price_{tag}_equal_category_mean"] = category.mean(axis=1)
            blocks.append(summary)
    # Count distinct events, including cancellations; repeated item rows are never orders.
    orders = events.drop_duplicates(["customer_id", "order_id"]).copy()
    orders["completed"] = orders.order_status.eq("completed").astype(int)
    for window in [90, 180, 365]:
        part = orders[orders.age.le(window)]
        summary = part.groupby("customer_id").agg(
            n=("order_id", "size"), completed=("completed", "sum")
        )
        prior = float(
            part[
                part.customer_id.isin(base.index[base.split.eq("train")])
            ].completed.mean()
        )
        summary[f"v5_ecom_{window}d_completion_posterior"] = (
            summary.completed + 12 * prior
        ) / (summary.n + 12)
        blocks.append(summary[[f"v5_ecom_{window}d_completion_posterior"]])
    # A cross-feature measures purchasing preference relative to declared income.
    preference = pd.concat(blocks, axis=1).reindex(base.index)
    preference["v5_price_income_residual"] = (
        preference.v5_price_pooled_all_mean - np.log1p(base.declared_monthly_income)
    )
    result = base.join(preference).replace([np.inf, -np.inf], np.nan)
    assert result.application_id.equals(base.application_id) and result.target.equals(
        base.target
    )
    assert result.index.is_unique and result.columns.is_unique
    result.reset_index().to_parquet(
        ROOT / "data/processed/application_features_v5.parquet", index=False
    )
    destination = OUT / "v5"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "feature_audit.json").write_text(
        json.dumps(
            {
                "future_ecommerce_rows": future_rows,
                "target_used": False,
                "normalization_population": "Training customer baskets only; target labels excluded.",
                "cancelled_basket_use": "Observed product preference and completion rate; never spending.",
                "category_log_price_reference": reference.to_dict(),
                "new_features": preference.columns.tolist(),
                "rows": len(result),
            },
            indent=2,
        )
    )
    print(
        f"Saved {len(result):,} rows and {len(preference.columns)} new features.",
        flush=True,
    )


if __name__ == "__main__":
    main()
