"""A fitted approximation of observed rules; its seed is ours, not the dataset seed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_workflow(audit, n=10000, demo_seed=91403):
    rng = np.random.Generator(np.random.PCG64(demo_seed))
    mixture = audit["basic_rules"]["workflow_lag_mixture"]
    route = rng.choice(3, size=n, p=mixture["candidate_weights"])
    bounds = mixture["components"]
    low = np.array([r["low_hours"] for r in bounds])[route]
    high = np.array([r["high_hours"] for r in bounds])[route]
    # Formatting application timestamps to seconds explains ceil on observed lags.
    raw_hours = rng.uniform(low, high)
    return pd.DataFrame(
        {
            "route": route,
            "unrounded_delay_hours": raw_hours,
            "observed_delay_seconds": np.ceil(raw_hours * 3600).astype(int),
        }
    )


def simulate_price_measurements(
    audit, n_customers=2000, orders_per_customer=12, demo_seed=73103
):
    rng = np.random.Generator(np.random.PCG64(demo_seed))
    params = audit["ecommerce"]
    categories = list(params["category_log_intercepts"])
    category_intercept = np.array(
        [params["category_log_intercepts"][c] for c in categories]
    )
    # This is a deliberately simple price-only model; order/category probabilities are approximate.
    latent = rng.normal(
        0, params["estimated_customer_effect_prior_sd"], size=n_customers
    )
    customer = np.repeat(np.arange(n_customers), orders_per_customer)
    order = np.arange(len(customer))
    order_effect = rng.normal(
        0, params["estimated_shared_order_log_noise_sd"], size=len(order)
    )
    items = rng.choice([1, 2, 3], size=len(order), p=[0.65, 0.24, 0.11])
    item_order = np.repeat(order, items)
    item_customer = customer[item_order]
    category = rng.integers(0, len(categories), size=len(item_order))
    quantity = rng.choice(
        [1, 2, 3, 4], size=len(item_order), p=[0.735, 0.189, 0.057, 0.019]
    )
    noise = rng.normal(0, params["estimated_item_log_noise_sd"], size=len(item_order))
    pre_round_unit = np.exp(
        category_intercept[category]
        + latent[item_customer]
        + order_effect[item_order]
        + noise
    )
    unit = np.maximum(5000, np.rint(pre_round_unit / 1000) * 1000).astype(int)
    result = pd.DataFrame(
        {
            "synthetic_customer_index": item_customer,
            "synthetic_order_index": item_order,
            "product_category": np.array(categories)[category],
            "quantity": quantity,
            "pre_round_unit_price": pre_round_unit,
            "unit_price": unit,
            "item_total": unit * quantity,
            "known_demo_latent": latent[item_customer],
        }
    )
    result["order_total"] = result.groupby(
        "synthetic_order_index"
    ).item_total.transform("sum")
    return result
