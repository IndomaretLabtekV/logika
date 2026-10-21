"""Local pretrained Kumo Tabular: leakage-safe context, matching V5 OOF folds.

Run with the separate Python 3.11 / environments/teman/requirements-kumo.txt environment.
No training data is sent to an API. Weights are downloaded from NVIDIA's Hub.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/optimization/kumo"
CORE_COLUMNS = [
    "v5_price_pooled_all_mean",
    "v5_price_pooled_all_equal_order_mean",
    "v4_price_all_mean",
    "v4_price_180d_mean",
    "v2_ecom_completed_per_observed_month",
    "v2_ecom_30d_orders",
    "v2_ecom_order_age_days_median",
    "ecom_last_purchase_recency_days",
    "v2_bureau_history_score_mean",
    "v2_bureau_180d_score_mean",
    "v2_bureau_history_delinquent_accounts_last_12m_mean",
    "bureau_utilization_max",
    "bureau_inquiries_max",
    "v3_repay_all_late_1_posterior_24",
    "v3_repay_all_late_30_posterior_24",
    "v3_repay_all_fully_paid_share",
    "repayment_installment_count",
    "repayment_paid_ratio",
    "v2_repay_365d_paid",
    "v2_latest_loan_age_days",
    "v2_existing_future_installments",
    "v2_existing_estimated_remaining_principal",
    "installment_income_ratio",
    "loan_income_ratio",
    "declared_monthly_income",
    "employment_tenure_months",
    "age",
    "dependents",
    "v2_account_3m_income_like_credits_mean",
    "v2_account_3m_outflows_mean",
    "v2_account_3m_minimum_balance_min",
    "has_repayments",
    "has_account",
    "loan_purpose",
    "employment_type",
    "housing_status",
    "region",
]
EXCLUDED = {
    "application_id",
    "customer_id",
    "target",
    "split",
    "decision_date",
    "application_datetime",
    "decision_datetime",
}


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return float(np.max(np.abs(tpr - fpr)))


def metrics(y, p):
    return {"ks": ks(y, p), "auc": float(roc_auc_score(y, p))}


def read_data(feature_set="core"):
    if feature_set == "core":
        columns = CORE_COLUMNS
    else:
        columns = json.loads(
            (
                ROOT / "artifacts/optimization/v5/"
                "confirmation_recipe_v5_tabm_weight1.json"
            ).read_text()
        )["spec"]["columns"]
    assert not set(columns) & EXCLUDED
    full = pd.read_parquet(
        ROOT / "data/processed/application_features_v5.parquet",
        columns=["application_id", "split", "target"] + columns,
    )
    train = full[full.split.eq("train")].reset_index(drop=True)
    test = full[full.split.eq("test")].reset_index(drop=True)
    assert train.application_id.is_unique and test.application_id.is_unique
    return train, test, columns


def make_tables(frame, columns, device="cuda"):
    import sdm

    x = frame[columns].copy()
    stypes = {}
    for c in columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
            stypes[c] = "numerical"
        else:
            # No label encoding fitted to query rows; SDM aligns categories to context.
            x[c] = x[c].astype("string")
            stypes[c] = "categorical"
    return sdm.TableTensor.from_pandas(x, stypes=stypes, device=device)


def context_indices(training_indices, y, max_context, seed):
    """Subsample ONLY this fold's training rows, retaining observed class prevalence."""
    if max_context >= len(training_indices):
        return training_indices.copy()
    chosen, _ = train_test_split(
        training_indices,
        train_size=max_context,
        stratify=np.asarray(y)[training_indices],
        random_state=seed,
    )
    return chosen


def predict_batches(model, x, batch_size):
    import torch

    predictions = []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(x), batch_size):
            output = model.predict(x[start : start + batch_size])
            # Match the positive class by name, never assume a shuffled class index.
            p = output[:, "1"].numerical.reshape(-1).float().cpu().numpy()
            predictions.append(p)
    p = np.concatenate(predictions)
    assert len(p) == len(x) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def run(args):
    import sdm
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    assert torch.cuda.is_available(), "Use the isolated CUDA environment."
    name = f"{args.size}_{args.features}_ctx{args.context}_e{args.estimators}"
    destination = OUT / name
    destination.mkdir(parents=True, exist_ok=True)
    data, test, columns = read_data(args.features)
    y = data.target.astype(int)
    baseline_path = ROOT / "artifacts/optimization/v5/selection_oof_tabm.parquet"
    v5 = None
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
        assert baseline.application_id.equals(data.application_id)
        assert np.array_equal(baseline.target, y)
        v5 = 0.75 * baseline.version4 + 0.25 * baseline.v5_tabm_weight1
    folds = list(StratifiedKFold(3, shuffle=True, random_state=8247).split(data, y))
    model = x = xt = yt = None
    info = {
        "name": name,
        "size": args.size,
        "feature_set": args.features,
        "columns": columns,
        "context_limit": args.context,
        "num_estimators": args.estimators,
        "estimator_batch_size": 1,
        "query_batch_size": args.batch_size,
        "autocast": "float16",
        "weights": "nvidia/Kumo-Tabular",
        "weights_revision": "v1.0.1",
        "weights_commit": "3c3e10bbdb590ace29e7026847f92db3c603096d",
        "split": {"n_splits": 3, "shuffle": True, "random_state": 8247},
        "seed": 2026,
        "finetuned": False,
        "gpu": torch.cuda.get_device_name(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "structured-data-models",
                "torch",
                "pandas",
                "numpy",
                "scikit-learn",
            ]
        },
        "training_data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(
                data[["application_id", "target"] + columns], index=False
            ).values.tobytes()
        ).hexdigest(),
    }
    for fold in args.folds:
        ti, vi = folds[fold]
        if args.pilot:
            vi = vi[: args.pilot]
        oof_path = (
            destination / f"fold{fold}{'_pilot' if args.pilot else ''}_oof.parquet"
        )
        test_path = destination / f"fold{fold}_test.parquet"
        meta_path = oof_path.with_suffix(".json")
        if oof_path.exists() and meta_path.exists() and not args.force:
            cached = json.loads(meta_path.read_text())
            assert cached["recipe"] == info, "Cache recipe changed; use --force."
            if not args.test or test_path.exists():
                print(
                    json.dumps(
                        {
                            "cache": name,
                            "fold": fold,
                            "kumo": cached["kumo"],
                            "v5_same_rows": cached["v5_same_rows"],
                        }
                    ),
                    flush=True,
                )
                continue
        if model is None:
            x = make_tables(data, columns)
            xt = make_tables(test, columns) if args.test else None
            yt = sdm.TableTensor.from_pandas(
                pd.DataFrame({"target": y}),
                stypes={"target": "categorical"},
                device="cuda",
            )
            model = sdm.models.KumoTabular(
                task="classification", size=args.size, device="cuda"
            )
        start = time.time()
        ci = context_indices(ti, y, args.context, 2026 + fold)
        assert len(np.intersect1d(ci, folds[fold][1])) == 0
        generator = torch.Generator(device="cuda").manual_seed(2026 + fold)
        torch.manual_seed(2026 + fold)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            model.fit(
                x[ci],
                yt[ci],
                num_estimators=args.estimators,
                estimator_batch_size=1,
                generator=generator,
            )
        fit_seconds = time.time() - start
        print(
            f"{name} fold={fold}: context={len(ci)}, fit_seconds={fit_seconds:.1f}",
            flush=True,
        )
        p = predict_batches(model, x[vi], args.batch_size)
        # Check predictions are stable when the same rows are queried in a smaller batch.
        small = predict_batches(model, x[vi[:16]], 8)
        batch_error = float(np.max(np.abs(small - p[:16])))
        assert np.allclose(small, p[:16], atol=2e-4, rtol=2e-3), batch_error
        pd.DataFrame(
            {
                "application_id": data.application_id.iloc[vi].to_numpy(),
                "target": y.iloc[vi].to_numpy(),
                "prediction": p,
            }
        ).to_parquet(oof_path, index=False)
        if args.test:
            pt = predict_batches(model, xt, args.batch_size)
            pd.DataFrame(
                {"application_id": test.application_id, "prediction": pt}
            ).to_parquet(test_path, index=False)
        result = {
            "recipe": info,
            "fold": fold,
            "context_rows": len(ci),
            "context_positive_rate": float(y.iloc[ci].mean()),
            "context_ids_sha256": hashlib.sha256(
                "\n".join(data.application_id.iloc[ci].astype(str)).encode()
            ).hexdigest(),
            "validation_rows": len(vi),
            "fit_seconds": fit_seconds,
            "elapsed_seconds": time.time() - start,
            "max_query_batch_difference": batch_error,
            "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "kumo": metrics(y.iloc[vi], p),
            "v5_same_rows": (
                metrics(y.iloc[vi], v5.iloc[vi]) if v5 is not None else None
            ),
        }
        meta_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k != "recipe"}), flush=True
        )
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--features", choices=["core", "v5"], default="core")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--folds", type=int, nargs="+", choices=[0, 1, 2], default=[0, 1, 2]
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Only first N validation rows; no final scores.",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
