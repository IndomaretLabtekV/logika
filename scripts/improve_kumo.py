"""Controlled Kumo experiments; preserve the public-0.56262 checkpoint.

Screen on the original three folds. Confirm a locked recipe on new folds before
exporting a candidate. All inference stays local; no Kaggle submission is sent.
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
from sklearn.model_selection import StratifiedKFold

from kumo_tabular import (
    CORE_COLUMNS,
    EXCLUDED,
    ROOT,
    make_tables,
    context_indices,
    metrics,
    ks,
    predict_batches,
)

DEST = ROOT / "artifacts/optimization/kumo_improvement"
LATENT_COLUMNS = [
    "re_price_log_gls",
    "re_price_log_posterior",
    "re_price_posterior_sd",
    "re_bureau_score_gls",
    "re_bureau_score_posterior",
    "re_bureau_posterior_sd",
    "re_income_geometric_mean",
    "re_income_log_sd",
    "re_income_log_measurement_sd",
    "re_declared_income_log_gap",
    "re_outflow_income_geometric_ratio",
    "re_late_1_posterior",
    "re_late_1_posterior_sd",
    "re_late_7_posterior",
    "re_late_7_posterior_sd",
    "re_late_30_posterior",
    "re_late_30_posterior_sd",
]
DERIVED_COLUMNS = [
    "ki_installment_income_geo_ratio",
    "ki_loan_income_geo_ratio",
    "ki_log_installment",
    "ki_log_declared_income",
    "ki_log_loan_amount",
]


def definitions():
    old = json.loads(
        (
            ROOT / "artifacts/optimization/v5/confirmation_recipe_v5_tabm_weight1.json"
        ).read_text()
    )["spec"]["columns"]
    dense = old + [
        "re_price_log_posterior",
        "re_price_posterior_sd",
        "re_bureau_score_posterior",
        "re_bureau_posterior_sd",
        "re_income_geometric_mean",
        "re_declared_income_log_gap",
        "re_late_1_posterior",
        "re_late_1_posterior_sd",
        "re_late_30_posterior",
        "ki_installment_income_geo_ratio",
    ]
    latent = list(
        dict.fromkeys(CORE_COLUMNS + LATENT_COLUMNS + DERIVED_COLUMNS + ["term_months"])
    )
    specs = [
        dict(name="core37_e8", columns=CORE_COLUMNS, estimators=8),
        dict(name="denoised_e8", columns=latent, estimators=8),
        dict(name="v5_latent100_e8", columns=dense, estimators=8),
        dict(name="v5_e16", columns=old, estimators=16),
        dict(name="baseline_e8", columns=old, estimators=8),
        dict(name="v5_fullcontext_e8", columns=old, estimators=8),
        dict(name="tabicl_v5_e8", columns=old, estimators=8, family="tabicl"),
        dict(name="tabicl_denoised_e8", columns=latent, estimators=8, family="tabicl"),
        dict(name="v5_balanced2_e8", columns=old, estimators=8, negative_ratio=2),
        dict(name="v5_balanced1_e8", columns=old, estimators=8, negative_ratio=1),
    ]
    for s in specs:
        s.update(size="large", context_limit=100000, recipe="default", seed_offset=0)
        if s.get("family") == "tabicl":
            s["size"] = "v2"
        if s["name"] == "baseline_e8":
            s["context_limit"] = 50000
        assert (
            len(s["columns"]) == len(set(s["columns"]))
            and not set(s["columns"]) & EXCLUDED
        )
    return specs


def load_frame(specs):
    columns = list(dict.fromkeys(c for s in specs for c in s["columns"]))
    required = [c for c in columns if c not in DERIVED_COLUMNS]
    uses_derived = any(c in DERIVED_COLUMNS for c in columns)
    required += [
        c
        for c in [
            "proposed_monthly_installment",
            "loan_amount",
            "declared_monthly_income",
            "re_income_geometric_mean",
        ]
        if uses_derived and c not in required
    ]
    feature_file = (
        "application_features_reverse.parquet"
        if uses_derived or any(c.startswith("re_") for c in required)
        else "application_features_v5.parquet"
    )
    full = pd.read_parquet(
        ROOT / "data/processed" / feature_file,
        columns=["application_id", "customer_id", "split", "target"] + required,
    )
    if uses_derived:
        income = full.re_income_geometric_mean.where(
            full.re_income_geometric_mean.gt(0)
        )
        full["ki_installment_income_geo_ratio"] = (
            full.proposed_monthly_installment / income
        )
        full["ki_loan_income_geo_ratio"] = full.loan_amount / income
        for new, old in [
            ("ki_log_installment", "proposed_monthly_installment"),
            ("ki_log_declared_income", "declared_monthly_income"),
            ("ki_log_loan_amount", "loan_amount"),
        ]:
            full[new] = np.log1p(full[old].where(full[old].ge(0)))
    train = full[full.split.eq("train")].reset_index(drop=True)
    test = full[full.split.eq("test")].reset_index(drop=True)
    assert train.application_id.is_unique and test.application_id.is_unique
    assert set(train.customer_id).isdisjoint(test.customer_id)
    return train, test


def fingerprint(frame, columns):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()
    ).hexdigest()


def fit_context(model, x, yt, indices, y, spec, seed, estimator_batch_size):
    """Optional class-stratified contexts, all drawn from outer training only."""
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    sampling = None
    if "negative_ratio" in spec:
        positive = indices[np.asarray(y)[indices] == 1]
        negative = indices[np.asarray(y)[indices] == 0]
        n_negative = min(len(negative), int(len(positive) * spec["negative_ratio"]))
        rng = np.random.default_rng(seed)
        members = [
            rng.permutation(
                np.concatenate(
                    [positive, rng.choice(negative, n_negative, replace=False)]
                )
            )
            for _ in range(spec["estimators"])
        ]
        assert all(np.isin(m, indices).all() for m in members)
        xc = torch.stack([x[m] for m in members])
        yc = torch.stack([yt[m] for m in members])
        num_estimators = None  # Leading dimension supplies independent member contexts.
        sampling = dict(
            rows_per_member=len(members[0]),
            num_members=len(members),
            population_positive_rate=float(np.asarray(y)[indices].mean()),
            context_positive_rate=float(len(positive) / len(members[0])),
            member_index_sha256=[
                hashlib.sha256(m.astype(np.int64).tobytes()).hexdigest()
                for m in members
            ],
            probability_correction="Population-odds / sampled-context-odds",
        )
    else:
        xc, yc, num_estimators = x[indices], yt[indices], spec["estimators"]
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        model.fit(
            xc,
            yc,
            num_estimators=num_estimators,
            estimator_batch_size=estimator_batch_size,
            generator=generator,
        )
    return sampling


def restore_population_prior(prediction, sampling):
    if sampling is None:
        return prediction
    prior, sampled = (
        sampling["population_positive_rate"],
        sampling["context_positive_rate"],
    )
    factor = (prior / (1 - prior)) / (sampled / (1 - sampled))
    p = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1 - 1e-7)
    return factor * p / (1 - p + factor * p)


def predict_context_batches(model, x, batch_size, sampling):
    """Broadcast each query to the independently fitted member contexts."""
    if sampling is None:
        return predict_batches(model, x, batch_size)
    import torch

    parts = []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(x), batch_size):
            # Materialize the leading member dimension. Sharing a singleton
            # group loses position-dependent fitted preprocessing state.
            query = torch.stack(
                [x[start : start + batch_size]] * sampling["num_members"]
            )
            output = model.predict(query)
            parts.append(output[:, "1"].numerical.reshape(-1).float().cpu().numpy())
    p = np.concatenate(parts)
    assert len(p) == len(x) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def evaluate(args):
    import sdm
    import torch

    torch.set_num_threads(8)
    specs = [s for s in definitions() if s["name"] in args.candidates]
    assert len(specs) == len(args.candidates)
    DEST.mkdir(parents=True, exist_ok=True)
    data, test = load_frame(specs)
    y = data.target.astype(int)
    assert torch.cuda.is_available()
    folds = list(
        StratifiedKFold(args.splits, shuffle=True, random_state=args.fold_seed).split(
            data, y
        )
    )
    stage = f"cv{args.splits}_seed{args.fold_seed}"
    model = None
    active_family = None
    for spec in specs:
        path = DEST / stage / spec["name"]
        path.mkdir(parents=True, exist_ok=True)
        recipe = dict(
            spec=spec,
            n_splits=args.splits,
            fold_seed=args.fold_seed,
            query_batch_size=args.batch_size,
            estimator_batch_size=args.estimator_batch_size,
            autocast="float16",
            weights=(
                "jingang/TabICL"
                if spec.get("family") == "tabicl"
                else "nvidia/Kumo-Tabular"
            ),
            weights_revision=(
                "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
                if spec.get("family") == "tabicl"
                else "v1.0.1"
            ),
            weights_commit=(
                "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
                if spec.get("family") == "tabicl"
                else "3c3e10bbdb590ace29e7026847f92db3c603096d"
            ),
            training_sha256=fingerprint(
                data, ["application_id", "target"] + spec["columns"]
            ),
            test_sha256=fingerprint(test, ["application_id"] + spec["columns"]),
            packages={
                p: importlib.metadata.version(p)
                for p in [
                    "torch",
                    "structured-data-models",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                ]
            },
        )
        definition_file = path / "definition.json"
        if definition_file.exists() and not args.force:
            same_definition = json.loads(definition_file.read_text()) == recipe
            if any(path.glob("fold*_oof.parquet")):
                assert same_definition, "Existing cached experiment definition changed."
            elif not same_definition:
                print(
                    "Historical metadata differs; recomputing without cached predictions.",
                    flush=True,
                )
        definition_file.write_text(json.dumps(recipe, indent=2) + "\n")
        x = xt = yt = None
        for fold in args.folds if args.folds is not None else range(args.splits):
            ti, vi = folds[fold]
            oof_file = path / f"fold{fold}_oof.parquet"
            test_file = path / f"fold{fold}_test.parquet"
            meta_file = path / f"fold{fold}_metadata.json"
            if oof_file.exists() and meta_file.exists() and not args.force:
                meta = json.loads(meta_file.read_text())
                assert (
                    meta["recipe"] == recipe
                ), "Different cached recipe; use a new candidate name."
                if not args.test or test_file.exists():
                    print(
                        f"cached {stage} {spec['name']} fold {fold}: {meta['metrics']}",
                        flush=True,
                    )
                    continue
            if x is None:
                print(
                    f"Preparing {stage} {spec['name']}: {len(spec['columns'])} features",
                    flush=True,
                )
                x = make_tables(data, spec["columns"])
                xt = make_tables(test, spec["columns"]) if args.test else None
                yt = sdm.TableTensor.from_pandas(
                    pd.DataFrame({"target": y}),
                    stypes={"target": "categorical"},
                    device="cuda",
                )
            family = spec.get("family", "kumo")
            if model is None or active_family != family:
                if model is not None:
                    model.clear()
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                if family == "tabicl":
                    from huggingface_hub import hf_hub_download

                    checkpoint = hf_hub_download(
                        "jingang/TabICL",
                        "tabicl-classifier-v2-20260212.ckpt",
                        revision=recipe["weights_commit"],
                    )
                    assert Path(checkpoint).parent.name == recipe["weights_commit"]
                    # Same remapping as the official loader, with an immutable revision.
                    from sdm.models.tabiclv2.ckpt import remap_ckpt

                    model = sdm.models.TabICLv2(
                        task="classification", pretrained=False, device="cuda"
                    )
                    state = torch.load(
                        checkpoint, map_location="cuda", weights_only=True
                    )["state_dict"]
                    model.models[sdm.Task.classification].load_state_dict(
                        remap_ckpt(state, is_classifier=True), assign=True
                    )
                    del state
                else:
                    model = sdm.models.KumoTabular(
                        task="classification", size="large", device="cuda"
                    )
                active_family = family
            start = time.time()
            ci = context_indices(
                ti, y, spec["context_limit"], 2026 + fold + spec["seed_offset"]
            )
            assert not len(np.intersect1d(ci, vi))
            torch.manual_seed(2026 + fold + spec["seed_offset"])
            torch.cuda.reset_peak_memory_stats()
            sampling = fit_context(
                model,
                x,
                yt,
                ci,
                y,
                spec,
                2026 + fold + spec["seed_offset"],
                args.estimator_batch_size,
            )
            fit_seconds = time.time() - start
            print(
                f"Fitted {stage} {spec['name']} fold {fold} in {fit_seconds:.1f}s; predicting validation/test.",
                flush=True,
            )
            p = restore_population_prior(
                predict_context_batches(model, x[vi], args.batch_size, sampling),
                sampling,
            )
            pd.DataFrame(
                {
                    "application_id": data.application_id.iloc[vi].to_numpy(),
                    "target": y.iloc[vi].to_numpy(),
                    "prediction": p,
                }
            ).to_parquet(oof_file, index=False)
            if args.test:
                pt = restore_population_prior(
                    predict_context_batches(model, xt, args.batch_size, sampling),
                    sampling,
                )
                pd.DataFrame(
                    {"application_id": test.application_id, "prediction": pt}
                ).to_parquet(test_file, index=False)
            meta = dict(
                recipe=recipe,
                fold=fold,
                training_rows=len(ti),
                context_pool_rows=len(ci),
                context_rows=(sampling["rows_per_member"] if sampling else len(ci)),
                context_sampling=sampling,
                context_ids_sha256=hashlib.sha256(
                    "\n".join(data.application_id.iloc[ci]).encode()
                ).hexdigest(),
                validation_rows=len(vi),
                training_ids_sha256=hashlib.sha256(
                    "\n".join(data.application_id.iloc[ti]).encode()
                ).hexdigest(),
                validation_ids_sha256=hashlib.sha256(
                    "\n".join(data.application_id.iloc[vi]).encode()
                ).hexdigest(),
                fit_seconds=fit_seconds,
                elapsed_seconds=time.time() - start,
                metrics=metrics(y.iloc[vi], p),
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
            meta_file.write_text(json.dumps(meta, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in meta.items() if k != "recipe"}), flush=True
            )
            model.clear()
            gc.collect()
            torch.cuda.empty_cache()
        del x, xt, yt
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=[s["name"] for s in definitions()],
        required=True,
    )
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=8247)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--estimator-batch-size", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--force", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
