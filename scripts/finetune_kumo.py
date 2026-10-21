"""Nested validation for conservative fine-tuning of Kumo's last two ICL layers.

Reference: NVIDIA/structured-data-models examples/tabular/finetune.py.
All labels used for weight updates and checkpoint selection belong to outer train.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from improve_kumo import ROOT, definitions, fingerprint, load_frame
from kumo_tabular import context_indices, make_tables, metrics, predict_batches

DEST = ROOT / "artifacts/optimization/kumo_finetune"
NAME = "v5_last2_ft128_e8"
CHECKPOINT_STEPS = [0, 32, 64, 128]
_MODEL_POOL = None  # Reuse frozen pretrained weights across stages; reset trainable weights per fold.


def id_hash(data, indices):
    return hashlib.sha256(
        "\n".join(data.application_id.iloc[indices]).encode()
    ).hexdigest()


def parameter_snapshot(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def restore_parameters(model, state):
    import torch

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def model_bundle():
    import sdm

    global _MODEL_POOL
    if _MODEL_POOL is None:
        model = sdm.models.KumoTabular(
            task="classification", size="large", device="cuda"
        )
        for p in model.parameters():
            p.requires_grad_(False)
        block = model.models[sdm.Task.classification].icl_block
        for module in [*block.layers[-2:], block.norm, block.head]:
            for p in module.parameters():
                p.requires_grad_(True)
        _MODEL_POOL = (model, parameter_snapshot(model))
        print(
            "Trainable parameters",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            flush=True,
        )
    return _MODEL_POOL


def training_recipe(model):
    import sdm.processing as sp

    recipe = model.default_recipe()
    for module in recipe.features.modules():
        if isinstance(module, sp.Choice):
            module.method = "random"
    recipe.target = sp.StypeDispatch(
        categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")],
        numerical=sp.Standardize(),
    )
    recipe.output = sp.Identity()
    return recipe


def inner_metrics(model, x, yt, y, fitting, validation, fold):
    import torch

    model.eval()
    ci = context_indices(fitting, y, 16384, 9126 + fold)
    assert not len(np.intersect1d(ci, validation))
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        model.fit(
            x[ci],
            yt[ci],
            num_estimators=4,
            estimator_batch_size=4,
            generator=torch.Generator(device="cuda").manual_seed(9126 + fold),
        )
    p = predict_batches(model, x[validation], 8192)
    result = metrics(y.iloc[validation], p)
    model.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fit_inner(model, x, yt, y, outer_training, outer_validation, fold):
    import torch
    import torch.nn.functional as F
    import sdm

    fitting, validation = train_test_split(
        outer_training,
        test_size=0.15,
        stratify=y.iloc[outer_training],
        random_state=43001 + fold,
    )
    assert not len(np.intersect1d(fitting, validation))
    assert not len(
        np.intersect1d(np.concatenate([fitting, validation]), outer_validation)
    )
    torch.manual_seed(6026 + fold)
    start = time.time()
    initial = inner_metrics(model, x, yt, y, fitting, validation, fold)
    history = [dict(step=0, **initial)]
    best_step, best_auc, best_state = 0, initial["auc"], parameter_snapshot(model)
    print(f"fold {fold} inner step 0: {initial}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-3)
    recipe = training_recipe(model)
    rng = np.random.default_rng(62026 + fold)
    losses = []
    for step in range(1, max(CHECKPOINT_STEPS) + 1):
        model.train()
        rows = rng.choice(fitting, 1536, replace=False)
        context, query = rows[:1024], rows[1024:]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                x_context=x[context],
                y_context=yt[context],
                x_query=x[query],
                num_estimators=1,
                recipe=recipe,
                generator=torch.Generator(device="cuda").manual_seed(
                    702600 + fold * 1000 + step
                ),
            )[0]
            logits, target_index = sdm.evaluation.to_class_indices(
                out, yt[query], missing_score=-torch.inf
            )
            loss = F.cross_entropy(logits.float(), target_index)
        assert torch.isfinite(loss), "Nonfinite training loss."
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(norm) and norm > 0, "Invalid or absent gradients."
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 16 == 0:
            print(
                f"fold {fold} step {step}: mean loss {np.mean(losses[-16:]):.6f}; gradient norm {float(norm):.6f}",
                flush=True,
            )
        if step in CHECKPOINT_STEPS:
            score = inner_metrics(model, x, yt, y, fitting, validation, fold)
            history.append(dict(step=step, **score))
            print(f"fold {fold} inner step {step}: {score}", flush=True)
            if score["auc"] > best_auc:
                best_step, best_auc, best_state = (
                    step,
                    score["auc"],
                    parameter_snapshot(model),
                )
    restore_parameters(model, best_state)
    model.eval()
    model.clear()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return best_state, dict(
        selected_step=best_step,
        selected_inner_auc=best_auc,
        checkpoint_selection_metric="Inner ROC-AUC; outer validation never used for selection.",
        history=history,
        mean_training_loss=float(np.mean(losses)),
        fitting_indices=fitting,
        validation_indices=validation,
        training_seconds=time.time() - start,
    )


def frozen_sources(stage, fold):
    directory = (
        ROOT / "artifacts/optimization/kumo/large_v5_ctx50000_e8"
        if stage == "cv3_seed8247"
        else ROOT / "artifacts/optimization/kumo_improvement" / stage / "baseline_e8"
    )
    splits = int(stage.split("_")[0][2:])
    if not all(
        (directory / f"fold{i}_oof.parquet").exists()
        and (directory / f"fold{i}_test.parquet").exists()
        for i in range(splits)
    ):
        directory = DEST / stage / "baseline_e8"
    return directory / f"fold{fold}_oof.parquet", directory / f"fold{fold}_test.parquet"


def ensure_frozen(args):
    """Fresh clones can reproduce the frozen comparison without historical OOF files."""
    import torch, sdm

    stage = f"cv{args.splits}_seed{args.fold_seed}"
    if all(
        all(f.exists() for f in frozen_sources(stage, i)) for i in range(args.splits)
    ):
        return
    spec = next(s for s in definitions() if s["name"] == "baseline_e8")
    data, test = load_frame([spec])
    y = data.target.astype(int)
    path = DEST / stage / "baseline_e8"
    path.mkdir(parents=True, exist_ok=True)
    recipe = dict(
        spec=spec,
        n_splits=args.splits,
        fold_seed=args.fold_seed,
        estimators=8,
        estimator_batch_size=4,
        query_batch_size=8192,
        autocast="float16",
        training_sha256=fingerprint(
            data, ["application_id", "target"] + spec["columns"]
        ),
        test_sha256=fingerprint(test, ["application_id"] + spec["columns"]),
    )
    folds = list(
        StratifiedKFold(args.splits, shuffle=True, random_state=args.fold_seed).split(
            data, y
        )
    )
    x = xt = yt = model = base_state = None
    for fold, (ti, vi) in enumerate(folds):
        meta_file = path / f"fold{fold}_metadata.json"
        oof_file = path / f"fold{fold}_oof.parquet"
        test_file = path / f"fold{fold}_test.parquet"
        if meta_file.exists() and oof_file.exists() and test_file.exists():
            assert json.loads(meta_file.read_text())["recipe"] == recipe
            continue
        if x is None:
            x = make_tables(data, spec["columns"])
            xt = make_tables(test, spec["columns"])
            yt = sdm.TableTensor.from_pandas(
                pd.DataFrame({"target": y}),
                stypes={"target": "categorical"},
                device="cuda",
            )
            model, base_state = model_bundle()
        restore_parameters(model, base_state)
        model.eval()
        model.clear()
        ci = context_indices(ti, y, 50000, 2026 + fold)
        assert not len(np.intersect1d(ci, vi))
        torch.manual_seed(2026 + fold)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            model.fit(
                x[ci],
                yt[ci],
                num_estimators=8,
                estimator_batch_size=4,
                generator=torch.Generator(device="cuda").manual_seed(2026 + fold),
            )
        p = predict_batches(model, x[vi], 8192)
        pt = predict_batches(model, xt, 8192)
        pd.DataFrame(
            dict(
                application_id=data.application_id.iloc[vi].to_numpy(),
                target=y.iloc[vi].to_numpy(),
                prediction=p,
            )
        ).to_parquet(oof_file, index=False)
        pd.DataFrame(
            dict(application_id=test.application_id, prediction=pt)
        ).to_parquet(test_file, index=False)
        meta = dict(
            recipe=recipe,
            fold=fold,
            metrics=metrics(y.iloc[vi], p),
            training_ids_sha256=id_hash(data, ti),
            validation_ids_sha256=id_hash(data, vi),
            context_ids_sha256=id_hash(data, ci),
        )
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        print("FROZEN", stage, fold, meta["metrics"], flush=True)
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args):
    import torch, sdm

    global _MODEL_POOL

    torch.set_num_threads(8)
    spec = next(s for s in definitions() if s["name"] == "baseline_e8")
    data, test = load_frame([spec])
    y = data.target.astype(int)
    stage = f"cv{args.splits}_seed{args.fold_seed}"
    path = DEST / stage / NAME
    path.mkdir(parents=True, exist_ok=True)
    recipe = dict(
        name=NAME,
        columns=spec["columns"],
        n_splits=args.splits,
        fold_seed=args.fold_seed,
        checkpoint_steps=CHECKPOINT_STEPS,
        learning_rate=1e-5,
        weight_decay=1e-3,
        trainable="Last two ICL blocks, ICL norm and classification head only.",
        training_context=1024,
        training_query=512,
        training_autocast="bfloat16",
        augmentation="Random choice of all four default numerical transformations; column/category permutations.",
        inner_validation_fraction=0.15,
        inner_context_limit=16384,
        inner_estimators=4,
        outer_context_limit=50000,
        outer_estimators=8,
        estimator_batch_size=4,
        query_batch_size=8192,
        inference_autocast="float16",
        weights="nvidia/Kumo-Tabular",
        weights_revision="v1.0.1",
        weights_commit="3c3e10bbdb590ace29e7026847f92db3c603096d",
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
    definition = path / "definition.json"
    if definition.exists() and any(path.glob("fold*_oof.parquet")):
        assert (
            json.loads(definition.read_text()) == recipe
        ), "Different cached recipe; choose a new name."
    definition.write_text(json.dumps(recipe, indent=2) + "\n")
    folds = list(
        StratifiedKFold(args.splits, shuffle=True, random_state=args.fold_seed).split(
            data, y
        )
    )
    x = xt = yt = None
    model, base_state = (None, None) if _MODEL_POOL is None else _MODEL_POOL
    for fold in args.folds if args.folds is not None else range(args.splits):
        ti, vi = folds[fold]
        meta_file = path / f"fold{fold}_metadata.json"
        oof_file = path / f"fold{fold}_oof.parquet"
        test_file = path / f"fold{fold}_test.parquet"
        if (
            meta_file.exists()
            and oof_file.exists()
            and (not args.test or test_file.exists())
            and not args.force
        ):
            meta = json.loads(meta_file.read_text())
            assert meta["recipe"] == recipe
            print(f"cached {stage} fold {fold}: {meta['metrics']}", flush=True)
            continue
        if x is None:
            x = make_tables(data, spec["columns"])
            xt = make_tables(test, spec["columns"]) if args.test else None
            yt = sdm.TableTensor.from_pandas(
                pd.DataFrame({"target": y}),
                stypes={"target": "categorical"},
                device="cuda",
            )
        if model is None:
            model, base_state = model_bundle()
        restore_parameters(model, base_state)
        model.clear()
        torch.cuda.reset_peak_memory_stats()
        state, inner = fit_inner(model, x, yt, y, ti, vi, fold)
        fitting, validation = inner.pop("fitting_indices"), inner.pop(
            "validation_indices"
        )
        checkpoint = path / f"fold{fold}_trainable_weights.pt"
        torch.save(
            dict(weights=state, recipe=recipe, selected_step=inner["selected_step"]),
            checkpoint,
        )
        ci = context_indices(ti, y, 50000, 2026 + fold)
        assert not len(np.intersect1d(ci, vi))
        source_oof, source_test = frozen_sources(stage, fold)
        reused_frozen = (
            inner["selected_step"] == 0
            and source_oof.exists()
            and (not args.test or source_test.exists())
        )
        if reused_frozen:
            outer = pd.read_parquet(source_oof)
            assert np.array_equal(outer.application_id, data.application_id.iloc[vi])
            p = outer.prediction.to_numpy()
            outer.to_parquet(oof_file, index=False)
            if args.test:
                pd.read_parquet(source_test).to_parquet(test_file, index=False)
        else:
            torch.manual_seed(2026 + fold)
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.float16
            ):
                model.fit(
                    x[ci],
                    yt[ci],
                    num_estimators=8,
                    estimator_batch_size=4,
                    generator=torch.Generator(device="cuda").manual_seed(2026 + fold),
                )
            p = predict_batches(model, x[vi], 8192)
            pd.DataFrame(
                dict(
                    application_id=data.application_id.iloc[vi].to_numpy(),
                    target=y.iloc[vi].to_numpy(),
                    prediction=p,
                )
            ).to_parquet(oof_file, index=False)
            if args.test:
                pt = predict_batches(model, xt, 8192)
                pd.DataFrame(
                    dict(application_id=test.application_id, prediction=pt)
                ).to_parquet(test_file, index=False)
        meta = dict(
            recipe=recipe,
            fold=fold,
            metrics=metrics(y.iloc[vi], p),
            inner=inner,
            training_ids_sha256=id_hash(data, ti),
            validation_ids_sha256=id_hash(data, vi),
            context_ids_sha256=id_hash(data, ci),
            inner_fitting_ids_sha256=id_hash(data, fitting),
            inner_validation_ids_sha256=id_hash(data, validation),
            outer_validation_used_for_training=False,
            trainable_parameter_names=list(state),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            reused_frozen_predictions=reused_frozen,
            peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        meta_file.write_text(json.dumps(meta, indent=2) + "\n")
        print(
            "COMPLETED",
            stage,
            fold,
            json.dumps(
                dict(
                    metrics=meta["metrics"],
                    selected_step=inner["selected_step"],
                    peak_gpu_gb=meta["peak_gpu_gb"],
                )
            ),
            flush=True,
        )
        model.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--fold-seed", type=int, default=8247)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--force", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
