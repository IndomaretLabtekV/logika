"""Strict local Kumo inference, with explicit context and post-recipe column views.

``predict`` returns ``(positive_probabilities, metadata)``. Row arguments are
zero-based positions, not DataFrame index labels; labels are indexed only at
fitting positions. ``column_cap=None`` removes the default recipe's internal
SelectColumns step, not its fitted constant-column removal or category-count
expansion. This is API support, NOT evidence that a full-width fit fits in VRAM.

``prepare_inputs`` is CPU-only. ``technical_preflight`` executes a real cached
fit with the requested full context/view and a small prefix of query rows; it
never scores predictions or reads query labels. No automatic OOM retry, context
reduction, feature truncation, download, or checkpoint/cache write is performed.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "nvidia/Kumo-Tabular"
REVISION = "v1.0.1"
COMMIT = "3c3e10bbdb590ace29e7026847f92db3c603096d"
LARGE_SHA256 = "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613"


class KumoInferenceError(RuntimeError):
    """An inference failure carrying provenance and observed memory, not scores."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_identity(size="large"):
    """Verify local, content-addressed bytes and the previously audited loader.

    Large additionally matches the historical audited byte hash. Small/medium
    must be cached at the same pinned commit with a SHA256-named HF blob; their
    actual byte hashes are reported, not claimed to match the large audit.
    This routine does not import torch/sdm or initialize CUDA.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    if size not in {"small", "medium", "large"}:
        raise ValueError("size must be small, medium, or large")
    audit = json.loads((ROOT / "eda/out/168_kumo_checkpoint_audit.json").read_text())
    if (audit["repository"], audit["revision"], audit["snapshot_commit"],
            audit["checkpoint_sha256"]) != (REPOSITORY, REVISION, COMMIT, LARGE_SHA256):
        raise ValueError("Historical Kumo checkpoint audit differs from pinned identity")
    if importlib.metadata.version("structured-data-models") != audit["sdm_version"]:
        raise ValueError("Installed sdm version differs from checkpoint audit")
    package = Path(importlib.util.find_spec("sdm").origin).parent
    for relative, key in (("models/kumo/tabular/model.py", "model_loader_sha256"),
                          ("models/_huggingface.py", "downloader_sha256")):
        if _file_hash(package / relative) != audit[key]:
            raise ValueError(f"Installed audited loader changed: {relative}")
    cache = Path(HF_HUB_CACHE) / "models--nvidia--Kumo-Tabular"
    path = cache / "snapshots" / COMMIT / size / "classifier.pt"
    blob = path.resolve(strict=True)
    digest = _file_hash(blob)
    if size == "large":
        if digest != LARGE_SHA256:
            raise ValueError("Large checkpoint differs from pinned audited bytes")
    elif len(blob.name) != 64 or digest != blob.name or not blob.is_relative_to(cache / "blobs"):
        raise ValueError("Small/medium checkpoint must match its cached SHA256-addressed blob")
    config = cache / "snapshots" / COMMIT / "config.json"
    if _file_hash(config) != audit["config_sha256"]:
        raise ValueError("Pinned checkpoint configuration changed")
    return {
        "repository": REPOSITORY, "revision": REVISION, "snapshot_commit": COMMIT,
        "size": size, "checkpoint_path": str(path), "checkpoint_sha256": digest,
        "checkpoint_bytes": blob.stat().st_size, "sdm_version": audit["sdm_version"],
        "config_sha256": audit["config_sha256"],
        "model_loader_sha256": audit["model_loader_sha256"],
        "downloader_sha256": audit["downloader_sha256"],
        "historical_large_bytes_matched": size == "large", "network_used": False,
        "load_method": "local torch.load; public load_state_dict(assign=True)",
    }


def _rows(rows, length, name):
    values = np.asarray(rows)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional integer position array")
    if (values < 0).any() or (values >= length).any():
        raise ValueError(f"{name} contains out-of-range positions")
    values = values.astype(np.int64, copy=False)
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} contains duplicate positions")
    return values


def _fit_labels(labels, rows):
    if isinstance(labels, pd.Series):
        values = labels.iloc[rows].to_numpy()
    elif isinstance(labels, np.ndarray):
        values = labels[rows]
    else:
        values = np.asarray([labels[int(row)] for row in rows])
    if values.ndim != 1 or pd.isna(values).any() or not np.isin(values, [0, 1]).all():
        raise ValueError("Fitting labels must be nonmissing binary 0/1 values")
    values = values.astype(np.int8)
    if set(values.tolist()) != {0, 1}:
        raise ValueError("Fitting context must contain both binary classes")
    return values


def prepare_inputs(frame, fit_rows, query_rows, labels, columns, model_seed,
                   max_context=None):
    """Prepare only real context/query rows; fit vocabulary on actual context.

    With max_context set, take a reproducible class-proportional sample without
    replacement. Both classes are retained and the selected rows are reported.
    Only the supplied fitting labels are accessed, including for sampling.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("frame must be a pandas DataFrame with unique column names")
    if isinstance(columns, str):
        raise ValueError("columns must be a sequence, not a single string")
    columns = list(columns)
    if not columns or len(set(columns)) != len(columns) or not all(isinstance(c, str) for c in columns):
        raise ValueError("columns must be nonempty unique string names")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Requested columns are absent: {sorted(missing)}")
    if not isinstance(model_seed, (int, np.integer)) or isinstance(model_seed, (bool, np.bool_)) or not 0 <= model_seed < 2**63:
        raise ValueError("model_seed must be an explicit integer in [0, 2**63)")
    fit = _rows(fit_rows, len(frame), "fit_rows")
    query = _rows(query_rows, len(frame), "query_rows")
    if np.intersect1d(fit, query).size:
        raise ValueError("Fit and query positions must be disjoint")
    if len(labels) != len(frame):
        raise ValueError("labels must align positionally with frame")
    y = _fit_labels(labels, fit)
    original_fit_count = len(fit)
    if max_context is not None:
        if not isinstance(max_context, (int, np.integer)) or isinstance(max_context, (bool, np.bool_)) or max_context < 2:
            raise ValueError("max_context must be None or an integer >= 2")
        if max_context < len(fit):
            rng = np.random.default_rng(int(model_seed))
            classes = [np.flatnonzero(y == label) for label in (0, 1)]
            n1 = int(round(max_context * len(classes[1]) / len(fit)))
            n1 = min(max(n1, 1, max_context - len(classes[0])), len(classes[1]), max_context - 1)
            selected = np.sort(np.concatenate([
                rng.choice(classes[0], max_context - n1, replace=False),
                rng.choice(classes[1], n1, replace=False),
            ]))
            fit, y = fit[selected], y[selected]
    positions = frame.columns.get_indexer(columns)
    context = frame.iloc[fit, positions].copy().reset_index(drop=True)
    queries = frame.iloc[query, positions].copy().reset_index(drop=True)
    stypes, category_sizes, unknown_counts = {}, {}, {}
    for column in columns:
        dtype = context[column].dtype
        categorical = (isinstance(dtype, pd.CategoricalDtype)
                       or pd.api.types.is_object_dtype(dtype)
                       or pd.api.types.is_string_dtype(dtype))
        if categorical:
            # Do not reuse a category dtype's global vocabulary, even if supplied.
            vocabulary = pd.Index(pd.unique(context[column].dropna().astype(object)))
            unknown_counts[column] = int((queries[column].notna() & ~queries[column].isin(vocabulary)).sum())
            context[column] = pd.Categorical(context[column], categories=vocabulary)
            queries[column] = pd.Categorical(queries[column], categories=vocabulary)
            stypes[column] = "categorical"
            category_sizes[column] = len(vocabulary)
        elif pd.api.types.is_numeric_dtype(dtype):
            stypes[column] = "numerical"
        else:
            raise TypeError(f"Unsupported feature dtype for {column}: {dtype}")
    metadata = {
        "fitting_partition_rows": original_fit_count, "context_count": len(fit),
        "context_rows": fit.tolist(), "context_rows_sha256": hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
        "context_labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "context_class_counts": {str(c): int((y == c).sum()) for c in (0, 1)},
        "query_count": len(query), "query_rows_sha256": hashlib.sha256(query.astype("<i8").tobytes()).hexdigest(),
        "columns": columns, "input_column_count": len(columns),
        "model_seed": int(model_seed), "context_seed": int(model_seed),
        "max_context": None if max_context is None else int(max_context),
        "context_reduced": len(fit) != original_fit_count,
        "context_sampling": "none" if len(fit) == original_fit_count else "class-proportional without replacement, both classes retained",
        "category_policy": "actual-context-only vocabulary; query-only categories become missing",
        "category_vocabulary_sizes": category_sizes, "query_unknown_category_counts": unknown_counts,
        "query_labels_accessed": False, "query_labels_passed_to_model": False,
    }
    return {"context": context, "queries": queries,
            "targets": pd.DataFrame({"target": y.astype(str)}),
            "stypes": stypes, "metadata": metadata}


def _recipe(column_cap):
    """Customize the public recipe, never a model instance or package source."""
    import sdm.processing as sp
    from sdm.models.kumo.tabular.recipe import default_recipe

    if column_cap is not None and (not isinstance(column_cap, (int, np.integer))
                                  or isinstance(column_cap, (bool, np.bool_)) or column_cap <= 0):
        raise ValueError("column_cap must be None (uncapped) or a positive integer")
    recipe = default_recipe()
    if not isinstance(recipe.features, sp.Sequential):
        raise NotImplementedError("Installed default recipe does not expose a supported Sequential feature pipeline")
    steps = list(recipe.features)
    selectors = [step for step in steps if isinstance(step, sp.SelectColumns)]
    if len(selectors) != 1 or selectors[0].max_columns != 500 or selectors[0].method != "first":
        raise NotImplementedError("Installed recipe differs from the audited SelectColumns(500, method='first') contract")
    if sum(isinstance(module, sp.SelectColumns) for module in recipe.features.modules()) != 1:
        raise NotImplementedError("Nested SelectColumns is unsupported; cannot guarantee requested view")
    recipe.features = [step if not isinstance(step, sp.SelectColumns)
                       else sp.SelectColumns(int(column_cap), method="first")
                       for step in steps if not (isinstance(step, sp.SelectColumns) and column_cap is None)]
    return recipe


def _load_model(sdm, torch, identity, device):
    # No Hub loader is invoked: this cannot download or follow a moving ref.
    model = sdm.models.KumoTabular(task="classification", size=identity["size"],
                                  pretrained=False, device="meta")
    state = torch.load(identity["checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({f"models.classification.{key}": value for key, value in state.items()},
                          strict=True, assign=True)
    del state
    return model.to(device).eval()


def predict(frame, fit_rows, query_rows, labels, columns, model_seed,
            n_estimators=8, size="large", max_context=None, column_cap=500,
            chunk=128, memory_fraction=.70):
    """Fit once, predict chunks with the identical cached estimator draw, release.

    The cap is after category-count expansion, constant dropping and Latin
    column shuffle; it is per semantic block (all numerical at this stage).
    Uncapped removes only SelectColumns. Query chunking does NOT chunk the full
    context row-embedding allocation. Failures raise KumoInferenceError with
    metadata and are never retried using a different statistical configuration.
    """
    for name, value in (("n_estimators", n_estimators), ("chunk", chunk)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must lie in (0, 1]")
    prepared = prepare_inputs(frame, fit_rows, query_rows, labels, columns, model_seed, max_context)
    identity = checkpoint_identity(size)
    recipe = _recipe(column_cap)
    import torch
    import sdm
    from sdm.models.callback import Callback

    metadata = prepared["metadata"]
    metadata.update({
        "checkpoint": identity, "n_estimators": int(n_estimators),
        "estimator_batch_size": 1, "installed_default_estimator_batch_size": 1,
        "column_cap": None if column_cap is None else int(column_cap),
        "column_cap_semantics": "post-preprocessing, post-Latin-shuffle per semantic block; all features numerical here",
        "view": "uncapped recipe" if column_cap is None else f"internal cap {column_cap}",
        "uncapped_recipe_supported": True, "full_view_memory_feasibility": "not established by API support",
        "chunk": int(chunk), "memory_fraction": float(memory_fraction),
        "dtype": "CUDA float16 autocast", "device": "cuda:0",
        "torch_version": torch.__version__, "processed_contexts": [],
        "fit_buffer_note": "Full context R*(processed_C+4)*cell_channels buffer is allocated before internal batching; smaller query chunks cannot remove it",
        "cell_channels": 128 if size == "small" else 256,
        "max_allocated_bytes": None, "max_reserved_bytes": None, "status": "preparing",
    })

    class RecordContext(Callback):
        def on_context_preprocessing_end(self, model, x, y, related_tables):
            width = int(x.numerical.size(-1))
            metadata["processed_contexts"].append({
                "rows": int(x.numerical.size(-2)), "columns": width,
                "column_names": [name for names in x.columns.values() for name in names],
                "row_embedding_buffer_bytes": int(x.numerical.size(-2)) * (width + 4) * metadata["cell_channels"] * 2,
            })
            return x, y, related_tables

    model = xc = yc = xq = prediction = generator = None
    device = torch.device("cuda:0")
    initialized = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU preparation is available through prepare_inputs")
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device)
        initialized = True
        torch.cuda.reset_peak_memory_stats(device)
        model = _load_model(sdm, torch, identity, device)
        stypes = sdm.infer_stypes(prepared["context"], overrides=prepared["stypes"])
        xc = sdm.TableTensor.from_pandas(df=prepared["context"], stypes=stypes, device=device)
        yc = sdm.TableTensor.from_pandas(df=prepared["targets"],
                                       stypes=sdm.infer_stypes(prepared["targets"], overrides={"target": "categorical"}),
                                       device=device)
        generator = torch.Generator(device=device).manual_seed(int(model_seed))
        probabilities = np.empty(len(prepared["queries"]), dtype=np.float64)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            metadata["status"] = "fitting"
            model.fit(xc, yc, recipe=recipe, num_estimators=int(n_estimators),
                      estimator_batch_size=1, generator=generator, callbacks=[RecordContext()])
            xc = yc = None
            metadata["status"] = "predicting"
            for start in range(0, len(probabilities), int(chunk)):
                xq = sdm.TableTensor.from_pandas(df=prepared["queries"].iloc[start:start + int(chunk)],
                                               stypes=stypes, device=device)
                prediction = model.predict(xq)
                output = prediction.to_pandas()
                positive = [column for column in output.columns if str(column) == "1"]
                if len(positive) != 1:
                    raise RuntimeError(f"Cannot identify positive label '1' in output columns {list(output.columns)}")
                values = output[positive[0]].to_numpy(dtype=np.float64)
                if len(values) != min(int(chunk), len(probabilities) - start) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise RuntimeError("Kumo returned invalid positive-class probabilities")
                probabilities[start:start + len(values)] = values
                prediction = xq = None
        torch.cuda.synchronize(device)
        metadata["positive_label"] = "1"
        metadata["status"] = "complete"
        return probabilities, metadata
    except Exception as error:
        metadata["failed_phase"] = metadata["status"]
        metadata["status"] = "failed"
        metadata["error_type"] = type(error).__name__
        message = f"{type(error).__name__}: {error}"
        # Release tensors held in failed sdm/torch frames as well as our locals.
        error.__traceback__ = None
        raise KumoInferenceError(message + "; no context, column or estimator reduction attempted", metadata) from None
    finally:
        if initialized:
            metadata["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            metadata["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            metadata["max_allocated_gib"] = metadata["max_allocated_bytes"] / 2**30
            metadata["max_reserved_gib"] = metadata["max_reserved_bytes"] / 2**30
        if model is not None:
            model.clear()
        model = xc = yc = xq = prediction = generator = None
        del prepared, recipe
        gc.collect()
        if initialized:
            torch.cuda.empty_cache()


def technical_preflight(frame, fit_rows, query_rows, labels, columns, model_seed,
                        *, query_count=8, **inference_kwargs):
    """Run actual inference on a small query prefix, preserving context and view.

    This is intentionally a GPU operation when called, not a synthetic CPU
    estimate. It reports only engineering provenance/memory and never KS/AUC.
    A successful small-query run does not establish full-query throughput.
    """
    if not isinstance(query_count, (int, np.integer)) or isinstance(query_count, (bool, np.bool_)) or query_count <= 0:
        raise ValueError("query_count must be a positive integer")
    query = _rows(query_rows, len(frame), "query_rows")
    fit = _rows(fit_rows, len(frame), "fit_rows")
    if np.intersect1d(fit, query).size:
        raise ValueError("Fit and the complete supplied query population must be disjoint")
    try:
        _, metadata = predict(frame, fit, query[:query_count], labels, columns,
                              model_seed, **inference_kwargs)
    except KumoInferenceError as error:
        error.metadata.update(technical_preflight=True, scoring_performed=False,
                              original_query_count=len(query), preflight_query_count=min(len(query), int(query_count)))
        raise
    metadata.update(technical_preflight=True, scoring_performed=False,
                    original_query_count=len(query), preflight_query_count=min(len(query), int(query_count)))
    return metadata
