"""TabM trained from competition labels; early stopping is nested inside training."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spline_model import prepare


def neural_imports():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if Path("/tmp/logika-neural-libs").exists():
        sys.path.insert(0, "/tmp/logika-neural-libs")
    import torch
    from tabm import TabM
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

    return torch, TabM, PiecewiseLinearEmbeddings, compute_bins


def make_neural_encoder(data, columns):
    categorical = data[columns].select_dtypes(include="object").columns.tolist()
    numeric = [c for c in columns if c not in categorical]
    encoder = ColumnTransformer(
        [
            (
                "continuous",
                Pipeline(
                    [
                        ("median", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("missing", MissingIndicator(features="all", sparse=False), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    x = encoder.fit_transform(prepare(data, columns, categorical)).astype("float32")
    variable = np.ptp(x, axis=0) > 0
    return encoder, categorical, variable, np.clip(x[:, variable], -5, 5)


def make_neural_model(bins, params):
    torch, TabM, Embeddings, _ = neural_imports()
    # Binary columns naturally have one interval; their embedding remains valid.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The .* feature has just two bin edges.*"
        )
        embeddings = Embeddings(bins, d_embedding=8, activation=True, version="B")
    return TabM.make(
        n_num_features=len(bins),
        d_out=1,
        k=params.get("k", 16),
        n_blocks=params.get("n_blocks", 2),
        d_block=params.get("d_block", 128),
        dropout=params.get("dropout", 0.15),
        num_embeddings=embeddings,
    )


def train_neural(data, indices, spec, seed, validation_indices=None, epochs=None):
    torch, _, _, compute_bins = neural_imports()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    columns = spec["columns"]
    training = data.iloc[indices]
    encoder, categorical, variable, matrix = make_neural_encoder(training, columns)
    x = torch.as_tensor(matrix, device=device)
    y = torch.as_tensor(training.target.to_numpy(dtype="float32"), device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The .* feature has just two bin edges.*"
        )
        bins = compute_bins(torch.as_tensor(matrix), n_bins=16)
    model = make_neural_model(bins, spec["params"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    weight = spec["params"]["positive_weight"]
    pos_weight = torch.tensor(float(weight), device=device)
    if validation_indices is not None:
        validation = data.iloc[validation_indices]
        xv = encoder.transform(prepare(validation, columns, categorical)).astype(
            "float32"
        )
        xv = torch.as_tensor(np.clip(xv[:, variable], -5, 5), device=device)
        yv = torch.as_tensor(validation.target.to_numpy(dtype="float32"), device=device)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, (epochs or 80) + 1):
        model.train()
        shuffled = torch.randperm(len(x), device=device)
        for chunk in shuffled.split(512):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[chunk]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[chunk, None].expand_as(logits), pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
        if validation_indices is not None:
            model.eval()
            with torch.inference_mode():
                pv = torch.cat(
                    [
                        model(chunk).squeeze(-1).sigmoid().mean(1)
                        for chunk in xv.split(2048)
                    ]
                ).clamp(1e-7, 1 - 1e-7)
                # Validate the ensemble prediction; fit members separately, per official TabM recipe.
                current_loss = float(
                    (-(weight * yv * pv.log() + (1 - yv) * (1 - pv).log()))
                    .mean()
                    .item()
                )
            if current_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = current_loss, epoch, 0
            else:
                stale += 1
            if stale >= 12:
                break
        if epoch % 10 == 0:
            print(f"  TabM epoch {epoch}; internal best epoch {best_epoch}", flush=True)
    bundle = {
        "family": "tabm",
        "encoder": encoder,
        "columns": columns,
        "categorical": categorical,
        "variable": variable,
        "bins": [b.cpu() for b in bins],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "weight": weight,
        "params": spec["params"],
        "epochs": epochs or best_epoch,
        "seed": seed,
    }
    del model, x, y, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return bundle, best_epoch


def fit_neural(spec, data, indices, seed):
    inner_train, inner_valid = train_test_split(
        np.asarray(indices),
        test_size=0.125,
        random_state=seed + 4123,
        stratify=data.target.iloc[indices],
    )
    _, epochs = train_neural(
        data, inner_train, spec, seed, validation_indices=inner_valid
    )
    assert epochs > 0
    bundle, _ = train_neural(data, indices, spec, seed, epochs=epochs)
    return bundle


def predict_neural(bundle, data):
    torch, _, _, _ = neural_imports()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = (
        bundle["encoder"]
        .transform(prepare(data, bundle["columns"], bundle["categorical"]))
        .astype("float32")
    )
    x = torch.as_tensor(np.clip(matrix[:, bundle["variable"]], -5, 5), device=device)
    model = make_neural_model(bundle["bins"], bundle["params"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.inference_mode():
        weight = bundle["weight"]
        probabilities = []
        for chunk in x.split(2048):
            members = model(chunk).squeeze(-1).sigmoid()
            # Undo the prior weighting for each member before probability averaging.
            members = members / (weight * (1 - members) + members)
            probabilities.append(members.mean(1))
        p = torch.cat(probabilities).cpu().numpy().astype("float64")
    del model, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return p
