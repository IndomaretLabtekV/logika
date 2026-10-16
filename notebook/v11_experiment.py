"""Nested TabM tuning. Execution is restricted to Kaggle CUDA."""
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import tabm
import rtdl_num_embeddings
from scipy.special import ndtri
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, log_loss

CANDIDATES = {
    "control": dict(lr=.002, wd=.0003, bins=48),
    "lower_lr": dict(lr=.0005, wd=.0003, bins=48),
    "stronger_wd": dict(lr=.002, wd=.01, bins=48),
    "fewer_bins": dict(lr=.002, wd=.0003, bins=16),
}


def tabm_fit(X, y, rest, settings, out, tag, epochs=60, stop_y=None):
    assert Path("/kaggle/input").exists() and torch.cuda.is_available()
    assert len(X) == len(y) and set(np.unique(y)) == {0, 1}
    assert all(list(r.columns) == list(X.columns) for r in rest)
    assert stop_y is None or len(stop_y) == sum(len(r) for r in rest)
    torch.manual_seed(42)
    cats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in X if c not in cats]
    rest_x = pd.concat(rest, ignore_index=True)
    xn, rn, _ = nn_prep(X[nums].to_numpy(dtype=float), rest_x[nums].to_numpy(dtype=float))
    xc, rc, cards = [], [], []
    for c in cats:
        categories = sorted(X[c].dropna().unique())
        xc.append(pd.Categorical(X[c], categories=categories).codes.astype(np.int64) + 1)
        rc.append(pd.Categorical(rest_x[c], categories=categories).codes.astype(np.int64) + 1)
        cards.append(len(categories) + 1)
    assert np.isfinite(xn).all() and np.isfinite(rn).all()
    xn, rn = [torch.as_tensor(a, device="cuda") for a in (xn, rn)]
    xc, rc = [torch.as_tensor(np.column_stack(a), device="cuda") for a in (xc, rc)]
    yt = torch.as_tensor(y, dtype=torch.float32, device="cuda")
    emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(xn, n_bins=settings["bins"]),
        d_embedding=16, activation=False, version="B")
    model = tabm.TabM.make(n_num_features=xn.shape[1], cat_cardinalities=cards,
                         d_out=1, num_embeddings=emb).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["wd"])

    def predict():
        model.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(model(rn[i:i+2048], rc[i:i+2048]).squeeze(-1)).mean(1)
                           for i in range(0, len(rn), 2048)]).cpu().numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        return p

    best_auc, best_epoch, best_state, wait = -1., epochs, None, 0
    curves = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for idx in torch.randperm(len(xn), device="cuda").split(512):
            logits = model(xn[idx], xc[idx]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx].unsqueeze(1).expand_as(logits))
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += loss.item() * len(idx)
        row = dict(epoch=epoch, online_member_bce=total / len(xn))
        if stop_y is not None:
            pred = predict()
            auc = roc_auc_score(stop_y, pred)
            row.update(inner_auc=auc, inner_ks=ks(stop_y, pred),
                       inner_ensemble_bce=log_loss(stop_y, pred, labels=[0, 1]))
            if auc > best_auc:
                best_auc, best_epoch, wait = auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        curves.append(row)
        pd.DataFrame(curves).to_csv(out / f"curve_{tag}.csv", index=False)
        print(f"{tag} epoch {epoch}: {row}", flush=True)
        if stop_y is not None and wait >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict()
    del model, opt, xn, rn, xc, rc, yt, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return np.split(pred, np.cumsum([len(r) for r in rest])[:-1]), best_epoch


def run_tabm(X, y, test, out):
    oof = {n: np.full(len(y), np.nan) for n in ("control", "tuned")}
    test_pred = {n: np.zeros(len(test)) for n in oof}
    fold_ids = np.full(len(y), -1)
    trials = []
    deadline = time.monotonic() + 11 * 3600
    for fold, (train, valid) in enumerate(splits(y, 5, 42)):
        fit, stop = next(splits(y[train], 5, 170 + fold))
        assert not np.intersect1d(train, valid).size
        assert not np.intersect1d(fit, stop).size and len(fit) + len(stop) == len(train)
        scores = {}
        for name, settings in CANDIDATES.items():
            if time.monotonic() > deadline:
                raise TimeoutError("Partial curves saved; no incomplete submission emitted.")
            predictions, epoch = tabm_fit(X.iloc[train[fit]], y[train[fit]],
                [X.iloc[train[stop]]], settings, out, f"f{fold}_{name}_inner",
                stop_y=y[train[stop]])
            scores[name] = (ks(y[train[stop]], predictions[0]), epoch)
            trials.append(dict(fold=fold, candidate=name, inner_ks=scores[name][0], epochs=epoch))
            pd.DataFrame(trials).to_csv(out / "inner_trials.csv", index=False)
        winner = max(scores, key=lambda n: scores[n][0])
        for policy, name in [("control", "control"), ("tuned", winner)]:
            (pv, pt), epoch = tabm_fit(X.iloc[train], y[train], [X.iloc[valid], test],
                CANDIDATES[name], out, f"f{fold}_{policy}_outer", epochs=scores[name][1])
            oof[policy][valid] = pv
            test_pred[policy] += pt / 5
            np.savez(out / f"{policy}_fold{fold}.npz", rows=valid, prediction=pv, test=pt,
                     candidate=name, epochs=epoch)
        fold_ids[valid] = fold
        print(f"Outer fold {fold} complete; selected {winner}", flush=True)
    assert (fold_ids >= 0).all() and all(np.isfinite(p).all() for p in oof.values())
    return oof, test_pred, fold_ids


def main_tabm():
    assert Path("/kaggle/input").exists() and torch.cuda.is_available(), "Kaggle CUDA only"
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    out = Path("/kaggle/working/v11")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    assert not set(tables["train_applications"]["customer_id"]) & set(tables["test_applications"]["customer_id"])
    features = build_all(tables).sort("application_id")
    assert features["application_id"].n_unique() == features.height
    train, test = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    assert train["customer_id"].n_unique() == train.height
    cols = [c for c in features.columns if c not in
            {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    X, Xt = train.select(cols).to_pandas(), test.select(cols).to_pandas()
    y = train["target"].to_numpy().astype(int)
    manifest = dict(packages={n: importlib.metadata.version(n) for n in
                    ["torch", "tabm", "rtdl_num_embeddings", "polars", "numpy", "scikit-learn"]},
                    candidates=CANDIDATES, feature_sha256=FEATURE_SHA256, nn_prep_sha256=NN_PREP_SHA256,
                    columns=cols, seed=42, outer_folds=5, inner_holdout=.2, max_epochs=60, patience=8,
                    caveat="Single-seed nested holdout tuning; compare control and tuned within this run.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    train.select("application_id", "target").write_parquet(out / "train_ids.parquet")
    test.select("application_id").write_parquet(out / "test_ids.parquet")
    oof, predictions, folds = run_tabm(X, y, Xt, out)
    train.select("application_id", "target").with_columns(
        fold=pl.Series(folds), **{k: pl.Series(v) for k, v in oof.items()}).write_parquet(out / "oof.parquet")
    evaluate(y, oof, folds, out)
    for name, pred in predictions.items():
        assert len(pred) == test.height and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
        test.select("application_id").with_columns(default_probability=pl.Series(pred)).write_csv(
            out / f"submission_{name}.csv")
    curves = pd.concat([pd.read_csv(p).assign(run=p.stem) for p in out.glob("curve_*inner.csv")])
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, group in curves.groupby("run"):
        ax.plot(group.epoch, group.inner_auc, alpha=.5, label=name)
    ax.set(xlabel="Epoch", ylabel="Inner validation AUC", title="TabM training diagnostics")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=140)
    plt.close(fig)
