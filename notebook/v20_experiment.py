"""v20: Causilo / TabPFN with 16 estimators (v8 used 4) on the full training fold. Kaggle T4x2 only."""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.stats import ks_2samp

N_FEAT, CHUNK = 100, 5000
DEADLINE = time.monotonic() + 11.3 * 3600
# (name, library, n_estimators, n_numeric_features)
CONFIGS = [("causilo16", "causilo", 16, 100), ("tabpfn16", "tabpfn", 16, 100), ("causilo16_f200", "causilo", 16, 200)]


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(np.arange(len(y)), valid), valid


def select(X, y, n):
    """Top n numeric columns by univariate KS on training rows only (NaN rows excluded), greedy |rho|<0.95."""
    scores = []
    for j in range(X.shape[1]):
        m = np.isfinite(X[:, j])
        scores.append(ks(y[m], X[m, j]) if m.sum() > 100 and len(np.unique(y[m])) == 2 and np.nanstd(X[m, j]) > 0 else 0)
    order = np.argsort(scores)[::-1]
    rank = lambda v: np.argsort(np.argsort(np.where(np.isfinite(v), v, np.nanmedian(v))))  # noqa: E731
    sub = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)
    keep, ranks = [], []
    for j in order:
        r = rank(X[sub, j]).astype(float)
        if all(abs(np.corrcoef(r, q)[0, 1]) < 0.95 for q in ranks):
            keep.append(j)
            ranks.append(r)
        if len(keep) == n:
            break
    return keep


class Causilo:
    def __init__(self, n_est, cat_idx, device):
        from causilo import CausiloClassifier
        self.cat_idx = cat_idx
        self.m = CausiloClassifier(n_estimators=n_est, random_state=42, device=device)

    def frame(self, X):
        d = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        for j in self.cat_idx:
            d[f"f{j}"] = pd.Categorical(d[f"f{j}"], categories=self.cats[j])
        return d

    def fit(self, X, yy):
        self.cats = {j: np.unique(X[:, j][np.isfinite(X[:, j])]) for j in self.cat_idx}
        self.m.fit(self.frame(X), yy)
        return self

    def predict_proba(self, X):
        return np.asarray(self.m.predict_proba(self.frame(X)))


def make(lib, n_est, cat_idx, device):
    if lib == "causilo":
        return Causilo(n_est, cat_idx, device)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(n_estimators=n_est, device=device, ignore_pretraining_limits=True,
                            categorical_features_indices=cat_idx, random_state=42)


def run_fold(cfg, fold, tr, va, X, Xt, y, nums, cats, device, out, res):
    name, lib, n_est, n_feat = cfg
    keep = [nums[j] for j in select(X[tr][:, nums], y[tr], n_feat)] + cats
    cat_idx = list(range(len(keep) - len(cats), len(keep)))
    Q = np.vstack([X[va], Xt])[:, keep]
    t0 = time.monotonic()
    m = make(lib, n_est, cat_idx, device)
    m.fit(X[tr][:, keep], y[tr])
    pred = []
    for s in range(0, len(Q), CHUNK):
        if time.monotonic() > DEADLINE:
            raise TimeoutError(f"{name} fold {fold} hit the deadline")
        pred.append(m.predict_proba(Q[s:s + CHUNK])[:, 1])
        if s == 0:
            print(f"{name} fold {fold} first chunk {time.monotonic() - t0:.0f}s", flush=True)
    p = np.concatenate(pred)
    res[fold] = (p[:len(va)], p[len(va):])
    np.savez(out / f"{name}_fold{fold}.npz", valid=va, oof=res[fold][0], test=res[fold][1])
    print(f"{name} fold {fold} KS {ks(y[va], res[fold][0]):.5f} in {time.monotonic() - t0:.0f}s on {device}", flush=True)
    del m
    torch.cuda.empty_cache()


def main_v20(build_all):
    assert Path("/kaggle/input").exists() and torch.cuda.is_available()
    try:  # TabPFN weights are gated; the token is only present if a Kaggle secret is attached
        from kaggle_secrets import UserSecretsClient
        for k in ("TABPFN_TOKEN", "HF_TOKEN"):
            try:
                os.environ.setdefault(k, UserSecretsClient().get_secret(k))
            except Exception:  # noqa: BLE001
                pass
        if "TABPFN_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
            os.environ["TABPFN_TOKEN"] = os.environ["HF_TOKEN"]
    except ImportError:
        pass
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    out = Path("/kaggle/working/v20")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id")
    train, test = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    catc = [c for c in cols if features[c].dtype == pl.String]
    numc = [c for c in cols if c not in catc]
    enc = lambda df: np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in numc] +  # noqa: E731
                                     [df[c].fill_null("_").replace_strict(sorted(features[c].fill_null("_").unique().to_list()),
                                      list(range(features[c].fill_null("_").n_unique())), return_dtype=pl.Float64).to_numpy()
                                      for c in catc]).astype(np.float32)
    X, Xt = enc(train), enc(test)
    X[np.isinf(X)], Xt[np.isinf(Xt)] = np.nan, np.nan
    nums, cats = list(range(len(numc))), list(range(len(numc), len(numc) + len(catc)))
    y = train["target"].to_numpy().astype(int)
    folds = list(splits(y, 5, 42))
    gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    oof_all, tst_all, metrics = {}, {}, {}
    for cfg in CONFIGS:
        name = cfg[0]
        res, failed = {}, []

        def worker(g):
            for f in range(g, 5, len(gpus)):
                try:
                    run_fold(cfg, f, *folds[f], X, Xt, y, nums, cats, gpus[g], out, res)
                except Exception as e:  # noqa: BLE001
                    failed.append(f)
                    print(f"{name} fold {f} failed: {e!r}", flush=True)
                    return
        th = [threading.Thread(target=worker, args=(g,)) for g in range(len(gpus))]
        [t.start() for t in th]
        [t.join() for t in th]
        if len(res) < 5:
            print(f"{name} dropped: folds done {sorted(res)}", flush=True)
            continue
        oof = np.zeros(len(y))
        for f in range(5):
            oof[folds[f][1]] = res[f][0]
        oof_all[name] = oof
        tst_all[name] = np.mean([res[f][1] for f in range(5)], axis=0)
        metrics[name] = ks(y, oof)
        print(f"{name} OOF KS {metrics[name]:.5f}", flush=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        pl.DataFrame({"application_id": train["application_id"], "target": y, **oof_all}).write_parquet(out / "oof.parquet")
        pl.DataFrame({"application_id": test["application_id"], **tst_all}).write_parquet(out / "test.parquet")
