"""v16: TabICLv2 full five-fold OOF with chunked queries (v8 died scoring 44k query rows in one call). Kaggle T4 only."""
import json
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import ks_2samp
from tabicl import TabICLClassifier

N_FEAT, CTX, CHUNK, N_EST = 100, 40000, 2000, 4
DEADLINE = time.monotonic() + 11 * 3600


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(np.arange(len(y)), valid), valid


def select(X, y):
    """Top N_FEAT numeric columns by univariate KS on training rows only (NaN rows excluded), greedy |rho|<0.95."""
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
        if len(keep) == N_FEAT:
            break
    return keep


def add_missing_groups(Xc, Xq):
    """One 0/1 column per distinct missingness pattern of the context columns (thin-file vs history)."""
    seen, cols = set(), []
    miss_c = ~np.isfinite(Xc)
    for j in range(Xc.shape[1]):
        key = np.packbits(miss_c[:, j]).tobytes()
        if miss_c[:, j].any() and key not in seen:
            seen.add(key)
            cols.append(j)
    f = lambda A: (~np.isfinite(A[:, cols])).astype(np.float32)  # noqa: E731
    return np.hstack([Xc, f(Xc)]), np.hstack([Xq, f(Xq)])


def run_fold(fold, tr, va, X, Xt, y, device, out, res):
    keep = select(X[tr], y[tr])
    rng = np.random.default_rng(fold)
    ctx = tr if len(tr) <= CTX else np.sort(np.concatenate(
        [rng.choice(tr[y[tr] == c], int(round(CTX * (y[tr] == c).mean())), replace=False) for c in (0, 1)]))
    Q = np.vstack([X[va], Xt])
    Xc, Xq = add_missing_groups(X[ctx][:, keep], Q[:, keep])
    m = TabICLClassifier(n_estimators=N_EST, batch_size=1, device=device, random_state=42, kv_cache=False)
    t0 = time.monotonic()
    m.fit(Xc, y[ctx])
    pred = []
    for s in range(0, len(Xq), CHUNK):
        if time.monotonic() > DEADLINE:
            raise TimeoutError(f"fold {fold} hit the deadline")
        pred.append(m.predict_proba(Xq[s:s + CHUNK])[:, 1])
        if s == 0:
            print(f"fold {fold} first chunk {time.monotonic() - t0:.0f}s", flush=True)
    p = np.concatenate(pred)
    pv, pt = p[:len(va)], p[len(va):]
    np.savez(out / f"fold{fold}.npz", valid=va, oof=pv, test=pt)
    res[fold] = (pv, pt)
    print(f"fold {fold} KS {ks(y[va], pv):.5f} in {time.monotonic() - t0:.0f}s on {device}", flush=True)


def main_v16(build_all):
    assert Path("/kaggle/input").exists() and torch.cuda.is_available()
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    out = Path("/kaggle/working/v16")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id")
    train, test = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    cats = [c for c in cols if features[c].dtype == pl.String]
    nums = [c for c in cols if c not in cats]
    enc = lambda df: np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in nums] +  # noqa: E731
                                     [df[c].fill_null("_").replace_strict(sorted(features[c].fill_null("_").unique().to_list()),
                                      list(range(features[c].fill_null("_").n_unique())), return_dtype=pl.Float64).to_numpy()
                                      for c in cats]).astype(np.float32)
    X, Xt = enc(train), enc(test)
    X[np.isinf(X)], Xt[np.isinf(Xt)] = np.nan, np.nan
    y = train["target"].to_numpy().astype(int)
    folds = list(splits(y, 5, 42))
    gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    res = {}
    # one worker thread per GPU, each takes folds round-robin
    def worker(g):
        for f in range(g, 5, len(gpus)):
            try:
                run_fold(f, *folds[f], X, Xt, y, gpus[g], out, res)
            except Exception as e:  # noqa: BLE001
                print(f"fold {f} failed: {e!r}", flush=True)
    th = [threading.Thread(target=worker, args=(g,)) for g in range(len(gpus))]
    [t.start() for t in th]
    [t.join() for t in th]
    done = sorted(res)
    print("completed folds", done)
    oof = np.full(len(y), np.nan)
    tst = np.zeros(test.height)
    for f in done:
        oof[folds[f][1]] = res[f][0]
        tst += res[f][1] / len(done)
    pl.DataFrame({"application_id": train["application_id"], "target": y, "tabicl": oof}).write_parquet(out / "oof.parquet")
    pl.DataFrame({"application_id": test["application_id"], "tabicl": tst}).write_parquet(out / "test.parquet")
    m = np.isfinite(oof)
    (out / "metrics.json").write_text(json.dumps(dict(folds=done, ks=ks(y[m], oof[m])), indent=2))
    print("TabICL OOF KS", ks(y[m], oof[m]))
