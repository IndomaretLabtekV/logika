"""v18: inductive-bias variants (linear_tree, monotone constraints, ExtraTrees) on hand features. Kaggle CPU only."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from scipy.stats import ks_2samp

V8 = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabm", "tabpfn", "causilo"]
LGB = dict(learning_rate=0.02, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=4, metric="auc")
VARIANTS = {
    "lgb_linear": dict(kind="lgb", feats="hand", params=dict(LGB, objective="binary", linear_tree=True, linear_lambda=1.0,
                                                            num_leaves=15, min_data_in_leaf=300)),
    "lgb_mono": dict(kind="lgb_mono", feats="hand", params=dict(LGB, objective="binary", num_leaves=15, min_data_in_leaf=200,
                                                               monotone_constraints_method="advanced")),
    "extratrees": dict(kind="et", feats="hand"),
}


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def splits(y, n, seed):
    rng = np.random.default_rng(seed)
    parts = [np.array_split(rng.permutation(np.flatnonzero(y == c)), n) for c in (0, 1)]
    for k in range(n):
        valid = np.sort(np.concatenate([p[k] for p in parts]))
        yield np.setdiff1d(np.arange(len(y)), valid), valid


def groups(n, size, seed):
    """Random query groups for ranking objectives: rows are shuffled, then cut into blocks of `size`."""
    perm = np.random.default_rng(seed).permutation(n)
    sizes = [size] * (n // size) + ([n % size] if n % size else [])
    return perm, np.array(sizes)


def fit_one(spec, X, y, Xs, ys, preds):
    kind = spec["kind"]
    if kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        enc = lambda D: D.apply(lambda c: c.cat.codes if hasattr(c, "cat") else c).fillna(-999).to_numpy(np.float32)  # noqa: E731
        m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=40, max_features=0.3, n_jobs=4, random_state=0)
        m.fit(enc(X), y)
        return [m.predict_proba(enc(P))[:, 1] for P in preds], 600
    if kind == "lgb_mono":
        # sign of the training-rows Spearman correlation for numeric columns with a clear univariate effect
        mono = []
        for c in X.columns:
            if hasattr(X[c], "cat"):
                mono.append(0)
                continue
            v = X[c].to_numpy(dtype=float)
            m_ = np.isfinite(v)
            r = pd.Series(v[m_]).rank().corr(pd.Series(y[m_].astype(float))) if m_.sum() > 1000 else 0
            mono.append(int(np.sign(r)) if abs(r) > 0.05 else 0)
        spec = dict(spec, kind="lgb", params=dict(spec["params"], monotone_constraints=mono))
        kind = "lgb"
    if kind in ("lgb", "lgb_rank"):
        if kind == "lgb_rank":
            perm, g = groups(len(y), 1000, 0)
            dtr = lgb.Dataset(X.iloc[perm], y[perm], group=g)
            ps, gs = groups(len(ys), 1000, 1)
            dst = lgb.Dataset(Xs.iloc[ps], ys[ps], group=gs, reference=dtr)
            params = dict(spec["params"], metric="auc")
        else:
            dtr = lgb.Dataset(X, y)
            dst = lgb.Dataset(Xs, ys, reference=dtr)
            params = spec["params"]
        m = lgb.train(params, dtr, 5000, valid_sets=[dst], callbacks=[lgb.early_stopping(200, verbose=False)])
        return [m.predict(P, num_iteration=m.best_iteration) for P in preds], m.best_iteration
    perm, g = groups(len(y), 1000, 0)
    dtr = xgb.DMatrix(X.iloc[perm], y[perm], enable_categorical=True)
    dtr.set_group(g)
    dst = xgb.DMatrix(Xs, ys, enable_categorical=True)
    params = dict(objective="rank:pairwise", eta=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.5,
                  min_child_weight=20, eval_metric="auc", tree_method="hist", nthread=4)
    m = xgb.train(params, dtr, 5000, evals=[(dst, "s")], early_stopping_rounds=200, verbose_eval=False)
    it = (0, m.best_iteration + 1)
    return [m.predict(xgb.DMatrix(P, enable_categorical=True), iteration_range=it) for P in preds], m.best_iteration


def main_v18(build_all):
    assert Path("/kaggle/input").exists()
    data = next(Path("/kaggle/input").rglob("train_applications.csv")).parent
    v8dir = next(Path("/kaggle/input").rglob("v8_oof.parquet")).parent
    out = Path("/kaggle/working/v18")
    out.mkdir(exist_ok=True)
    names = ["train_applications", "test_applications", "customer_profiles", "previous_applications",
             "repayment_history", "account_behavior", "bureau_observations", "ecommerce_transactions"]
    tables = {n: pl.read_csv(data / f"{n}.csv", try_parse_dates=True, infer_schema_length=100000) for n in names}
    features = build_all(tables).sort("application_id")
    train, test = (features.filter(pl.col("is_train") == v) for v in (1, 0))
    lo = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))  # noqa: E731
    o = train.select("application_id").join(pl.read_parquet(v8dir / "v8_oof.parquet"), on="application_id",
                                            validate="1:1", maintain_order="left")
    t = test.select("application_id").join(pl.read_parquet(v8dir / "v8_test.parquet"), on="application_id",
                                           validate="1:1", maintain_order="left")
    assert o.null_count().sum_horizontal()[0] == 0 and t.null_count().sum_horizontal()[0] == 0
    meta_tr = pd.DataFrame({f"m_{n}": lo(o[n].to_numpy()) for n in V8})
    meta_te = pd.DataFrame({f"m_{n}": lo(t[n].to_numpy()) for n in V8})
    cols = [c for c in features.columns if c not in {"application_id", "customer_id", "decision_datetime", "target", "is_train"}]
    X, Xt = train.select(cols).to_pandas(), test.select(cols).to_pandas()
    for c in cols:
        if features[c].dtype == pl.String:
            cats = sorted(X[c].dropna().unique())
            X[c], Xt[c] = pd.Categorical(X[c], categories=cats), pd.Categorical(Xt[c], categories=cats)
        else:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            Xt[c] = Xt[c].replace([np.inf, -np.inf], np.nan)
    views = {"hand": (X, Xt),
             "meta": (pd.concat([meta_tr, X[["has_history"]]], axis=1), pd.concat([meta_te, Xt[["has_history"]]], axis=1)),
             "full": (pd.concat([meta_tr, X], axis=1), pd.concat([meta_te, Xt], axis=1))}
    y = train["target"].to_numpy().astype(int)
    oof = {v: np.full(len(y), np.nan) for v in VARIANTS}
    tst = {v: np.zeros(test.height) for v in VARIANTS}
    folds = np.full(len(y), -1)
    log = []
    for fold, (tr, va) in enumerate(splits(y, 5, 42)):
        fit, stop = next(splits(y[tr], 7, 300 + fold))
        for name, spec in VARIANTS.items():
            A, At = views[spec["feats"]]
            t0 = time.monotonic()
            (pv, pt), it = fit_one(spec, A.iloc[tr[fit]], y[tr[fit]], A.iloc[tr[stop]], y[tr[stop]], [A.iloc[va], At])
            # rank outputs are unbounded scores: map to (0,1) by rank so every variant is a valid probability
            oof[name][va] = pv
            rank_kind = spec["kind"] in ("lgb_rank", "xgb_rank")
            tst[name] += (pd.Series(pt).rank().to_numpy() / len(pt) if rank_kind else pt) / 5
            log.append(dict(fold=fold, variant=name, iters=int(it), ks=ks(y[va], pv), sec=time.monotonic() - t0))
            print(log[-1], flush=True)
            pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        folds[va] = fold
    # per-fold rank normalisation so pooled OOF of unbounded ranking scores is comparable across folds
    for name in [n for n, v in VARIANTS.items() if v["kind"] in ("lgb_rank", "xgb_rank")]:
        r = np.zeros(len(y))
        for k in range(5):
            m = folds == k
            r[m] = pd.Series(oof[name][m]).rank().to_numpy() / m.sum()
        oof[name] = np.clip(r, 1e-4, 1 - 1e-4)
    res = {n: ks(y, oof[n]) for n in VARIANTS}
    print(res)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pl.DataFrame({"application_id": train["application_id"], "target": y, "fold": folds, **oof}).write_parquet(out / "oof.parquet")
    pl.DataFrame({"application_id": test["application_id"], **{n: np.clip(v, 1e-4, 1 - 1e-4) for n, v in tst.items()}}).write_parquet(out / "test.parquet")
