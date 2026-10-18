"""validation_v2: one protocol for local screening (one fold) and full Kaggle runs.

Outer: StratifiedKFold(5, shuffle, seed) per seed; the outer validation fold is only scored, never used for any decision.
Inner: a stratified 1/7 split of the outer training fold picks the iteration / epoch (early stopping). The model is then
refitted on the whole outer training fold with that fixed budget (GBDT: best x REFIT, TabM: best epoch).
Every run writes OOF and test predictions with application_id. Blend weights are chosen later, cross-fitted over the
outer folds (eda/blend2.py), never on the rows they are scored on."""
import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

ID_COLS = ["application_id", "customer_id", "decision_datetime", "target", "is_train"]
CAT_COLS = ["loan_purpose", "region", "employment_type", "housing_status"]
N_FOLDS, INNER, REFIT = 5, 7, 1.15
SEEDS = [42, 7, 2026]
N_ROUNDS, EARLY_STOP = 10000, 200
THREADS = os.cpu_count() or 4

LGB = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=31, min_child_samples=100,
           feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=THREADS)
LGB_S = dict(LGB, num_leaves=7, max_depth=3, min_child_samples=200, feature_fraction=0.4, lambda_l2=10.0)
XGB = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.02, max_depth=5, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist", device="cuda")
XGB_S = dict(XGB, max_depth=3, min_child_weight=20, colsample_bytree=0.4, reg_lambda=10.0)
CAT = dict(learning_rate=0.03, depth=6, l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
           task_type="GPU", verbose=0)
CAT_S = dict(CAT, depth=4, l2_leaf_reg=10)
TABM = dict(lr=2e-3, wd=3e-4, bins=48, batch=512, epochs=60, patience=8)


def ks(y, p):
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def outer(y, seed):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def inner(y_tr, seed, fold):
    """(fit, stop) positions inside the outer training fold."""
    return next(StratifiedKFold(INNER, shuffle=True, random_state=1000 * seed + fold).split(np.zeros(len(y_tr)), y_tr))


def frames(features: pl.DataFrame):
    """features (build_all output) -> X, Xt (pandas; categoricals typed with train categories), y, train ids, test ids."""
    f = features.sort("application_id")
    tr, te = f.filter(pl.col("is_train") == 1), f.filter(pl.col("is_train") == 0)
    cols = sorted(c for c in f.columns if c not in ID_COLS)
    X, Xt = tr.select(cols).to_pandas(), te.select(cols).to_pandas()
    for c in cols:
        if c in CAT_COLS or f[c].dtype == pl.String:
            cats = sorted(X[c].dropna().unique())
            X[c], Xt[c] = pd.Categorical(X[c], categories=cats), pd.Categorical(Xt[c], categories=cats)
        else:
            X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
            Xt[c] = Xt[c].astype(float).replace([np.inf, -np.inf], np.nan)
    return X, Xt, tr["target"].to_numpy().astype(int), tr["application_id"], te["application_id"]


# --- model fits: fit(X, y, tr, fit, stop, rest, seed) -> list of predictions for each frame in rest ---------------

def lgb_fit(params):
    import lightgbm as lgb

    def fit(X, y, tr, fi, st, rest, seed):
        p = dict(params, seed=seed)
        d_fit = lgb.Dataset(X.iloc[tr[fi]], y[tr[fi]])
        d_st = lgb.Dataset(X.iloc[tr[st]], y[tr[st]], reference=d_fit)
        m = lgb.train(p, d_fit, N_ROUNDS, valid_sets=[d_st], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        n = math.ceil(m.best_iteration * REFIT)
        m = lgb.train(p, lgb.Dataset(X.iloc[tr], y[tr]), n)
        return [m.predict(r) for r in rest], n
    return fit


def xgb_fit(params):
    import xgboost as xgb

    def fit(X, y, tr, fi, st, rest, seed):
        p = dict(params, seed=seed)
        d_fit = xgb.DMatrix(X.iloc[tr[fi]], y[tr[fi]], enable_categorical=True)
        d_st = xgb.DMatrix(X.iloc[tr[st]], y[tr[st]], enable_categorical=True)
        m = xgb.train(p, d_fit, N_ROUNDS, evals=[(d_st, "st")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        n = math.ceil((m.best_iteration + 1) * REFIT)
        m = xgb.train(p, xgb.DMatrix(X.iloc[tr], y[tr], enable_categorical=True), n)
        return [m.predict(xgb.DMatrix(r, enable_categorical=True)) for r in rest], n
    return fit


def cat_fit(params):
    from catboost import CatBoostClassifier, Pool

    def s(df):
        out = df.copy()
        for c in CAT_COLS:
            out[c] = out[c].astype(str)
        return out

    def fit(X, y, tr, fi, st, rest, seed):
        cats = [c for c in CAT_COLS if c in X.columns]
        m = CatBoostClassifier(**params, iterations=N_ROUNDS, od_type="Iter", od_wait=EARLY_STOP, use_best_model=True,
                               random_seed=seed)
        m.fit(Pool(s(X.iloc[tr[fi]]), y[tr[fi]], cat_features=cats), eval_set=Pool(s(X.iloc[tr[st]]), y[tr[st]], cat_features=cats))
        n = math.ceil((m.get_best_iteration() + 1) * REFIT)
        m = CatBoostClassifier(**params, iterations=n, random_seed=seed)
        m.fit(Pool(s(X.iloc[tr]), y[tr], cat_features=cats))
        return [m.predict_proba(s(r))[:, 1] for r in rest], n
    return fit


def tabm_fit(cfg=TABM, device="cuda"):
    import torch
    import tabm
    import rtdl_num_embeddings
    from sklearn.metrics import roc_auc_score
    from nn_prep import nn_prep

    def arrays(X, rows_list):
        nums = [c for c in X.columns if c not in CAT_COLS]
        return [X.iloc[r][nums].to_numpy(dtype=float) for r in rows_list]

    def codes(df):
        return np.column_stack([df[c].cat.codes.to_numpy() + 1 for c in CAT_COLS]).astype(np.int64)

    def train(xn, xc, yv, cards, n_epochs, seed, watch=None):
        torch.manual_seed(seed)
        dev = torch.device(device)
        xn_t, xc_t = torch.as_tensor(xn, device=dev), torch.as_tensor(xc, device=dev)
        yt = torch.as_tensor(yv, dtype=torch.float32, device=dev)
        emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            rtdl_num_embeddings.compute_bins(xn_t, n_bins=cfg["bins"]), d_embedding=16, activation=False, version="B")
        model = tabm.TabM.make(n_num_features=xn.shape[1], cat_cardinalities=cards, d_out=1, num_embeddings=emb).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        def predict(a, c):
            model.eval()
            a, c = torch.as_tensor(a, device=dev), torch.as_tensor(c, device=dev)
            with torch.no_grad():
                return torch.cat([torch.sigmoid(model(a[i:i + 1024], c[i:i + 1024]).squeeze(-1).float()).mean(1)
                                  for i in range(0, len(a), 1024)]).cpu().numpy()

        best, best_ep, wait = -1.0, n_epochs, 0
        for ep in range(1, n_epochs + 1):
            model.train()
            for idx in torch.randperm(len(xn_t), device=dev).split(cfg["batch"]):
                out = model(xn_t[idx], xc_t[idx]).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[idx].unsqueeze(1).expand_as(out))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if watch is not None:
                auc = roc_auc_score(watch[2], predict(watch[0], watch[1]))
                best, best_ep, wait = (auc, ep, 0) if auc > best else (best, best_ep, wait + 1)
                if wait >= cfg["patience"]:
                    break
        return model, predict, best_ep

    def fit(X, y, tr, fi, st, rest, seed):
        cards = [len(X[c].cat.categories) + 1 for c in CAT_COLS]
        a_fit, a_st = arrays(X, [tr[fi], tr[st]])
        x_fit, x_st, _ = nn_prep(a_fit, a_st)
        _, _, ep = train(x_fit, codes(X.iloc[tr[fi]]), y[tr[fi]], cards, cfg["epochs"], seed,
                         watch=(x_st, codes(X.iloc[tr[st]]), y[tr[st]]))
        nums = [c for c in X.columns if c not in CAT_COLS]
        a_tr = X.iloc[tr][nums].to_numpy(dtype=float)
        a_rest = np.vstack([r[nums].to_numpy(dtype=float) for r in rest])
        x_tr, x_rest, _ = nn_prep(a_tr, a_rest)
        model, predict, _ = train(x_tr, codes(X.iloc[tr]), y[tr], cards, ep, seed)
        p = predict(x_rest, np.vstack([codes(r) for r in rest]))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return np.split(p, np.cumsum([len(r) for r in rest])[:-1]), ep
    return fit


def ks_nan(y, x):
    x = np.asarray(x, float)
    if np.isnan(x).all():
        return 0.0
    lo, hi = np.nanmin(x) - 1, np.nanmax(x) + 1
    return max(ks(y, np.where(np.isnan(x), lo, x)), ks(y, np.where(np.isnan(x), hi, x)))


def fm_select(X, y, tr, n_num=100):
    """v8 recipe: numeric columns ranked by univariate KS on the training fold, greedy |Spearman| < 0.95, top n_num,
    plus the 4 categoricals."""
    nums = [c for c in X.columns if c not in CAT_COLS]
    sub = X.iloc[tr]
    ranked = sorted(nums, key=lambda c: -ks_nan(y[tr], sub[c].to_numpy(dtype=float)))[:2 * n_num]
    corr = sub[ranked].rank().corr().abs().fillna(0.0)
    keep = []
    for c in ranked:
        if all(corr.loc[c, k] < 0.95 for k in keep):
            keep.append(c)
        if len(keep) == n_num:
            break
    return keep + CAT_COLS


def icl_fit(make, n_num=100, max_context=60000, chunk=10000, as_codes=False):
    """In-context learners: no early stopping (the inner split is unused). make(seed, cat_idx) -> sklearn-like model."""
    def fit(X, y, tr, fi, st, rest, seed):
        cols = fm_select(X, y, tr, n_num)
        ctx = tr if len(tr) <= max_context else np.random.default_rng(seed).choice(tr, max_context, replace=False)

        def frame(d):
            d = d[cols].reset_index(drop=True)
            if as_codes:
                d = d.copy()
                for c in CAT_COLS:
                    d[c] = d[c].cat.codes.astype(float).replace(-1, np.nan)
                return d.to_numpy(dtype=np.float32)
            return d
        m = make(seed, [cols.index(c) for c in CAT_COLS])
        m.fit(frame(X.iloc[ctx]), y[ctx])
        preds = [np.concatenate([np.asarray(m.predict_proba(frame(r.iloc[i:i + chunk])))[:, 1]
                                 for i in range(0, len(r), chunk)]) for r in rest]
        import torch
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return preds, len(cols)
    return fit


def causilo_fit(n_num=100, n_est=4, max_context=60000, device="cuda"):
    from causilo import CausiloClassifier
    return icl_fit(lambda seed, ci: CausiloClassifier(n_estimators=n_est, random_state=seed, device=device),
                   n_num, max_context)


def tabpfn_fit(n_num=100, n_est=4, max_context=60000, device="cuda"):
    from tabpfn import TabPFNClassifier
    return icl_fit(lambda seed, ci: TabPFNClassifier(n_estimators=n_est, random_state=seed, device=device,
                                                     ignore_pretraining_limits=True, categorical_features_indices=ci,
                                                     fit_mode="fit_with_cache"),
                   n_num, max_context, chunk=5000, as_codes=True)


def tabdpt_fit(n_num=100, context=4096, n_ens=8, device="cuda"):
    """TabDPT retrieves `context` nearest training rows per query (its own context reduction)."""
    from tabdpt import TabDPTClassifier

    class M:
        def __init__(self, seed):
            self.m, self.seed = TabDPTClassifier(device=device, context_reduction="retrieval", missing_indicators=True,
                                                 compile=False, verbose=False), seed

        def fit(self, Xa, ya):
            self.m.fit(Xa, ya)

        def predict_proba(self, Xa):
            return np.mean([self.m.predict_proba(Xa, context_size=context, batch_size=64, seed=self.seed + k) for k in range(n_ens)], axis=0)
    return icl_fit(lambda seed, ci: M(seed), n_num, 10 ** 9, chunk=2048, as_codes=True)


def realmlp_fit(n_ens=1, **kw):
    """pytabkit RealMLP-TD (tuned defaults). Checkpoint selection only on the inner stop split (X_val), then the
    selected network predicts the outer fold and test (no refit: RealMLP's schedule is tied to its epoch budget)."""
    from pytabkit import RealMLP_TD_Classifier

    def prep(X, rows):
        """Median fill fitted on `rows`, one indicator per distinct missingness pattern (thin-file blocks)."""
        nums = [c for c in X.columns if c not in CAT_COLS]
        med = X.iloc[rows][nums].median()
        miss = X.iloc[rows][nums].isna()
        groups, seen = [], set()
        for c in nums:
            key = np.packbits(miss[c].to_numpy()).tobytes()
            if miss[c].any() and key not in seen:
                seen.add(key)
                groups.append(c)

        def tf(d):
            out = d[nums].fillna(med).fillna(0.0)
            for c in groups:
                out[f"na_{c}"] = d[c].isna().astype(float)
            for c in CAT_COLS:
                out[c] = d[c]
            return out
        return tf

    def fit(X, y, tr, fi, st, rest, seed):
        tf = prep(X, tr[fi])
        X = tf(X)
        rest = [tf(r) for r in rest]
        m = RealMLP_TD_Classifier(device="cuda", random_state=seed, n_ens=n_ens, verbosity=0, val_metric_name="1-auc_ovr",
                                  **kw)
        m.fit(X.iloc[tr[fi]], y[tr[fi]], X_val=X.iloc[tr[st]], y_val=y[tr[st]], cat_col_names=CAT_COLS)
        return [m.predict_proba(r)[:, 1] for r in rest], 0
    return fit


def woe_lr_fit(n_bins=20, alpha=20.0, Cs=(0.003, 0.01, 0.03, 0.1, 0.3)):
    """Credit scorecard: per feature, bins fitted on the training rows (NaN and exact zero get their own bins, else
    quantile bins of the rest; categoricals by level), smoothed WOE = log odds of the bin shrunk to the prior with
    `alpha` pseudo-rows, then L2 logistic regression on the standardised WOE columns. C is picked on the inner stop
    split, then the LR is refitted on the whole training fold (binning refitted there too)."""
    from sklearn.linear_model import LogisticRegression

    def encoder(X, rows, yv):
        prior = yv.mean()
        lo = np.log(prior / (1 - prior))
        specs = []
        for c in X.columns:
            if c in CAT_COLS:
                codes = X[c].cat.codes.to_numpy()
                specs.append((c, "cat", None))
                b = codes[rows]
            else:
                x = X[c].to_numpy(dtype=float)[rows]
                nz = np.isfinite(x) & (x != 0)
                edges = np.unique(np.quantile(x[nz], np.linspace(0, 1, n_bins + 1))[1:-1]) if nz.sum() > n_bins else np.array([])
                specs.append((c, "num", edges))
                b = bins_of(x, edges)
            keys, inv = np.unique(b, return_inverse=True)
            pos = np.bincount(inv, weights=yv, minlength=len(keys))
            n = np.bincount(inv, minlength=len(keys))
            rate = (pos + alpha * prior) / (n + alpha)
            specs[-1] = specs[-1] + (dict(zip(keys.tolist(), (np.log(rate / (1 - rate)) - lo).tolist())),)
        return specs

    def bins_of(x, edges):
        out = np.where(np.isnan(x), -2, np.where(x == 0, -1, np.searchsorted(edges, x, side="right")))
        return out

    def transform(specs, D):
        cols = []
        for c, kind, edges, table in specs:
            b = D[c].cat.codes.to_numpy() if kind == "cat" else bins_of(D[c].to_numpy(dtype=float), edges)
            cols.append(np.array([table.get(int(k), 0.0) for k in b]) if len(table) < 50 else
                        np.vectorize(lambda k: table.get(int(k), 0.0))(b))
        return np.column_stack(cols)

    def fit_lr(A, yv, C, seed):
        mu, sd = A.mean(0), A.std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit((A - mu) / sd, yv)
        return lambda B: m.decision_function((B - mu) / sd)

    def fit(X, y, tr, fi, st, rest, seed):
        sp = encoder(X, tr[fi], y[tr[fi]])
        A, S = transform(sp, X.iloc[tr[fi]]), transform(sp, X.iloc[tr[st]])
        best = max(Cs, key=lambda C: ks(y[tr[st]], fit_lr(A, y[tr[fi]], C, seed)(S)))
        sp = encoder(X, tr, y[tr])
        f = fit_lr(transform(sp, X.iloc[tr]), y[tr], best, seed)
        return [1 / (1 + np.exp(-f(transform(sp, r)))) for r in rest], best
    return fit


def lr_q_fit(Cs=(0.001, 0.003, 0.01, 0.03, 0.1, 1.0), onehot=True):
    """Logistic regression on nn_prep inputs (mid-rank ECDF -> normal fitted on the training rows, NaN -> 0, one
    indicator per missingness pattern) + one-hot categoricals. C picked on the inner stop split, refit on the fold."""
    from sklearn.linear_model import LogisticRegression
    from nn_prep import nn_prep

    def mats(X, fit_rows, others):
        nums = [c for c in X.columns if c not in CAT_COLS]
        a, b, _ = nn_prep(X.iloc[fit_rows][nums].to_numpy(dtype=float),
                          np.vstack([o[nums].to_numpy(dtype=float) for o in others]))
        def oh(d):
            return np.column_stack([(d[c].cat.codes.to_numpy()[:, None] == np.arange(len(d[c].cat.categories))).astype(np.float32)
                                    for c in CAT_COLS])
        if onehot:
            a = np.hstack([a, oh(X.iloc[fit_rows])])
            b = np.hstack([b, np.vstack([oh(o) for o in others])])
        return a, np.split(b, np.cumsum([len(o) for o in others])[:-1])

    def fit(X, y, tr, fi, st, rest, seed):
        a, (s_,) = mats(X, tr[fi], [X.iloc[tr[st]]])
        best = max(Cs, key=lambda C: ks(y[tr[st]], LogisticRegression(C=C, max_iter=3000).fit(a, y[tr[fi]]).decision_function(s_)))
        a, outs = mats(X, tr, rest)
        m = LogisticRegression(C=best, max_iter=3000).fit(a, y[tr])
        return [m.predict_proba(o)[:, 1] for o in outs], best
    return fit


def run(name, fit, X, Xt, y, ids, test_ids, out: Path, seeds=SEEDS, folds=None, deadline=None):
    """Runs fit over seeds x outer folds. OOF = mean over seeds (NaN where no fold ran); test = mean over all fits.
    Writes out/oof_<name>.parquet, test_<name>.parquet and log_<name>.csv."""
    out.mkdir(parents=True, exist_ok=True)
    oof_sum, oof_n, test_sum, log = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(Xt)), []
    for seed in seeds:
        for k, (tr, va) in enumerate(outer(y, seed)):
            if folds is not None and k not in folds:
                continue
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"{name}: deadline before seed {seed} fold {k}")
            t0 = time.time()
            fi, st = inner(y[tr], seed, k)
            (pv, pt), budget = fit(X, y, tr, fi, st, [X.iloc[va], Xt], seed)
            oof_sum[va] += pv
            oof_n[va] += 1
            test_sum += pt
            log.append(dict(model=name, seed=seed, fold=k, ks=ks(y[va], pv), budget=budget, sec=round(time.time() - t0, 1)))
            print(log[-1], flush=True)
            pd.DataFrame(log).to_csv(out / f"log_{name}.csv", index=False)
    oof = np.where(oof_n > 0, oof_sum / np.maximum(oof_n, 1), np.nan)
    pl.DataFrame({"application_id": ids, "target": y, name: oof}).write_parquet(out / f"oof_{name}.parquet")
    pl.DataFrame({"application_id": test_ids, name: test_sum / len(log)}).write_parquet(out / f"test_{name}.parquet")
    m = oof_n > 0
    print(f"{name}: OOF KS {ks(y[m], oof[m]):.5f} on {m.sum()} rows", flush=True)
    return oof, test_sum / len(log), log


def manifest(out: Path, extra=None):
    import importlib.metadata as md
    pk = {}
    for n in ["numpy", "pandas", "polars", "scikit-learn", "scipy", "lightgbm", "xgboost", "catboost", "torch", "tabm",
              "rtdl_num_embeddings", "causilo"]:
        try:
            pk[n] = md.version(n)
        except md.PackageNotFoundError:
            pk[n] = None
    cfg = dict(N_FOLDS=N_FOLDS, INNER=INNER, REFIT=REFIT, SEEDS=SEEDS, N_ROUNDS=N_ROUNDS, EARLY_STOP=EARLY_STOP,
               LGB=LGB, LGB_S=LGB_S, XGB=XGB, XGB_S=XGB_S, CAT=CAT, CAT_S=CAT_S, TABM=TABM)
    (out / "manifest.json").write_text(json.dumps(dict(packages=pk, config=cfg, **(extra or {})), indent=2, default=str))
