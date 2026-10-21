"""Driver for validation_v2. Local screening:  python run_valid2.py OUT_DIR lgb,tabm --folds 0 --seeds 42
Kaggle full run: same call without --folds/--seeds. Extra feature parquet (application_id + columns) via --extra."""
import argparse
import time
from pathlib import Path

import polars as pl

import valid2 as v

MODELS = {
    "lgb": lambda: v.lgb_fit(v.LGB), "lgb_s": lambda: v.lgb_fit(v.LGB_S),
    "xgb": lambda: v.xgb_fit(v.XGB), "xgb_s": lambda: v.xgb_fit(v.XGB_S),
    "cat": lambda: v.cat_fit(v.CAT), "cat_s": lambda: v.cat_fit(v.CAT_S),
    "tabm": lambda: v.tabm_fit(),
    "causilo": lambda: v.causilo_fit(),
    "realmlp": lambda: v.realmlp_fit(),
    "woe_lr": lambda: v.woe_lr_fit(),
    "lr_q": lambda: v.lr_q_fit(),
}


def model(name):
    """'causilo_e16_f150' -> causilo_fit(n_est=16, n_num=150); other names from MODELS."""
    if name.startswith("seg_"):
        return v.seg_fit(model(name[4:]))
    base = name.split("_")[0]
    if base in ("causilo", "tabpfn", "tabdpt", "kumo"):
        parts = name.split("_")[1:]
        kw = {"size": parts.pop(0)} if base == "kumo" and parts and parts[0] in ("small", "medium", "large") else {}
        kw |= {dict(e="n_est", f="n_num", c="max_context", x="context", n="n_ens", b="n_bag")[t[0]]: int(t[1:]) for t in parts}
        return getattr(v, f"{base}_fit")(**kw)
    return MODELS[name]()


def main(features: pl.DataFrame, out: Path, names, folds=None, seeds=None, extra=None, drop=(), deadline=None):
    if extra is not None:
        features = features.join(extra, on="application_id", how="left", validate="1:1")
    features = features.drop([c for c in drop if c in features.columns])
    X, Xt, y, ids, tids = v.frames(features)
    out.mkdir(parents=True, exist_ok=True)
    v.manifest(out, dict(models=names, folds=folds, seeds=seeds or v.SEEDS, n_features=X.shape[1],
                         extra=None if extra is None else extra.columns, drop=list(drop)))
    res = {}
    for n in names:
        res[n] = v.run(n, model(n), X, Xt, y, ids, tids, out, seeds=seeds or v.SEEDS, folds=folds, deadline=deadline)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("models")
    ap.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--extra", help="parquet with application_id + new columns")
    ap.add_argument("--drop", default="", help="comma-separated feature columns to drop")
    ap.add_argument("--features", help="build_all parquet (default: data_pq/features.parquet via common.features)")
    ap.add_argument("--hours", type=float, help="deadline in hours from start")
    a = ap.parse_args()
    t0 = time.time()
    if a.features:
        feats = pl.read_parquet(a.features)
    else:
        from common import features
        feats = features()
    main(feats, Path(a.out), a.models.split(","), a.folds, a.seeds,
         pl.read_parquet(a.extra) if a.extra else None, [c for c in a.drop.split(",") if c],
         t0 + a.hours * 3600 if a.hours else None)
    print(f"done in {time.time() - t0:.0f}s")
