"""Bet 10: auxiliary supervision from repayment behaviour.

The trees learn from 70k binary targets. Repayment history is a far denser signal (1.36M installments of ~70k
customers, train and test), and it is a feature table, not the competition label. An auxiliary LightGBM learns to
predict each customer's repayment latent from the columns that the no-history 30% also have (application, bureau,
e-commerce, prices). Its prediction gives the no-history segment a repayment-risk estimate learned from everyone
else, and gives customers with history a prior next to their own (short) record.

Aux targets (all label-free): oe_theta (f266, observed minus expected lateness), mk_hit12 (f251), any 90+ installment.
Cross-fitted: customers with history are predicted by the fold model that did not see them; no-history customers by
the mean of the 5 fold models. The competition target is never used.
Usage: cd eda && ../.venv/bin/python 277_aux_latent.py [threads]
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import KFold

import common as c
import valid2 as v


def main():
    th = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    f = c.features().sort("application_id")
    aux = (f.select("application_id")
           .join(pl.read_parquet(c.PQ / "f266_expected_late.parquet").select("application_id", "oe_theta"), on="application_id", how="left", maintain_order="left")
           .join(pl.read_parquet(c.PQ / "f251_repayment_markov.parquet").select("application_id", "mk_hit12"), on="application_id", how="left", maintain_order="left")
           .join(pl.read_parquet(c.PQ / "f271_spec_block.parquet").select("application_id", "ll_n90"), on="application_id", how="left", maintain_order="left"))
    nohist = f.filter(pl.col("has_history") == 0)
    keep = [k for k in f.columns if k not in v.ID_COLS and k != "has_history"
            and nohist[k].is_not_null().mean() > 0.5]
    print(f"{len(keep)} aux input columns (present for no-history customers)")
    X = f.select(keep).to_pandas()
    for k in keep:
        if k in v.CAT_COLS or f[k].dtype == pl.String:
            X[k] = X[k].astype("category")
        else:
            X[k] = X[k].astype(float).replace([np.inf, -np.inf], np.nan)
    hist = (f["has_history"] == 1).to_numpy()
    targets = {"aux_theta": aux["oe_theta"].to_numpy(), "aux_hit12": aux["mk_hit12"].to_numpy(),
               "aux_90": (aux["ll_n90"].fill_null(0) > 0).cast(pl.Float64).to_numpy()}
    out = {}
    idx = np.where(hist)[0]
    for name, t in targets.items():
        ok = idx[np.isfinite(t[idx])]
        pred = np.full(len(t), np.nan)
        rest = np.zeros(len(t))
        obj = "binary" if name == "aux_90" else "regression"
        p = dict(objective=obj, learning_rate=0.03, num_leaves=63, min_child_samples=100, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=th)
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=277).split(ok)):
            fi, st = a[: int(len(a) * 0.85)], a[int(len(a) * 0.85):]
            m = lgb.train(p, lgb.Dataset(X.iloc[ok[fi]], t[ok[fi]]), 5000,
                          valid_sets=[lgb.Dataset(X.iloc[ok[st]], t[ok[st]])], callbacks=[lgb.early_stopping(200, verbose=False)])
            pred[ok[b]] = m.predict(X.iloc[ok[b]])
            rest += m.predict(X) / 5
            print(f"   {name} fold {k}: {m.best_iteration} trees", flush=True)
        miss = ~np.isfinite(pred)
        pred[miss] = rest[miss]
        r = np.corrcoef(pred[ok], t[ok])[0, 1]
        print(f"{name}: cross-fitted corr with its own target {r:.4f} on {len(ok)} customers")
        out[name] = pred
        if name != "aux_90":
            out[f"{name}_surprise"] = np.where(np.isfinite(t), t - pred, np.nan)
    o = f.select("application_id").with_columns(**{k: pl.Series(val) for k, val in out.items()})
    o.write_parquet(c.PQ / "f277_aux_latent.parquet")
    tr = f.select("application_id", "target", "has_history", "is_train").join(o, on="application_id", maintain_order="left").filter(pl.col("is_train") == 1)
    for g in (0, 1):
        q = tr.filter(pl.col("has_history") == g)
        y = q["target"].to_numpy()
        print(f"has_history={g}: " + "  ".join(f"{k} KS {c.ks(y, np.nan_to_num(q[k].to_numpy(), nan=0)):.4f}" for k in out))


if __name__ == "__main__":
    main()
