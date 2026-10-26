"""T5: LightGBM on teammate V5 (+re_*) features per outer fold, valid2-style inner stop + refit, seed-42 folds.

Protocol: artifacts/optimization/continuation_25okt/protocol_262_lgb_v5.json (locked before scores).
Usage: cd eda && ../.venv/bin/python 262_lgb_v5.py --stage {fit --fold k, eval}
"""
import argparse
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import valid2 as v
from common import ROOT

t251 = importlib.import_module("251_teammate_recipe_s42")
OUT = t251.OUT
DROP = {"application_id", "customer_id", "target", "split", "application_datetime", "decision_datetime", "decision_date"}
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=31, min_child_samples=100, feature_fraction=0.3,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, num_threads=4, verbose=-1,
              deterministic=True, force_row_wise=True)


def stage_fit(fold):
    frame, fit_ids, val_ids = t251.fold_frame(fold)  # V5 + re_* (reverse.parquet)
    lab = t251.labels().set_index("application_id").target
    cols = [c for c in frame.columns if c not in DROP]
    x = frame[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype("category")  # categories from all rows' strings: label-free vocabulary
    fit = np.flatnonzero(frame.application_id.isin(fit_ids).to_numpy())
    val = np.flatnonzero(frame.application_id.isin(val_ids).to_numpy())
    test = np.setdiff1d(np.arange(len(frame)), np.r_[fit, val])
    y = lab.loc[frame.application_id.iloc[fit]].to_numpy()
    fi, st = next(StratifiedKFold(7, shuffle=True, random_state=1000 * 42 + fold).split(np.zeros(len(fit)), y))
    pv, pt, rounds = np.zeros(len(val)), np.zeros(len(test)), []
    for seed in (0, 1):
        p = {**PARAMS, "seed": seed}
        m = lgb.train(p, lgb.Dataset(x.iloc[fit[fi]], y[fi]), 10000, valid_sets=[lgb.Dataset(x.iloc[fit[st]], y[st])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        n = max(1, int(round(m.best_iteration * 1.15)))
        m = lgb.train(p, lgb.Dataset(x.iloc[fit], y), n)
        pv += m.predict(x.iloc[val]) / 2
        pt += m.predict(x.iloc[test]) / 2
        rounds.append(n)
    part = np.r_[["val"] * len(val), ["test"] * len(test)]
    t251.atomic_parquet(pd.DataFrame({"application_id": frame.application_id.iloc[np.r_[val, test]].to_numpy(),
                                      "part": part, "T5": np.r_[pv, pt]}), OUT / f"f{fold}/lgb_v5.parquet")
    yv = lab.loc[frame.application_id.iloc[val]].to_numpy()
    print("lgb_v5", fold, "cols", len(cols), "rounds", rounds, "fold KS", round(v.ks(yv, pv), 5), flush=True)


def stage_eval():
    lab = t251.labels()
    y = lab.target.to_numpy()
    oof, fks = np.full(len(y), np.nan), []
    for f, (_, va) in enumerate(v.outer(y, 42)):
        p = pd.read_parquet(OUT / f"f{f}/lgb_v5.parquet").query("part == 'val'").set_index("application_id")
        oof[va] = p.loc[lab.application_id.iloc[va], "T5"].to_numpy()
        fks.append(v.ks(y[va], oof[va]))
    rep = {"pooled_ks": v.ks(y, oof), "auc": float(roc_auc_score(y, oof)), "fold_ks": fks}
    print(f"T5 pooled {rep['pooled_ks']:.5f} auc {rep['auc']:.5f}", [round(x, 4) for x in fks], flush=True)
    t251.atomic_parquet(pd.DataFrame({"application_id": lab.application_id, "target": y, "T5": oof}), OUT / "oof_t5.parquet")
    test = pd.concat([pd.read_parquet(OUT / f"f{f}/lgb_v5.parquet").query("part == 'test'") for f in range(5)])
    t251.atomic_parquet(test.groupby("application_id")[["T5"]].mean().reset_index(), OUT / "test_t5.parquet")
    t251.atomic_json(rep, ROOT / "eda/out/262_lgb_v5.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "eval"])
    ap.add_argument("--fold", type=int)
    a = ap.parse_args()
    stage_fit(a.fold) if a.stage == "fit" else stage_eval()
