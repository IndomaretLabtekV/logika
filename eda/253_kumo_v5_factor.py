"""Cross-pipeline feature fusion: Kumo V5 (251 K5 recipe) + DSC0152 factor-model features, seed-42 folds.

Protocol: artifacts/optimization/continuation_25okt/protocol_253_kumo_v5_factor.json (locked before scores).
Stages: factor (CPU, all folds: label-free inputs once + loadings from outer-train labels per fold),
        kumo --fold k (GPU), eval (pooled OOF of KF; blends are scored by 250 with protocol 253).
Usage: cd eda && ../.venv/bin/python 253_kumo_v5_factor.py --stage {factor,kumo,eval} [--fold k]
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

t251 = importlib.import_module("251_teammate_recipe_s42")
OUT, DEF = t251.OUT, t251.DEF
DSC = ROOT / "scripts/dsc0152"
FAC = ["factor_z", "factor_sd", "fs_price", "fs_late", "fs_ldpd", "fs_bu_z"]


def stage_factor():
    sys.path.insert(0, str(DSC))
    cwd = os.getcwd()
    os.chdir(DSC)  # features.py reads data/ relative to its folder
    import features as ft

    app = ft.load_apps()
    M, E = ft.factor_inputs(app)
    os.chdir(cwd)
    lab = t251.labels()
    folds = v.outer(lab.target.to_numpy(), 42)
    pos = pd.Series(np.arange(len(app)), index=app.application_id)
    y = app.target.to_numpy()  # NaN for test rows; only outer-train rows are passed below
    for f, (tr, _) in enumerate(folds):
        rows = pos.loc[lab.application_id.iloc[tr]].to_numpy()
        assert not np.isnan(y[rows]).any()
        out = ft.factor_features(M, E, y, rows)
        assert set(out.columns) == set(FAC), out.columns
        out.insert(0, "application_id", app.application_id.to_numpy())
        t251.atomic_parquet(out, OUT / f"f{f}/factor.parquet")
        print("factor", f, flush=True)


def stage_kumo(fold):
    import strict_kumo_engine as eng

    frame, fit_ids, val_ids = t251.fold_frame(fold)
    frame = frame.merge(pd.read_parquet(OUT / f"f{fold}/factor.parquet"), on="application_id", how="left",
                        validate="one_to_one")
    y, ti, ci, hi, vi, te = t251.split_rows(frame, fit_ids, val_ids, fold)
    columns = json.loads(DEF.read_text())["context_columns"] + FAC
    query = np.r_[vi, te]
    p, meta = eng.predict(frame, ci, query, y, columns, 2026 + fold, n_estimators=8, column_cap=500, chunk=128,
                          memory_fraction=.70)
    part = np.r_[["val"] * len(vi), ["test"] * len(te)]
    t251.atomic_parquet(pd.DataFrame({"application_id": frame.application_id.iloc[query].to_numpy(), "part": part,
                                      "KF": p}), OUT / f"f{fold}/kumo_fac.parquet")
    meta.pop("context_rows", None)
    t251.atomic_json(meta, OUT / f"f{fold}/kumo_fac_meta.json")
    print("kumo_fac", fold, "peak GiB", meta.get("max_allocated_gib"), flush=True)


def stage_eval():
    lab = t251.labels()
    y = lab.target.to_numpy()
    oof, fks = np.full(len(y), np.nan), []
    for f, (_, va) in enumerate(v.outer(y, 42)):
        p = pd.read_parquet(OUT / f"f{f}/kumo_fac.parquet").query("part == 'val'").set_index("application_id")
        oof[va] = p.loc[lab.application_id.iloc[va], "KF"].to_numpy()
        fks.append(v.ks(y[va], oof[va]))
    rep = {"pooled_ks": v.ks(y, oof), "auc": float(roc_auc_score(y, oof)), "fold_ks": fks}
    print(f"KF pooled {rep['pooled_ks']:.5f} auc {rep['auc']:.5f}", [round(x, 4) for x in fks], flush=True)
    t251.atomic_parquet(pd.DataFrame({"application_id": lab.application_id, "target": y, "KF": oof}),
                        OUT / "oof_kf.parquet")
    test = pd.concat([pd.read_parquet(OUT / f"f{f}/kumo_fac.parquet").query("part == 'test'") for f in range(5)])
    t251.atomic_parquet(test.groupby("application_id")[["KF"]].mean().reset_index(), OUT / "test_kf.parquet")
    t251.atomic_json(rep, ROOT / "eda/out/253_kumo_v5_factor.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["factor", "kumo", "eval"])
    ap.add_argument("--fold", type=int)
    a = ap.parse_args()
    stage_kumo(a.fold) if a.stage == "kumo" else stage_factor() if a.stage == "factor" else stage_eval()
