"""Third Kumo view: DSC0152 feature engineering (598 cols + per-fold factor features), K64 cap300@28k view, seed 42 folds.

Protocol: artifacts/optimization/continuation_25okt/protocol_254_kumo_dsc.json (locked before scores).
Usage: cd eda && ../.venv/bin/python 254_kumo_dsc_features.py --stage {kumo --fold k, eval}
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

t251 = importlib.import_module("251_teammate_recipe_s42")
OUT = t251.OUT
CACHE = ROOT / "scripts/dsc0152/agas/shared/cache"
SEED = 2026  # --seed 2027 = second seed (protocol 257)


def sfx():
    return "" if SEED == 2026 else f"_s{SEED}"


def dsc_frame():
    """features_cache + extra_cache exactly as train_submission_v66.load_frames, all 100k rows."""
    f = pd.read_pickle(CACHE / "features_cache.pkl")
    ex = pd.read_pickle(CACHE / "extra_cache.pkl")
    add = [c for c in ex.columns if c != "application_id" and c not in f.columns]
    f = f.merge(ex[["application_id"] + add], on="application_id", how="left")
    cols = [c for c in f.columns if c not in ("application_id", "customer_id", "target")]
    assert len(cols) == 598, len(cols)
    return f.drop(columns=["target"]), cols


def stage_kumo(fold):
    import strict_kumo_engine as eng

    frame, cols = dsc_frame()
    fac = pd.read_parquet(OUT / f"f{fold}/factor.parquet")
    frame = frame.merge(fac, on="application_id", how="left", validate="one_to_one")
    lab = t251.labels()
    tr, va = v.outer(lab.target.to_numpy(), 42)[fold]
    pos = pd.Series(np.arange(len(frame)), index=frame.application_id)
    fit = pos.loc[lab.application_id.iloc[tr]].to_numpy()
    val = pos.loc[lab.application_id.iloc[va]].to_numpy()
    test = np.setdiff1d(np.arange(len(frame)), np.r_[fit, val])
    assert len(test) == 30000
    y = np.zeros(len(frame), dtype=int)
    y[fit] = lab.target.iloc[tr].to_numpy()
    for c in frame.columns:  # engine accepts numeric/object/category only
        if frame[c].dtype == bool:
            frame[c] = frame[c].astype(float)
    columns = cols + list(fac.columns[1:])
    query = np.r_[val, test]
    p, meta = eng.predict(frame, fit, query, y, columns, SEED + fold, n_estimators=8, max_context=28672,
                          column_cap=300, chunk=128, memory_fraction=.85)
    part = np.r_[["val"] * len(val), ["test"] * len(test)]
    t251.atomic_parquet(pd.DataFrame({"application_id": frame.application_id.iloc[query].to_numpy(), "part": part,
                                      "KD": p}), OUT / f"f{fold}/kumo_dsc{sfx()}.parquet")
    meta.pop("context_rows", None)
    t251.atomic_json(meta, OUT / f"f{fold}/kumo_dsc{sfx()}_meta.json")
    print("kumo_dsc", fold, "peak GiB", meta.get("max_allocated_gib"), flush=True)


def stage_eval():
    lab = t251.labels()
    y = lab.target.to_numpy()
    oof, fks = np.full(len(y), np.nan), []
    for f, (_, va) in enumerate(v.outer(y, 42)):
        p = pd.read_parquet(OUT / f"f{f}/kumo_dsc{sfx()}.parquet").query("part == 'val'").set_index("application_id")
        oof[va] = p.loc[lab.application_id.iloc[va], "KD"].to_numpy()
        fks.append(v.ks(y[va], oof[va]))
    rep = {"pooled_ks": v.ks(y, oof), "auc": float(roc_auc_score(y, oof)), "fold_ks": fks}
    print(f"KD pooled {rep['pooled_ks']:.5f} auc {rep['auc']:.5f}", [round(x, 4) for x in fks], flush=True)
    t251.atomic_parquet(pd.DataFrame({"application_id": lab.application_id, "target": y, "KD": oof}),
                        OUT / f"oof_kd{sfx()}.parquet")
    test = pd.concat([pd.read_parquet(OUT / f"f{f}/kumo_dsc{sfx()}.parquet").query("part == 'test'") for f in range(5)])
    t251.atomic_parquet(test.groupby("application_id")[["KD"]].mean().reset_index(), OUT / f"test_kd{sfx()}.parquet")
    t251.atomic_json(rep, ROOT / f"eda/out/254_kumo_dsc{sfx()}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["kumo", "eval"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    SEED = a.seed
    stage_kumo(a.fold) if a.stage == "kumo" else stage_eval()
