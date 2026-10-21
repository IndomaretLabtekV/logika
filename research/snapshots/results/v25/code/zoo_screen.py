"""TabArena model zoo on the validation_v2 fold-0 screen (seed 42, outer fold 0). Runs in .venv-zoo (tabarena +
autogluon). Each model is fitted by AutoGluon on the inner-fit rows with tuning_data = the inner 1/7 stop rows (any
early stopping / checkpointing sees only those), default TabArena hyperparameters, no bagging, no weighted ensemble.
The outer fold is only predicted. Output: results/screen/zoo_f0/oof_<name>.parquet (NaN outside fold 0), log.csv.
Usage: .venv-zoo/bin/python eda/zoo_screen.py LimiX TabFM Mitra ... [--minutes 40]"""
import argparse
import importlib
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import valid2 as v  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "screen" / "zoo_f0"
TA = {n: "tabarena.models" for n in ["ChimeraBoostModel", "EXAONETabularModel", "ILTMModel", "KNNNewModel", "LimiXModel",
                                     "ModernNCAModel", "NoriModel", "Nori30MModel", "OrionMSPModel",
                                     "PerpetualBoosterModel", "RealMLPModel", "RealTabPFNv25Model", "TabPFNv26Model",
                                     "SAPRPTOSSModel", "TabDPTModel", "TabDPTTurboModel", "TabFMModel", "TabICLModel",
                                     "TabICLv2Model", "TabMModel", "TabPFN3Model", "TabPFNWideModel", "TabSTARModel",
                                     "TabSwiftModel", "XRFMModel"]}
AG = {"MitraModel": "autogluon.tabular.models", "EBMModel": "autogluon.tabular.models",
      "NNFastAiTabularModel": "autogluon.tabular.models", "RFModel": "autogluon.tabular.models",
      "XTModel": "autogluon.tabular.models", "LinearModel": "autogluon.tabular.models",
      "TabularNeuralNetTorchModel": "autogluon.tabular.models", "CatBoostModel": "autogluon.tabular.models",
      "LGBModel": "autogluon.tabular.models", "XGBoostModel": "autogluon.tabular.models"}


def cls(name):
    mod = {**TA, **AG}[name]
    return getattr(importlib.import_module(mod), name)


def main(names, minutes, features=None, out=None, max_train=None, hp=None):
    from autogluon.tabular import TabularPredictor
    global OUT
    OUT = Path(out) if out else OUT
    X, Xt, y, ids, tids = v.frames(pl.read_parquet(features or ROOT / "data_pq" / "features.parquet"))
    tr, va = v.outer(y, 42)[0]
    fi, st = v.inner(y[tr], 42, 0)
    if max_train and len(fi) > max_train:
        # shortcut: stratified subsample of the inner-fit rows for models that cannot hold the full context
        rng = np.random.default_rng(0)
        keep = np.sort(np.concatenate([rng.choice(fi[y[tr[fi]] == c], int(max_train * (y[tr[fi]] == c).mean()), replace=False)
                                       for c in (0, 1)]))
        fi = keep
    df = X.assign(target=y)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "log.csv"
    for name in names:
        t0 = time.time()
        path = Path("/tmp") / f"ag_zoo_{name}"
        row = dict(model=name, n_fit=len(fi))
        try:
            pred = TabularPredictor(label="target", eval_metric="roc_auc", path=str(path), verbosity=1).fit(
                train_data=df.iloc[tr[fi]], tuning_data=df.iloc[tr[st]], hyperparameters={cls(name): dict(hp or {})},
                num_gpus=1, fit_weighted_ensemble=False, time_limit=minutes * 60, raise_on_no_models_fitted=True)
            p = pred.predict_proba(X.iloc[va])[1].to_numpy()
            oof = np.full(len(y), np.nan)
            oof[va] = p
            pl.DataFrame({"application_id": ids, "target": y, name: oof}).write_parquet(OUT / f"oof_{name}.parquet")
            row.update(ks=v.ks(y[va], p), status="ok")
        except Exception as e:  # noqa: BLE001 - a failing model is a result to log, not a reason to stop the zoo
            traceback.print_exc()
            row.update(ks=np.nan, status=f"{type(e).__name__}: {str(e)[:160]}")
        row["sec"] = round(time.time() - t0)
        print(row, flush=True)
        old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])]).to_csv(log_path, index=False)
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--features")
    ap.add_argument("--out")
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--hp", default="{}", help="JSON hyperparameters for every model")
    a = ap.parse_args()
    import json
    main(a.names, a.minutes, a.features, a.out, a.max_train, json.loads(a.hp))
