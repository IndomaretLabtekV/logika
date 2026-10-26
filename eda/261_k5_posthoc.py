"""POST-HOC exploration (not locked): raw Kumo V5 (K5, 2 seeds, no offset/spline) in place of R2, after seeing that the
re_* offset did not replicate on seed 42. Optimistic by selection; reported only as exploration.
Usage: cd eda && ../.venv/bin/python 261_k5_posthoc.py"""
import importlib
import json

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT

e250 = importlib.import_module("250_merge_blend_existing")
ids, y, tids, folds, fo, ft, _ = e250.families(str(ROOT / "results/dsc0152/agas/v66/output/preds_v66s42.npz"))
d = ROOT / "results/v71/teammate_s42"
k5 = [pl.read_parquet(d / f"oof{s}.parquet") for s in ("", "_s2027")]
fo["K5x2"] = np.mean([e250.by_id(t["K5"].to_numpy(), t["application_id"].to_numpy(), ids) for t in k5], axis=0)
out = {}
for name, ms in {"K5x2": ["K5x2"], "K5x2_D": ["K5x2", "D"], "K5x2_DK": ["K5x2", "D", "K64"], "R2DK": ["R2", "D", "K64"]}.items():
    p = np.zeros(len(y))
    for _, va in folds:
        p[va] = np.mean([e250.rank(fo[m][va]) for m in ms], axis=0)
    out[name] = {"pooled_ks": e250.ks(y, p), "auc": float(roc_auc_score(y, p))}
    print(name, round(out[name]["pooled_ks"], 5), round(out[name]["auc"], 5))
(ROOT / "eda/out/261_k5_posthoc.json").write_text(json.dumps(out, indent=1))
