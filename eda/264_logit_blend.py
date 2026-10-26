"""Equal-weight blend of within-cohort z-scored logits (probability families) / probit ranks (rank families).

Protocol: artifacts/optimization/continuation_25okt/protocol_264_logit_blend.json (locked before scores).
Usage: cd eda && ../.venv/bin/python 264_logit_blend.py
"""
import importlib
import json

import numpy as np
from scipy.special import logit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import ROOT

e250 = importlib.import_module("250_merge_blend_existing")
RANK_FAMILIES = {"D", "K64"}  # their stored vectors are already ranks in [0, 1]
CANDS = {"R2DK_logit": ["R2", "D", "K64"], "R2DK_KD2_logit": ["R2", "D", "K64", "KD2"]}


def score(name, x):
    if name in RANK_FAMILIES:
        return norm.ppf(e250.rank(x) - 0.5 / len(x))  # probit of mid-rank, finite at the ends
    return logit(np.clip(x, 1e-6, 1 - 1e-6))


def z(x):
    return (x - x.mean()) / x.std()


ids, y, tids, folds, fo, ft, _ = e250.families(str(ROOT / "results/dsc0152/agas/v66/output/preds_v66s42.npz"))
rng = np.random.default_rng(264)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(1000)]
rep = {}
for name, ms in CANDS.items():
    oof, base = np.zeros(len(y)), np.zeros(len(y))
    for _, va in folds:
        oof[va] = np.mean([z(score(m, fo[m][va])) for m in ms], axis=0)
        base[va] = np.mean([e250.rank(fo[m][va]) for m in ms], axis=0)
    d = e250.fast_ks(y, oof, boots) - e250.fast_ks(y, base, boots)
    rep[name] = {"pooled_ks": e250.ks(y, oof), "auc": float(roc_auc_score(y, oof)),
                 "fold_ks": [e250.ks(y[va], oof[va]) for _, va in folds],
                 "rank_version_ks": e250.ks(y, base), "delta_vs_rank": e250.ks(y, oof) - e250.ks(y, base),
                 "p_gt0_vs_rank": float((d > 0).mean())}
    print(name, {k: (round(v, 5) if isinstance(v, float) else [round(a, 4) for a in v]) for k, v in rep[name].items()})
(ROOT / "eda/out/264_logit_blend.json").write_text(json.dumps(rep, indent=1))
