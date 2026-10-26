"""Cross-fitted blend weights over the merge families (protocol_255_crossfit_weights.json, locked before KF/KD scores).

Fold k gets the simplex weights (step 0.1, D >= 0.2) that maximise pooled AUC of per-cohort ranks on the other 4 folds.
Test weights come from the same search on all 5 folds. Reports pooled KS/AUC, weights, bootstrap vs RDK and REF.
Usage: cd eda && ../.venv/bin/python 255_crossfit_weights.py [--families R,D,K64,KF,KD]
"""
import importlib
import itertools
import json
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from common import ROOT

e250 = importlib.import_module("250_merge_blend_existing")
rank, ks, fast_ks = e250.rank, e250.ks, e250.fast_ks
D_NPZ = ROOT / "results/dsc0152/agas/v66/output/preds_v66s42.npz"


def simplex(n, step=10):
    for c in itertools.product(range(step + 1), repeat=n - 1):
        if sum(c) <= step:
            yield np.array(list(c) + [step - sum(c)]) / step


CRIT = "auc"  # --criterion ks (protocol 259)


def search(ranks, y, rows, names):
    """Best weights on `rows` by AUC (or KS); ties -> lower max weight then lexicographic (grid order)."""
    best = None
    for w in simplex(len(names)):
        if w[names.index("D")] < 0.2:
            continue
        p = ranks[rows] @ w
        auc = roc_auc_score(y[rows], p) if CRIT == "auc" else ks(y[rows], p)
        key = (round(auc, 7), -w.max())
        if best is None or key > best[0]:
            best = (key, w)
    return best[1]


def main():
    global CRIT
    args = sys.argv[1:]
    CRIT = args[args.index("--criterion") + 1] if "--criterion" in args else "auc"
    want = args[args.index("--families") + 1].split(",") if "--families" in args else ["R", "D", "K64", "KF", "KD"]
    ids, y, tids, folds, fam_oof, fam_test, _ = e250.families(str(D_NPZ))
    names = [m for m in want if m in fam_oof]
    ranks = np.zeros((len(y), len(names)))
    for _, va in folds:  # per-cohort ranks, as everywhere in the merge
        for j, m in enumerate(names):
            ranks[va, j] = rank(fam_oof[m][va])
    oof, chosen = np.zeros(len(y)), []
    for k, (_, va) in enumerate(folds):
        other = np.concatenate([folds[j][1] for j in range(5) if j != k])
        w = search(ranks, y, other, names)
        chosen.append(w.tolist())
        oof[va] = rank(ranks[va] @ w)
        print(f"fold {k} weights", dict(zip(names, w.round(2))), f"fold KS {ks(y[va], oof[va]):.4f}", flush=True)
    w_all = search(ranks, y, np.arange(len(y)), names)
    test = np.column_stack([rank(fam_test[m]) for m in names]) @ w_all
    rdk = np.zeros(len(y))
    ref = np.zeros(len(y))
    base = ["R2" if "R2" in names else "R", "D", "K64"]
    for _, va in folds:
        rdk[va] = np.mean([rank(fam_oof[m][va]) for m in base], axis=0)
        ref[va] = np.mean([rank(fam_oof[m][va]) for m in ["T", "C", "KL"]], axis=0)
    rng = np.random.default_rng(255)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(1000)]
    b = {n: fast_ks(y, p, boots) for n, p in {"cf": oof, "RDK": rdk, "REF": ref}.items()}
    rep = {"families": names, "pooled_ks": ks(y, oof), "auc": float(roc_auc_score(y, oof)),
           "fold_ks": [ks(y[va], oof[va]) for _, va in folds], "fold_weights": chosen,
           "test_weights": dict(zip(names, w_all.tolist()))}
    for base in ["RDK", "REF"]:
        d = b["cf"] - b[base]
        rep[f"delta_vs_{base}"] = {"observed": rep["pooled_ks"] - ks(y, {"RDK": rdk, "REF": ref}[base]),
                                   "p_gt0": float((d > 0).mean()), "ci95": np.quantile(d, [.025, .975]).tolist()}
    print(json.dumps({k: v for k, v in rep.items() if k != "fold_weights"}, indent=1, default=float))
    tag = "_".join(names) + ("" if CRIT == "auc" else "_ks")
    (ROOT / f"eda/out/255_crossfit_{tag}.json").write_text(json.dumps(rep, indent=2))
    sub = pl.DataFrame({"application_id": tids, "default_probability": rank(test)})
    assert sub.height == 30000 and sub["application_id"].n_unique() == 30000
    folder = ROOT / "results/v71/merge_blend/crossfit"
    folder.mkdir(parents=True, exist_ok=True)
    sub.write_csv(folder / f"{tag}.csv")
    pl.DataFrame({"application_id": ids, "target": y, "cf": oof}).write_parquet(folder / f"oof_{tag}.parquet")


if __name__ == "__main__":
    main()
