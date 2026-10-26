"""Does the GRU-NN (242) add on top of our best blends? Locked before computing: B1 = 0.8 rank(REF v59-analogue)
+ 0.2 rank(NN), B2 = 0.8 rank(K64 2-seed) + 0.2 rank(NN); per outer cohort ranks; paired stratified bootstrap."""
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import valid2 as v
from common import ROOT

R = ROOT / "results"
train = pl.read_csv(ROOT / "data/train_applications.csv").sort("application_id")
ids, y = train["application_id"].to_numpy(), train["target"].to_numpy().astype(int)
folds = v.outer(y, 42)


def col(path, name=None):
    d = pl.read_parquet(path).sort("application_id")
    assert d["application_id"].to_list() == ids.tolist()
    return d[name or [c for c in d.columns if c not in ("application_id", "target")][0]].to_numpy()


def rank(x):
    return rankdata(x) / len(x)


nn_ = col(R / "v68/gru_nn/oof_gru_nn.parquet")
k64 = col(R / "v60/candidates/K__K64_2seed/oof.parquet")
trees = np.mean([rank(col(R / f"v22/v22/gbdt/oof_{m}.parquet")) for m in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]], axis=0)
cau = sum(w * col(p) for w, p in [(1, R / "v22/v22/fm/oof_causilo.parquet"), (2, R / "v37/oof_causilo.parquet"), (2, R / "v41/causilo_s12/oof_causilo.parquet")]) / 5
kum = col(R / "v28m/oof_kumo_large_c56000.parquet")
ref, b1, b2 = (np.empty(len(y)) for _ in range(3))
for _, va in folds:
    ref[va] = (rank(trees[va]) + rank(cau[va]) + rank(kum[va])) / 3
    b1[va] = .8 * rank(ref[va]) + .2 * rank(nn_[va])
    b2[va] = .8 * rank(k64[va]) + .2 * rank(nn_[va])
rng = np.random.default_rng(243)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
boots = [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(1000)]
for name, cand, base in [("B1 vs REF", b1, ref), ("B2 vs K64", b2, k64), ("B2 vs REF", b2, ref)]:
    d = np.array([v.ks(y[r], cand[r]) - v.ks(y[r], base[r]) for r in boots])
    fw = sum(v.ks(y[va], cand[va]) > v.ks(y[va], base[va]) for _, va in folds)
    print(f"{name}: {v.ks(y, cand):.5f} vs {v.ks(y, base):.5f} delta {v.ks(y, cand) - v.ks(y, base):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {(d > 0).mean():.3f} folds {fw}/5 AUC {roc_auc_score(y, cand):.5f}")
print("fold KS B1", [round(v.ks(y[va], b1[va]), 4) for _, va in folds], "B2", [round(v.ks(y[va], b2[va]), 4) for _, va in folds])
