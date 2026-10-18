"""validation_v2 blending on clean OOF: rank-average with non-negative weights. Weights are chosen on 4 outer folds
(seed-42 assignment) and applied to the 5th, so every scored row was never used to pick its weights. Reference =
the v8 blend recipe (weights reconstructed from the v8 submission, _aug folded into the base model, TabPFN dropped)
applied to the clean models. Paired bootstrap of the pooled cross-fitted KS difference vs the reference.
Usage: python blend2.py results/v22/v22 [more dirs ...]"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v

V8 = dict(lgb=.140, lgb_s=.056, xgb=.108, xgb_s=.084, cat=.037, cat_s=.042, tabm=.352, causilo=.137)


def load(dirs):
    oof, test = None, None
    for d in dirs:
        for p in sorted(Path(d).rglob("oof_*.parquet")):
            o = pl.read_parquet(p)
            t = pl.read_parquet(p.with_name(p.name.replace("oof_", "test_")))
            oof = o if oof is None else oof.join(o.drop("target"), on="application_id", validate="1:1")
            test = t if test is None else test.join(t, on="application_id", validate="1:1")
    return oof.sort("application_id"), test.sort("application_id")


def ranks(df, names):
    return np.column_stack([rankdata(df[n].to_numpy()) / df.height for n in names])


def greedy(R, y, iters=40, bags=10, seed=0):
    """Bagged Caruana selection with replacement -> non-negative weights summing to 1."""
    rng = np.random.default_rng(seed)
    W = np.zeros(R.shape[1])
    for _ in range(bags):
        rows = rng.choice(len(y), len(y), replace=True)
        Rb, yb = R[rows], y[rows]
        cnt, cur = np.zeros(R.shape[1]), np.zeros(len(yb))
        for it in range(iters):
            best = max(range(R.shape[1]), key=lambda j: v.ks(yb, (cur * it + Rb[:, j]) / (it + 1)))
            cnt[best] += 1
            cur = (cur * it + Rb[:, best]) / (it + 1)
        W += cnt / cnt.sum()
    return W / bags


def crossfit(R, y, folds, rule):
    s = np.zeros(len(y))
    ws = []
    for tr, va in folds:
        w = rule(R[tr], y[tr])
        s[va] = R[va] @ w
        ws.append(w)
    return s, np.mean(ws, axis=0)


def boot(y, a, b, n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        d.append(v.ks(y[i], a[i]) - v.ks(y[i], b[i]))
    d = np.array(d)
    return d.mean(), d.std(), (d > 0).mean()


def fold_ks(y, s, folds):
    return np.array([v.ks(y[va], s[va]) for _, va in folds])


def main(dirs):
    oof, test = load(dirs)
    y = oof["target"].to_numpy()
    names = [c for c in oof.columns if c not in ("application_id", "target")]
    R, Rt = ranks(oof, names), ranks(test, names)
    folds = v.outer(y, 42)
    print("single models:", {n: round(v.ks(y, R[:, j]), 5) for j, n in enumerate(names)})
    print("rank corr:\n", np.round(np.corrcoef(R.T), 3), names)
    trees = [n for n in names if n.split("_")[0] in ("lgb", "xgb", "cat")]
    fixed = lambda w: (lambda Rtr, ytr: np.array([w.get(n, 0) for n in names]) / sum(w.get(n, 0) for n in names))  # noqa: E731
    rules = {
        "v8_recipe": fixed(V8),
        "equal_trees": fixed({n: 1 for n in trees}),
        "equal_all": fixed({n: 1 for n in names}),
        "trees_causilo_50": fixed({**{n: .5 / len(trees) for n in trees}, "causilo": .5}),
        "greedy_bagged": lambda Rtr, ytr: greedy(Rtr, ytr),
    }
    kumo = [n for n in names if n.startswith("kumo")]
    if kumo:
        k = kumo[0]
        rules |= {
            "kumo_causilo_50": fixed({k: .5, "causilo": .5}),
            "trees_causilo_kumo": fixed({**{n: 1 / 3 / len(trees) for n in trees}, "causilo": 1 / 3, k: 1 / 3}),
            "v8_causilo_to_kumo": fixed({**V8, "causilo": V8["causilo"] / 2, k: V8["causilo"] / 2}),
            "v8_plus_kumo_25": fixed({**{n: x * .75 for n, x in V8.items()}, k: .25}),
        }
    res = {}
    for k, rule in rules.items():
        s, w = crossfit(R, y, folds, rule)
        res[k] = (s, w)
    ref = res["v8_recipe"][0]
    print(f"\n{'blend':18s} {'cf_KS':>8s} {'dKS_vs_v8':>10s} {'sd':>7s} {'P>0':>5s} folds_won  weights")
    for k, (s, w) in res.items():
        m, sd, p = boot(y, s, ref)
        won = int((fold_ks(y, s, folds) > fold_ks(y, ref, folds)).sum())
        print(f"{k:18s} {v.ks(y, s):8.5f} {m:+10.5f} {sd:7.5f} {p:5.2f} {won}/5  "
              + " ".join(f"{n}:{x:.2f}" for n, x in zip(names, w) if x > 0.005))
    return oof, test, names, R, Rt, res


if __name__ == "__main__":
    main(sys.argv[1:])
