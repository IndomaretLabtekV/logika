"""Fold-0 screening against the v8 recipe under validation_v2 (seed 42, outer fold 0, 14,000 rows).
Reference models come from results/screen/{ref_f0,causilo_f0}. For each candidate: KS alone, and a fixed-weight
addition (20%) to the clean reference blend (trees 50% + causilo 50%); paired bootstrap on the fold rows.
Usage: python screen2.py results/screen/icl_f0/oof_tabpfn.parquet [...]"""
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v

REF = {n: f"../results/screen/ref_f0/oof_{n}.parquet" for n in ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s", "tabm"]}
REF["causilo"] = "../results/screen/causilo_f0/oof_causilo.parquet"
V8 = dict(lgb=.140, lgb_s=.056, xgb=.108, xgb_s=.084, cat=.037, cat_s=.042, tabm=.352, causilo=.137)


def col(path):
    d = pl.read_parquet(path).sort("application_id")
    name = [c for c in d.columns if c not in ("application_id", "target")][0]
    return d["application_id"], d["target"].to_numpy(), d[name].to_numpy()


def rk(x):
    return rankdata(x) / len(x)


def boot(y, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([v.ks(y[i], a[i]) - v.ks(y[i], b[i]) for i in (rng.choice(len(y), len(y)) for _ in range(n))])
    return d.mean(), d.std(), (d > 0).mean()


def load_ref():
    ids, y, _ = col(REF["lgb"])
    m = ~np.isnan(_)
    P = {n: col(p)[2][m] for n, p in REF.items()}
    return ids.filter(m), y[m], {n: rk(x) for n, x in P.items()}


if __name__ == "__main__":
    ids, y, P = load_ref()
    trees = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
    v8 = sum(w * P[n] for n, w in V8.items())
    base = 0.5 * np.mean([P[n] for n in trees], axis=0) + 0.5 * P["causilo"]
    print(f"fold-0 reference: v8 recipe {v.ks(y, v8):.5f}  trees+causilo {v.ks(y, base):.5f}  causilo {v.ks(y, P['causilo']):.5f}")
    for path in sys.argv[1:]:
        cid, _, c = col(path)
        m = ~np.isnan(c)
        assert (cid.filter(m) == ids).all()
        c = rk(c[m])
        for tag, s, ref in [("alone vs v8", c, v8), ("base+20% vs base", 0.8 * base + 0.2 * c, base),
                            ("base+20% vs v8", 0.8 * base + 0.2 * c, v8)]:
            mu, sd, p = boot(y, s, ref)
            print(f"{path.split('/')[-1]:32s} {tag:18s} KS {v.ks(y, s):.5f}  d {mu:+.5f} sd {sd:.5f} P>0 {p:.2f}")
