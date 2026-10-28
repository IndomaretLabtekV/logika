"""Paired 5-fold check of label-free feature blocks: v22 LGB on main features vs main + data_pq/<block>.parquet.
Same seed-42 outer folds and inner stop split as validation_v2; screening only (no test predictions).
Usage: cd eda && ../.venv/bin/python 252_feature_add_eval.py f251_repayment_markov[,f250_distress_onset] [prefix|-] [seed]
"""
import sys

import numpy as np
import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v71/feature_add_eval"


def main():
    blocks = sys.argv[1].split(",")
    prefix = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    sfx = "" if seed == 42 else f"_s{seed}"
    feats = pl.read_parquet(PQ / "features.parquet")
    base = v.lgb_fit(dict(v.LGB, n_jobs=8))

    X, Xt, y, ids, test_ids = v.frames(feats)
    bp = OUT / f"oof_lgb_base{sfx}.parquet"
    if not bp.exists():
        v.run(f"lgb_base{sfx}", base, X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)

    aug = feats
    for b in blocks:
        e = pl.read_parquet(PQ / f"{b}.parquet")
        if prefix:
            e = e.select("application_id", *[c for c in e.columns if c.startswith(prefix)])
        aug = aug.join(e, on="application_id", how="left")
    tag = "+".join(blocks) + (f"_{prefix}" if prefix else "") + sfx
    Xa, Xta, ya, ida, _ = v.frames(aug)
    assert ida.to_list() == ids.to_list() and (ya == y).all()
    print(f"{tag}: {Xa.shape[1] - X.shape[1]} extra columns", flush=True)
    v.run(f"lgb_{tag}", base, Xa, Xta, y, ids, test_ids, OUT, seeds=[seed], test=False)

    b = pl.read_parquet(bp)[f"lgb_base{sfx}"].to_numpy()
    c = pl.read_parquet(OUT / f"oof_lgb_{tag}.parquet")[f"lgb_{tag}"].to_numpy()
    wins = 0
    for k, (_, va) in enumerate(v.outer(y, seed)):
        kb, kc = v.ks(y[va], b[va]), v.ks(y[va], c[va])
        wins += kc > kb
        print(f"fold {k}: base {kb:.5f}  new {kc:.5f}  delta {kc - kb:+.5f}")
    rng = np.random.default_rng(252)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.array([v.ks(y[r], c[r]) - v.ks(y[r], b[r])
                  for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))])
    print(f"pooled: base {v.ks(y, b):.5f} new {v.ks(y, c):.5f} delta {v.ks(y, c) - v.ks(y, b):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {np.mean(d > 0):.3f} folds won {wins}/5")


if __name__ == "__main__":
    main()
