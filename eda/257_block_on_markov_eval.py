"""Like 252, but the reference already contains f251 (the submitted trees+Markov features). Single v22 LGB, one seed.
Usage: cd eda && ../.venv/bin/python 257_block_on_markov_eval.py f255_process_chains prefix1,prefix2|- [seed]
"""
import sys

import numpy as np
import polars as pl

import valid2 as v
from common import PQ, ROOT

OUT = ROOT / "results/v71/block_on_markov"


def main():
    block = sys.argv[1]
    prefixes = None if len(sys.argv) < 3 or sys.argv[2] == "-" else tuple(sys.argv[2].split(","))
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    ref = pl.read_parquet(PQ / "features.parquet").join(pl.read_parquet(PQ / "f251_repayment_markov.parquet"), on="application_id", how="left")
    e = pl.read_parquet(PQ / f"{block}.parquet")
    if prefixes:
        e = e.select("application_id", *[c for c in e.columns if c.startswith(prefixes)])
    aug = ref.join(e, on="application_id", how="left")
    fit = v.lgb_fit(dict(v.LGB, n_jobs=8))
    X, Xt, y, ids, test_ids = v.frames(ref)
    rn = f"lgb_ref_s{seed}"
    if not (OUT / f"oof_{rn}.parquet").exists():
        v.run(rn, fit, X, Xt, y, ids, test_ids, OUT, seeds=[seed], test=False)
    Xa, Xta, ya, ida, _ = v.frames(aug)
    assert ida.to_list() == ids.to_list()
    tag = f"lgb_{block}_{'-'.join(prefixes) if prefixes else 'all'}_s{seed}"
    print(f"{tag}: {Xa.shape[1] - X.shape[1]} extra columns", flush=True)
    v.run(tag, fit, Xa, Xta, y, ids, test_ids, OUT, seeds=[seed], test=False)
    b = pl.read_parquet(OUT / f"oof_{rn}.parquet")[rn].to_numpy()
    c = pl.read_parquet(OUT / f"oof_{tag}.parquet")[tag].to_numpy()
    wins = 0
    for k, (_, va) in enumerate(v.outer(y, seed)):
        kb, kc = v.ks(y[va], b[va]), v.ks(y[va], c[va])
        wins += kc > kb
        print(f"fold {k}: ref {kb:.5f}  new {kc:.5f}  delta {kc - kb:+.5f}")
    rng = np.random.default_rng(257)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.array([v.ks(y[r], c[r]) - v.ks(y[r], b[r])
                  for r in (np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(500))])
    print(f"pooled: ref {v.ks(y, b):.5f} new {v.ks(y, c):.5f} delta {v.ks(y, c) - v.ks(y, b):+.5f} "
          f"ci95 {np.quantile(d, [.025, .975]).round(5).tolist()} P>0 {np.mean(d > 0):.3f} folds won {wins}/5")


if __name__ == "__main__":
    main()
