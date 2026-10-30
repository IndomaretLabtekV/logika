"""H2 evaluation over seeds: mean rank of the sequence-model OOFs (291) per tag, alone and as a rank member next to
eight_seg (per fold over 3 split seeds, paired bootstrap). Writes the seed-averaged test ranks for the tag.
Usage: cd eda && ../.venv/bin/python 292_seq_blend_eval.py seq|seqfull
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

import valid2 as v

R = Path(__file__).resolve().parent.parent
D = R / "results/v73/seq_model"
fb = importlib.import_module("281_final_blend")
bs = importlib.import_module("275_blend_scheme")


def main():
    tag = sys.argv[1]
    files = sorted(D.glob(f"oof_{tag}_s*.parquet"))
    yd = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")
    ids, y = yd["application_id"], yd["target"].to_numpy()
    o = []
    for p in files:
        d = pl.read_parquet(p)
        assert (d["application_id"] == ids).all()
        o.append(rankdata(d[tag].to_numpy()) / len(y))
        print(f"{p.name}: KS {v.ks(y, d[tag].to_numpy()):.5f}")
    s = np.mean(o, 0)
    ss = pl.read_csv(R / "data/sample_submission.csv").select("application_id")
    e8s = np.mean([rankdata(fb.member(m, ids, ss)[0]) / len(y) for m in fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"]], 0)
    print(f"{tag} mean of {len(files)} seeds: KS {v.ks(y, s):.5f}, spearman vs eight_seg {np.corrcoef(rankdata(s), rankdata(e8s))[0, 1]:.4f}")
    for w in (0.1, 0.15, 0.2, 0.3):
        b = (1 - w) * e8s + w * s
        mu, pp = bs.boot(y, b, e8s, n=1000, seed=2)
        d = [v.ks(y[va], b[va]) - v.ks(y[va], e8s[va]) for sd in (1, 2, 3) for _, va in v.outer(y, sd)]
        print(f"   w {w:.2f}: KS {v.ks(y, b):.5f} vs {v.ks(y, e8s):.5f} boot {mu:+.5f} P>0 {pp:.2f} folds won {sum(x > 0 for x in d)}/15")
    t = [pl.read_parquet(D / p.name.replace("oof_", "test_")) for p in files]
    tr = np.mean([rankdata(x[tag].to_numpy()) / x.height for x in t], 0)
    pl.DataFrame({"application_id": t[0]["application_id"], tag: tr}).write_parquet(D / f"test_{tag}_mean.parquet")


if __name__ == "__main__":
    main()
