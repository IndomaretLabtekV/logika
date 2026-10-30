"""Final recipe candidates, rule fixed before the last members finished (research-log 30 Oct ~10:35):
  tree part (weight 2/3) = mean rank of the per-segment trees [+ LightGBM v10 nested for B]
  smooth part (weight 1/3) = mean rank of the MLP seeds [+ the per-segment MLP for C]
A = 4 segment trees + MLP;  B = A with lgb_v10 in the tree part;  C = A with MLP-per-segment in the smooth part;
D = B and C together. Judged against eight_seg (per fold over 3 split seeds, paired bootstrap) and written to
submissions/v75/ with the usual checks for the names passed in argv.
Usage: cd eda && ../.venv/bin/python 295_final_recipe.py [A,B,C,D]
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

import valid2 as v

R = Path(__file__).resolve().parent.parent
fb = importlib.import_module("281_final_blend")
bs = importlib.import_module("275_blend_scheme")
SEGTREES = ["lgb_seg", "xgb_seg16", "lgb_s_seg", "cat_s_seg"]


def nn_member(tag, ss, n):
    d = R / "results/v73/seq_model"
    fs = sorted(d.glob(f"oof_{tag}_s*.parquet"))
    o = rankdata(np.mean([rankdata(pl.read_parquet(p)[tag].to_numpy()) for p in fs], 0)) / n
    t = [ss.join(pl.read_parquet(d / p.name.replace("oof_", "test_")), on="application_id", how="left", maintain_order="left")[tag].to_numpy() for p in fs]
    return o, rankdata(np.mean([rankdata(x) for x in t], 0)) / len(t[0]), len(fs)


def main():
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    ss = pl.read_csv(R / "data/sample_submission.csv").select("application_id")
    yd = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")
    ids, y = yd["application_id"], yd["target"].to_numpy()
    oof, test = {}, {}
    for m in fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"] + SEGTREES:
        a, b = fb.member(m, ids, ss)
        oof[m], test[m] = rankdata(a) / len(y), rankdata(b) / len(b)
    for tag in ("mlp", "mlpseg"):
        o, t, k = nn_member(tag, ss, len(y))
        oof[tag], test[tag] = o, t
        print(f"member {tag:8s} {k} seeds OOF KS {v.ks(y, o):.5f}")
    nd = R / "results/v73/lgb_nested"
    a = pl.read_parquet(nd / "oof_lgb_nested.parquet")
    assert pl.read_csv(nd / "log_lgb_nested.csv").height == 15 and (a["application_id"] == ids).all()
    b = ss.join(pl.read_parquet(nd / "test_lgb_nested.parquet"), on="application_id", how="left", maintain_order="left")
    oof["lgb_v10"], test["lgb_v10"] = rankdata(a["lgb_nested"].to_numpy()) / len(y), rankdata(b["lgb_nested"].to_numpy()) / b.height
    print(f"member lgb_v10  OOF KS {v.ks(y, oof['lgb_v10']):.5f}")

    def mix(src, trees, smooth):
        return 2 / 3 * np.mean([src[m] for m in trees], 0) + 1 / 3 * np.mean([src[m] for m in smooth], 0)

    cands = {"A": (SEGTREES, ["mlp"]), "B": (SEGTREES + ["lgb_v10"], ["mlp"]), "C": (SEGTREES, ["mlp", "mlpseg"]),
             "D": (SEGTREES + ["lgb_v10"], ["mlp", "mlpseg"])}
    base = np.mean([oof[m] for m in fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"]], 0)
    v72 = pl.read_csv(R / "submissions/v72/v72_six_markov_3seed.csv")["default_probability"].to_numpy()
    for name, (trees, smooth) in cands.items():
        s = mix(oof, trees, smooth)
        mu, pp = bs.boot(y, s, base, n=1000, seed=7)
        d = [v.ks(y[va], s[va]) - v.ks(y[va], base[va]) for sd in (1, 2, 3) for _, va in v.outer(y, sd)]
        print(f"{name}: OOF KS {v.ks(y, s):.5f} vs eight_seg {mu:+.5f} P>0 {pp:.2f} won {sum(x > 0 for x in d)}/15")
        if name in want:
            r = mix(test, trees, smooth)
            out = ss.with_columns(default_probability=pl.Series(r))
            assert out.height == 30000 and np.isfinite(r).all()
            rho = spearmanr(r, v72)[0]
            assert rho > 0.98, rho
            p = R / f"submissions/v75/v75_recipe_{name}.csv"
            out.write_csv(p)
            print(f"   wrote {p} (spearman vs v72 {rho:.5f})")


if __name__ == "__main__":
    main()
