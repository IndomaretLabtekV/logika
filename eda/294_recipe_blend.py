"""The 4-member recipe from the user's list (LightGBM per segment, XGBoost per segment, MLP, LightGBM v10 nested) and
its mixes with eight_seg, as equal rank means. Judged against eight_seg per fold over 3 split seeds and by paired
bootstrap. Writes submissions/v75/<name>.csv for names passed in argv (same checks as 281).
Usage: cd eda && ../.venv/bin/python 294_recipe_blend.py [name,...]
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


def mlp_member(ids):
    files = sorted((R / "results/v73/seq_model").glob("oof_mlp_s*.parquet"))
    o = np.mean([rankdata(pl.read_parquet(p)["mlp"].to_numpy()) for p in files], 0)
    t = pl.read_parquet(R / "results/v73/seq_model/test_mlp_mean.parquet")
    return o, t


def main():
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    ss = pl.read_csv(R / "data/sample_submission.csv").select("application_id")
    yd = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")
    ids, y = yd["application_id"], yd["target"].to_numpy()
    oof, test = {}, {}
    for m in fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg", "xgb_seg16"]:
        a, b = fb.member(m, ids, ss)
        oof[m], test[m] = rankdata(a) / len(y), rankdata(b) / len(b)
    o, t = mlp_member(ids)
    oof["mlp"] = rankdata(o) / len(y)
    tj = ss.join(t, on="application_id", how="left", maintain_order="left")["mlp"].to_numpy()
    test["mlp"] = rankdata(tj) / len(tj)
    nd = R / "results/v73/lgb_nested"
    if (nd / "log_lgb_nested.csv").exists() and pl.read_csv(nd / "log_lgb_nested.csv").height == 15:
        a = pl.read_parquet(nd / "oof_lgb_nested.parquet")
        b = ss.join(pl.read_parquet(nd / "test_lgb_nested.parquet"), on="application_id", how="left", maintain_order="left")
        oof["lgb_v10"], test["lgb_v10"] = rankdata(a["lgb_nested"].to_numpy()) / len(y), rankdata(b["lgb_nested"].to_numpy()) / b.height
    for m in ("mlp", "lgb_v10"):
        if m in oof:
            print(f"member {m:8s} OOF KS {v.ks(y, oof[m]):.5f}")
    e8s = fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"]
    cands = {"eight_seg": e8s, "recipe3": ["lgb_seg", "xgb_seg", "mlp"], "recipe3_x16": ["lgb_seg", "xgb_seg16", "mlp"],
             "seg2": ["lgb_seg", "xgb_seg"], "recipe4": ["lgb_seg", "xgb_seg", "mlp", "lgb_v10"],
             "recipe4_x16": ["lgb_seg", "xgb_seg16", "mlp", "lgb_v10"], "e8s_mlp": e8s + ["mlp"], "e8s_mlp_v10": e8s + ["mlp", "lgb_v10"],
             "e8s_mlp2": e8s + ["mlp", "mlp"]}
    base = np.mean([oof[m] for m in e8s], 0)
    v72 = pl.read_csv(R / "submissions/v72/v72_six_markov_3seed.csv")["default_probability"].to_numpy()
    for name, ms in cands.items():
        if not all(m in oof for m in ms):
            print(f"{name:12s} (waiting for members)")
            continue
        s = np.mean([oof[m] for m in ms], 0)
        mu, pp = bs.boot(y, s, base, n=1000, seed=4)
        d = [v.ks(y[va], s[va]) - v.ks(y[va], base[va]) for sd in (1, 2, 3) for _, va in v.outer(y, sd)]
        print(f"{name:12s} OOF KS {v.ks(y, s):.5f} vs eight_seg {mu:+.5f} P>0 {pp:.2f} folds won {sum(x > 0 for x in d)}/15")
        if name in want:
            r = np.mean([test[m] for m in ms], 0)
            out = ss.with_columns(default_probability=pl.Series(r))
            assert out.height == 30000 and np.isfinite(r).all()
            rho = spearmanr(r, v72)[0]
            assert rho > 0.98, rho
            p = R / f"submissions/v75/v75_{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            out.write_csv(p)
            print(f"   wrote {p} (spearman vs v72 {rho:.5f})")


if __name__ == "__main__":
    main()
