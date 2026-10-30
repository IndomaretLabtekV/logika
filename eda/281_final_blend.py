"""Final blend candidates from every member with test predictions, judged on OOF only.

Members (OOF and test are fit-count-weighted means over seeds):
  six     lgb, lgb_s, xgb, xgb_s, cat, cat_s   v72 kernel (3 seeds) + v73 kernel (5 seeds)
  extra   lgb_x, xgb_d8, cat_d8                v72 kernel (3 seeds) + local 279 (5 seeds)
  seg     lgb_seg, xgb_seg                     local 280 (8 seeds), one model per has_history segment
Candidates are equal-weight rank means. Each is compared with 'six' (the submitted v73 file) by paired bootstrap and
by per-fold KS over three split seeds. Writes submissions/v74/<name>.csv for the candidates passed in argv, after
checks: 30000 rows, ids equal sample_submission, no nulls, Spearman with the submitted v72 file.
Usage: cd eda && ../.venv/bin/python 281_final_blend.py [candidate,...]
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

import valid2 as v

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
bs = importlib.import_module("275_blend_scheme")
SIX = ["lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s"]
EXTRA = ["lgb_x", "xgb_d8", "cat_d8"]
SEG = ["lgb_seg", "xgb_seg"]
SEG4 = SEG + ["xgb5_seg", "xgbd8_seg"]
SHALLOW = ["lgb_s_seg", "cat_s_seg", "xgb_seg16"]
E8S = SIX + ["lgb_x", "xgb_d8"] + SEG
CANDIDATES = {"six": SIX, "nine": SIX + EXTRA, "six_seg": SIX + SEG, "nine_seg": SIX + EXTRA + SEG,
              "six_lgbseg": SIX + ["lgb_seg"], "nine_lgbseg": SIX + EXTRA + ["lgb_seg"],
              "eight": SIX + ["lgb_x", "xgb_d8"], "eight_seg": SIX + ["lgb_x", "xgb_d8"] + SEG,
              "eight_seg4": SIX + ["lgb_x", "xgb_d8"] + SEG4, "nine_seg4": SIX + EXTRA + SEG4,
              "e8s_x16": SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg16"],
              "e8s_lgbs": E8S + ["lgb_s_seg"], "e8s_cats": E8S + ["cat_s_seg"],
              "e8s_shallow": SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg16", "lgb_s_seg", "cat_s_seg"]}


def sources(m):
    """(path, n_fits) pairs for a member."""
    k = R / "v72k_out/v72/preds"
    if m in SIX:
        return [(k, 15), (R / "v73k_out/v72/preds", 25)]
    if m in EXTRA:
        return [(k, 15), (R / "v73/extras_local", 25)]
    if m == "xgb_seg16":
        return [(R / "v73/seg_members", 40, "xgb_seg"), (R / "v73/seg_members_b", 40, "xgb_seg")]
    return [(R / "v73/seg_members", 40)]


def member(m, ids, ss):
    o, t, n = 0.0, 0.0, 0
    for src in sources(m):
        path, k, f = (src + (m,))[:3]
        fits = pl.read_csv(path / f"log_{f}.csv").height
        assert fits == k, f"{path.name}: {fits} of {k} fits done"
        a = pl.read_parquet(path / f"oof_{f}.parquet")
        assert (a["application_id"] == ids).all(), (m, path)
        b = ss.join(pl.read_parquet(path / f"test_{f}.parquet"), on="application_id", how="left", maintain_order="left")[f].to_numpy()
        assert np.isfinite(b).all(), (m, path)
        o, t, n = o + k * a[f].to_numpy(), t + k * b, n + k
    return o / n, t / n


def main():
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    ss = pl.read_csv(ROOT / "data/sample_submission.csv").select("application_id")
    y = pl.read_parquet(R / "v72k_out/v72/preds/oof_lgb.parquet")["target"].to_numpy()
    ids = pl.read_parquet(R / "v72k_out/v72/preds/oof_lgb.parquet")["application_id"]
    oof, test = {}, {}
    for m in SIX + EXTRA + SEG4 + SHALLOW:
        try:
            a, b = member(m, ids, ss)
        except (FileNotFoundError, AssertionError) as e:
            print(f"skip {m}: {e}")
            continue
        oof[m], test[m] = rankdata(a) / len(a), rankdata(b) / len(b)
        print(f"member {m:8s} OOF KS {v.ks(y, oof[m]):.5f}")
    six = np.mean([oof[m] for m in SIX], 0)
    v72 = pl.read_csv(ROOT / "submissions/v72/v72_six_markov_3seed.csv")
    assert (v72["application_id"] == ss["application_id"]).all()
    for name, ms in CANDIDATES.items():
        if not all(m in oof for m in ms):
            continue
        s = np.mean([oof[m] for m in ms], 0)
        mu, pp = bs.boot(y, s, six, n=1000, seed=1)
        d = [v.ks(y[va], s[va]) - v.ks(y[va], six[va]) for sd in (1, 2, 3) for _, va in v.outer(y, sd)]
        print(f"{name:12s} OOF KS {v.ks(y, s):.5f}  vs six {mu:+.5f} P>0 {pp:.2f}  folds won {sum(x > 0 for x in d)}/15")
        if name in want:
            r = np.mean([test[m] for m in ms], 0)
            out = ss.with_columns(default_probability=pl.Series(r))
            assert out.height == 30000 and out["default_probability"].null_count() == 0 and np.isfinite(r).all()
            rho = spearmanr(r, v72["default_probability"].to_numpy())[0]
            assert rho > 0.98, rho
            p = ROOT / f"submissions/v74/v74_{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            out.write_csv(p)
            print(f"   wrote {p} (spearman vs v72 {rho:.5f})")


if __name__ == "__main__":
    main()
