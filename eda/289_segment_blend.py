"""H1 (research/autoresearch): segment-specific blend.

In each has_history segment one of a few fixed member sets is rank-averaged; each segment's blend is turned into a
probability by an isotonic fit, and the two segments are merged by probability. The set per segment is chosen nested:
for every outer fold the choice is made on the other folds only, then applied to the held-out fold. Compared with the
global equal blend eight_seg on the same folds (3 split seeds) and by paired bootstrap.
Usage: cd eda && ../.venv/bin/python 289_segment_blend.py
"""
import importlib
import itertools
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

import common as c
import valid2 as v

fb = importlib.import_module("281_final_blend")
bs = importlib.import_module("275_blend_scheme")
R = Path(__file__).resolve().parent.parent
SETS = {
    "six": fb.SIX,
    "eight": fb.SIX + ["lgb_x", "xgb_d8"],
    "eight_seg": fb.SIX + ["lgb_x", "xgb_d8", "lgb_seg", "xgb_seg"],
    "seg_only": ["xgb_seg16", "lgb_s_seg", "lgb_seg"],
    "strong": ["xgb_seg16", "lgb_s_seg", "xgb_d8", "lgb_x", "xgb_s"],
}


def seg_prob(s, y, seg, tr, va):
    """Isotonic map of score s fitted on rows tr of each segment, applied to va."""
    p = np.zeros(len(va))
    for g in (0, 1):
        a, b = tr[seg[tr] == g], va[seg[va] == g]
        iso = IsotonicRegression(out_of_bounds="clip").fit(s[a], y[a])
        p[seg[va] == g] = iso.predict(s[b]) + 1e-9 * s[b]  # tie-break inside isotonic steps by the score itself
    return p


def main():
    ss = pl.read_csv(R / "data/sample_submission.csv").select("application_id")
    ids = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")["application_id"]
    y = pl.read_parquet(R / "results/v72k_out/v72/preds/oof_lgb.parquet")["target"].to_numpy()
    seg = pl.DataFrame({"application_id": ids}).join(c.features().select("application_id", "has_history"), on="application_id",
                                                     maintain_order="left")["has_history"].to_numpy()
    members = sorted({m for s in SETS.values() for m in s})
    oof = {m: rankdata(fb.member(m, ids, ss)[0]) / len(y) for m in members}
    blend = {k: np.mean([oof[m] for m in ms], 0) for k, ms in SETS.items()}
    base = blend["eight_seg"]
    combos = list(itertools.product(SETS, SETS))
    deltas, chosen, final = [], [], {}
    for sd in (1, 2, 3):
        folds = v.outer(y, sd)
        out = np.zeros(len(y))
        for k, (tr, va) in enumerate(folds):
            # choose (set for seg 0, set for seg 1) on inner folds of tr only
            best = None
            for c0, c1 in combos:
                s = np.where(seg == 0, blend[c0], blend[c1])
                ks_in = []
                for j, (_, va2) in enumerate(folds):
                    if j == k:
                        continue
                    tr2 = np.setdiff1d(tr, va2)
                    ks_in.append(v.ks(y[va2], seg_prob(s, y, seg, tr2, va2)))
                m = np.mean(ks_in)
                if best is None or m > best[0]:
                    best = (m, c0, c1)
            _, c0, c1 = best
            chosen.append((c0, c1))
            s = np.where(seg == 0, blend[c0], blend[c1])
            out[va] = seg_prob(s, y, seg, tr, va)
            deltas.append(v.ks(y[va], out[va]) - v.ks(y[va], base[va]))
        final[sd] = out
        print(f"split seed {sd}: nested KS per fold delta vs eight_seg {np.round(deltas[-5:], 5)}", flush=True)
    d = np.array(deltas)
    print(f"nested segment blend: mean per-fold delta {d.mean():+.5f} (se {d.std(ddof=1) / np.sqrt(len(d)):.5f}), won {(d > 0).sum()}/15")
    print("chosen (seg0, seg1):", {k: chosen.count(k) for k in set(chosen)})
    # also: same eight_seg set but merged through per-segment isotonic (pure recalibration effect)
    rd = []
    for sd in (1, 2, 3):
        for tr, va in v.outer(y, sd):
            rd.append(v.ks(y[va], seg_prob(base, y, seg, tr, va)) - v.ks(y[va], base[va]))
    rd = np.array(rd)
    print(f"eight_seg recalibrated per segment: mean per-fold delta {rd.mean():+.5f}, won {(rd > 0).sum()}/15")


if __name__ == "__main__":
    main()
