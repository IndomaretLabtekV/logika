"""Paired KS audit for nested Kumo candidates; evaluation never selects a recipe.

Input: one ID-bearing OOF parquet, or a directory of s<seed>_f<fold>_outer.parquet.
Only the latter can demonstrate 5-fold x 3-seed coverage. Historical baseline
Kumo/v31 are reported as references; optional baseline_dir supplies truly matched
per-seed/per-fold baseline inference. Never infer missing predictions or seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import valid2 as v
from common import OUT, PQ, ROOT, features
from screen2 import boot


TREES = ("lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s")


def rk(x):
    return rankdata(x) / len(x)


def read_prediction(path, ids, target, col=None):
    d = pl.read_parquet(path)
    assert d["application_id"].n_unique() == d.height, f"Duplicate IDs: {path}"
    cols = [c for c in d.columns if c not in ("application_id", "target")]
    if col is None:
        assert len(cols) == 1, f"Choose --column from {cols}: {path}"
        col = cols[0]
    assert col in cols, f"Missing {col}: {path}"
    aligned = pl.DataFrame({"application_id": ids}).join(d, on="application_id", how="left", maintain_order="left")
    p = aligned[col].cast(pl.Float64).to_numpy()
    if "target" in d.columns:
        observed = aligned["target"].to_numpy()
        m = np.isfinite(p)
        assert np.array_equal(observed[m], target[m]), f"Label/ID mismatch: {path}"
    return p


def collect(directory, ids, y, seeds, folds, column):
    p = np.full((len(seeds), len(y)), np.nan)
    details = []
    for j, seed in enumerate(seeds):
        for fold in folds:
            path = directory / f"s{seed}_f{fold}_outer.parquet"
            assert path.is_file(), f"Missing required fit: {path}"
            d = pl.read_parquet(path)
            _, va = v.outer(y, seed)[fold]
            assert set(d["application_id"].to_list()) == set(ids[va].to_list()), f"Outer membership mismatch: {path}"
            x = read_prediction(path, ids, y, column)
            assert np.isfinite(x[va]).all() and np.isnan(x[np.setdiff1d(np.arange(len(y)), va)]).all()
            assert np.isnan(p[j, va]).all(), "Overlapping folds"
            assert ((x[va] >= 0) & (x[va] <= 1)).all(), path
            p[j, va] = x[va]
            details.append({"seed": seed, "fold": fold, "rows": len(va), "ks": v.ks(y[va], x[va]),
                            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    # For screening distinct seed folds are distinct row sets: do not average partial coverage.
    covered = np.isfinite(p)
    assert np.all(covered == covered[0]), "Different row coverage across seeds; finish all folds before averaging"
    return p.mean(axis=0), p, details


def historical(ids, y):
    r = ROOT / "results"
    def load(path):
        p = read_prediction(r / path, ids, y)
        assert np.isfinite(p).all(), path
        return p
    t = np.mean([rk(load(f"v22/v22/gbdt/oof_{m}.parquet")) for m in TREES], axis=0)
    c1 = load("v22/v22/fm/oof_causilo.parquet")
    k1 = load("v28m/oof_kumo_large_c56000.parquet")
    k3 = (k1 + 2 * load("v38/oof_kumo_large_c56000.parquet")) / 3
    c3 = (c1 + load("screen/seeds/s7/oof_causilo.parquet") + load("screen/seeds/s2026/oof_causilo.parquet")) / 3
    return {"kumo_seed42_historical": k1, "kumo_3seed_historical": k3,
            "v31_historical": (t + rk(c1) + rk(k1)) / 3,
            "v31_3seed_icl_reference": (t + rk(c3) + rk(k3)) / 3}


def compare(y, a, b, n, fold_masks):
    delta_mean, sd, prob = boot(y, a, b, n=n)
    fold_deltas = [v.ks(y[m], a[m]) - v.ks(y[m], b[m]) for m in fold_masks]
    return {"candidate_ks": v.ks(y, a), "reference_ks": v.ks(y, b),
            "delta": v.ks(y, a) - v.ks(y, b), "bootstrap_delta_mean": float(delta_mean),
            "bootstrap_sd": float(sd), "p_gt_zero": float(prob),
            "fold_deltas_seed42_assignment": fold_deltas,
            "fold_wins": sum(d > 0 for d in fold_deltas), "fold_count": len(fold_deltas)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--column")
    ap.add_argument("--baseline-dir", type=Path)
    ap.add_argument("--baseline-column")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    seeds, folds = [int(s) for s in a.seeds.split(",")], [int(s) for s in a.folds.split(",")]
    assert len(seeds) == len(set(seeds)) and len(folds) == len(set(folds))
    assert a.bootstrap > 1 and all(0 <= f < 5 for f in folds)
    _, _, y, ids, _ = v.frames(features())
    if a.candidate.is_dir():
        p, per_seed, details = collect(a.candidate, ids, y, seeds, folds, a.column)
    else:
        assert len(seeds) == 1, "Averaged legacy OOF file cannot prove multiple seed fits"
        p = read_prediction(a.candidate, ids, y, a.column)
        per_seed, details = None, []
    mask = np.isfinite(p)
    assert mask.any() and ((p[mask] >= 0) & (p[mask] <= 1)).all()
    ref = historical(ids, y)
    if a.baseline_dir:
        b, bs, bdetails = collect(a.baseline_dir, ids, y, seeds, folds, a.baseline_column)
        assert np.array_equal(np.isfinite(b), mask)
        ref["kumo_matched_baseline"] = b
    else:
        bs, bdetails = None, []
    # Historical blend ranks are defined on all 70k rows, as in the original v31 evaluator.
    fold_masks = [np.isin(np.flatnonzero(mask), va) for _, va in v.outer(y, 42)]
    fold_masks = [m for m in fold_masks if m.any()]
    comparisons = {name: compare(y[mask], p[mask], x[mask], a.bootstrap, fold_masks) for name, x in ref.items()}
    seed_metrics = []
    if per_seed is not None:
        for j, seed in enumerate(seeds):
            row = {"seed": seed, "pooled_ks": v.ks(y[mask], per_seed[j, mask])}
            if bs is not None:
                row["matched_baseline_ks"] = v.ks(y[mask], bs[j, mask])
                row["matched_delta"] = row["pooled_ks"] - row["matched_baseline_ks"]
            seed_metrics.append(row)
    complete = set(folds) == set(range(5)) and len(seeds) >= 3 and mask.all() and len(details) == 5 * len(seeds)
    required = ["kumo_matched_baseline", "v31_historical", "v31_3seed_icl_reference"]
    passes = complete and all(name in comparisons and comparisons[name]["delta"] > .003 and
                             comparisons[name]["p_gt_zero"] >= .9 and comparisons[name]["fold_wins"] >= 3
                             for name in required)
    result = {"rows": int(mask.sum()), "seeds": seeds, "folds": folds, "fit_provenance": details,
              "baseline_provenance": bdetails, "seed_metrics": seed_metrics, "comparisons": comparisons,
              "complete_5fold_3seed": bool(complete), "passes_local_gate": bool(passes),
              "public_forecast": None, "submission_allowed_by_this_report": False,
              "limits": ["Historical multi-seed Kumo file stores a seed average, not separate seed fits.",
                         "Public ICL transfer is uncertain; passing local gate alone is not a public score.",
                         "No recipe, threshold, feature set or blend weight is selected by this evaluator."],
              "features_sha256": hashlib.sha256((PQ / "features.parquet").read_bytes()).hexdigest()}
    print("SCREENING ONLY" if not complete else "COMPLETE LOCAL EVALUATION", flush=True)
    for name, c in comparisons.items():
        print(f"{name}: KS {c['candidate_ks']:.6f} vs {c['reference_ks']:.6f} delta={c['delta']:+.6f} "
              f"sd={c['bootstrap_sd']:.6f} P>0={c['p_gt_zero']:.3f} wins={c['fold_wins']}/{c['fold_count']}", flush=True)
    print("seed metrics:", seed_metrics, "local gate:", passes, flush=True)
    (OUT / f"155_{a.tag}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    names = list(comparisons)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(names, [comparisons[n]["delta"] for n in names],
            xerr=[comparisons[n]["bootstrap_sd"] for n in names])
    ax.axvline(0, color="black", lw=.7)
    ax.axvline(.003, color="green", ls="--", lw=.8)
    ax.set(xlabel="Paired pooled KS delta (error bar: bootstrap SD)", title=f"{a.tag}: {mask.sum():,} rows; no selection")
    fig.tight_layout()
    fig.savefig(OUT / f"155_{a.tag}.png", dpi=130)
    plt.close(fig)
    if details:
        pd.DataFrame(details).to_csv(OUT / f"155_{a.tag}_fits.csv", index=False)


if __name__ == "__main__":
    main()
