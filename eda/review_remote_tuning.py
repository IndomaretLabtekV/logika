"""Compare saved remote predictions with v8; no model fitting or notebook execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata

ROOT = Path(__file__).resolve().parents[1]


def align(reference, candidate):
    assert reference["application_id"].n_unique() == reference.height
    assert candidate["application_id"].n_unique() == candidate.height
    assert set(reference["application_id"]) == set(candidate["application_id"])
    return reference.select("application_id").join(candidate, on="application_id",
                                                   validate="1:1", maintain_order="left")


def score(y, p):
    assert len(y) == len(p) and np.isfinite(p).all() and set(np.unique(y)) == {0, 1}
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def main(folder):
    baseline = pl.read_parquet(ROOT / "eda/v8_review/out/00_bound_oof.parquet")
    labels = align(baseline, pl.read_csv(ROOT / "data/train_applications.csv").select("application_id", "target"))
    assert np.array_equal(labels["target"], baseline["target"])
    new = align(baseline, pl.read_parquet(folder / "oof.parquet"))
    assert np.array_equal(new["target"], baseline["target"])
    y = baseline["target"].to_numpy()
    base = rankdata(baseline["blend"].to_numpy()) / len(y)
    folds = new["fold"].to_numpy()
    assert set(np.unique(folds)) == set(range(5))
    predictions = {"v8": base}
    for name in ("control", "tuned"):
        p = new[name].to_numpy()
        assert ((p >= 0) & (p <= 1)).all()
        predictions[name] = p
        predictions[f"v8_mix_{name}"] = .8 * base + .2 * rankdata(p) / len(p)
    rng = np.random.default_rng(2026)
    classes = [np.flatnonzero(y == k) for k in (0, 1)]
    gains = {n: [] for n in predictions if n != "v8"}
    for _ in range(500):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        reference = score(y[idx], base[idx])
        for name in gains:
            gains[name].append(score(y[idx], predictions[name][idx]) - reference)
    rows = []
    for name, p in predictions.items():
        row = dict(candidate=name, pooled_ks=score(y, p), delta_vs_v8=score(y, p)-score(y, base))
        if name != "v8":
            row["conditional_bootstrap_95"] = np.quantile(gains[name], [.025, .975]).tolist()
            row["fold_deltas"] = [score(y[folds == k], p[folds == k])-score(y[folds == k], base[folds == k]) for k in range(5)]
        rows.append(row)
    out = folder / "v8_review"
    out.mkdir(exist_ok=True)
    result = dict(metrics=rows, fixed_new_model_weight=.2,
        caveat="Exploratory comparison on reused OOF rows and differing training folds. Bootstrap conditions on fitted models; public score is unknown.")
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r["candidate"] for r in rows], [r["delta_vs_v8"] for r in rows])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(ylabel="Pooled KS gain over v8", title="Fixed comparisons; no weight search")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)
    base_test = pl.read_csv(ROOT / "results/v8/submission.csv")
    bt = rankdata(base_test["default_probability"].to_numpy()) / base_test.height
    for name in ("control", "tuned"):
        test = align(base_test, pl.read_csv(folder / f"submission_{name}.csv"))
        p = test["default_probability"].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        mix = .8 * bt + .2 * rankdata(p) / len(p)
        base_test.select("application_id").with_columns(default_probability=pl.Series(mix)).write_csv(
            out / f"submission_v8_mix_{name}.csv")
    print(json.dumps(result, indent=2))


def self_check():
    reference = pl.DataFrame({"application_id": ["a", "b"]})
    reversed_rows = pl.DataFrame({"application_id": ["b", "a"], "prediction": [.8, .2]})
    assert align(reference, reversed_rows)["prediction"].to_list() == [.2, .8]
    for wrong in [reversed_rows.head(1), reversed_rows.with_columns(application_id=pl.lit("a"))]:
        try:
            align(reference, wrong)
        except AssertionError:
            continue
        raise AssertionError("Missing or duplicate IDs must be rejected")
    assert score(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.
    print("Alignment and KS self-check passed; no models were fitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        assert args.folder is not None, "Provide a directory containing completed remote outputs"
        main(args.folder)
