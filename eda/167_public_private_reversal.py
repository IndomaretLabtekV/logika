"""Fixed-prediction diagnosis of a 9k public / 21k private split. No model selection.

Preregistered: docs/integration/protokol_diagnosis_public_private.md.
Run --check-only for numerical checks; default runs the empirical OOF simulation.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import ks_2samp, rankdata

import valid2 as v
from common import OUT, ROOT
from blend2 import V8


def prepare_score(y, score):
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ends = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    return order, y[order], ends


def count_ks(prepared, counts):
    order, labels, ends = prepared
    weights = counts[order]
    positive = np.cumsum(weights * labels)
    negative = np.cumsum(weights) - positive
    assert positive[-1] > 0 and negative[-1] > 0, "Both classes required"
    difference = positive / positive[-1] - negative / negative[-1]
    return float(np.max(np.abs(difference[ends])))


def sampled_ks(y, score, indices):
    return count_ks(prepare_score(y, score), np.bincount(indices, minlength=len(y)))


def numerical_checks():
    assert callable(sampled_ks), "Missing sample-count KS kernel"
    rng = np.random.default_rng(167)
    checked = 0
    for tied in (False, True):
        y = np.tile([0, 0, 1], 40)
        scores = rng.integers(0, 6, len(y)) if tied else rng.normal(size=len(y))
        for _ in range(10):
            idx = rng.choice(len(y), 79, replace=False)
            actual = sampled_ks(y, scores, idx)
            expected = ks_2samp(scores[idx][y[idx] == 1], scores[idx][y[idx] == 0]).statistic
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
            checked += 1
    # Repeated observations must retain their multiplicity (bootstrap compatibility).
    idx = np.array([0, 0, 1, 2, 4, 5, 5, 8])
    actual = sampled_ks(y, scores, idx)
    expected = ks_2samp(scores[idx][y[idx] == 1], scores[idx][y[idx] == 0]).statistic
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    checked += 1
    print(f"numerical checks: {checked} tied/untied/repeated-row comparisons passed", flush=True)
    return checked


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975)),
            "frequency_gt_zero": float(np.mean(values > 0)),
            "frequency_lt_zero": float(np.mean(values < 0)),
            "frequency_le_minus_003": float(np.mean(values <= -.003))}


def main(checks):
    start = time.time()
    result_dir = ROOT / "results"
    reference = pl.read_parquet(result_dir / "v22/v22/fm/oof_causilo.parquet").sort("application_id")
    ids, y = reference["application_id"], reference["target"].to_numpy().astype(int)
    assert len(y) == ids.n_unique() == 70000 and set(y) == {0, 1}
    provenance = []

    def load(relative):
        path = result_dir / relative
        d = pl.read_parquet(path).sort("application_id")
        assert d.height == d["application_id"].n_unique() == len(y), path
        assert (d["application_id"] == ids).all(), path
        assert np.array_equal(d["target"].to_numpy(), y), path
        columns = [c for c in d.columns if c not in ("application_id", "target")]
        assert len(columns) == 1, (path, columns)
        score = d[columns[0]].to_numpy()
        assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(), path
        provenance.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                           "column": columns[0], "rows": len(score)})
        return score

    rank = lambda s: rankdata(s) / len(s)
    tree_names = ("lgb", "lgb_s", "xgb", "xgb_s", "cat", "cat_s")
    tree_scores = {m: load(f"v22/v22/gbdt/oof_{m}.parquet") for m in tree_names}
    # Preserve the historical v31 family scale: mean of member ranks, without
    # re-ranking that family before blending it with Causilo/Kumo ranks.
    trees = np.mean([rank(tree_scores[m]) for m in tree_names], axis=0)
    kumo = load("v28m/oof_kumo_large_c56000.parquet")
    kumo_three = (kumo + 2 * load("v38/oof_kumo_large_c56000.parquet")) / 3
    causilo = load("v22/v22/fm/oof_causilo.parquet")
    causilo_three = (causilo + 2 * load("v37/oof_causilo.parquet")) / 3
    tabm = load("v22/v22/nn/oof_tabm.parquet")
    base_parts = {**tree_scores, "causilo": causilo, "tabm": tabm}
    v8 = sum(weight * rank(base_parts[name]) for name, weight in V8.items())
    scores = {"trees": trees, "kumo42": kumo, "kumo3": kumo_three, "causilo3": causilo_three,
              "v31": (trees + rank(causilo) + rank(kumo)) / 3, "v8": v8}
    names = list(scores)
    pairs = [("kumo42", "trees"), ("causilo3", "trees"), ("kumo42", "kumo3"), ("v31", "v8")]
    pool = {name: v.ks(y, score) for name, score in scores.items()}
    per_fold = {name: [v.ks(y[va], score[va]) for _, va in v.outer(y, 42)] for name, score in scores.items()}
    prepared = [prepare_score(y, scores[name]) for name in names]
    rng = np.random.default_rng(20261021)
    repetitions = 1000
    # Axis1: public9k, private21k; all models see exactly the same rows each replicate.
    simulations = np.empty((repetitions, 2, len(names)))
    prevalence = np.empty((repetitions, 2))
    for j in range(repetitions):
        selected = rng.choice(len(y), 30000, replace=False)
        public, private = selected[:9000], selected[9000:]
        assert len(np.intersect1d(public, private)) == 0
        for partition, idx in enumerate((public, private)):
            counts = np.bincount(idx, minlength=len(y))
            prevalence[j, partition] = np.mean(y[idx])
            for k, spec in enumerate(prepared):
                simulations[j, partition, k] = count_ks(spec, counts)
        if (j + 1) % 250 == 0:
            print(f"replications {j+1}/{repetitions} elapsed={time.time()-start:.1f}s", flush=True)
    pair_report = {}
    for candidate, baseline in pairs:
        delta = simulations[:, :, names.index(candidate)] - simulations[:, :, names.index(baseline)]
        row = {"pooled_delta": pool[candidate] - pool[baseline],
               "public_9000": summary(delta[:, 0]), "private_21000": summary(delta[:, 1]),
               "equal_component_ks_proxy": summary(delta.mean(axis=1)),
               "public_wins_private_loses_frequency": float(np.mean((delta[:, 0] > 0) & (delta[:, 1] < 0))),
               "public_loses_private_wins_frequency": float(np.mean((delta[:, 0] < 0) & (delta[:, 1] > 0)))}
        pair_report[f"{candidate}_minus_{baseline}"] = row
        print(f"{candidate}-{baseline}: pooled={row['pooled_delta']:+.6f}; "
              f"publicSD={row['public_9000']['sd']:.6f} privateSD={row['private_21000']['sd']:.6f}; "
              f"P(public<0)={row['public_9000']['frequency_lt_zero']:.3f} "
              f"P(public<=-.003)={row['public_9000']['frequency_le_minus_003']:.3f}", flush=True)
    # Recalculate the same historical family contrasts as131, at the true public size.
    # The21k private marginal has the same sampling distribution as the old21k proxy.
    public_scores = {"trees": .56337, "kumo42": .56056, "causilo3": .55333}
    family = {}
    for name in ("kumo42", "causilo3"):
        delta = simulations[:, :, names.index(name)] - simulations[:, :, names.index("trees")]
        observed = public_scores[name] - public_scores["trees"]
        family[name] = {"observed_public_gap_to_trees": observed,
                        "full_oof_gap_to_trees": pool[name] - pool["trees"],
                        "public_minus_oof": public_scores[name] - pool[name]}
        for partition, size in enumerate((9000, 21000)):
            z = (observed - np.mean(delta[:, partition])) / np.std(delta[:, partition], ddof=1)
            family[name][f"proxy_{size}"] = {**summary(delta[:, partition]), "descriptive_z": float(z),
                "frequency_le_observed_public_gap": float(np.mean(delta[:, partition] <= observed))}
        print(f"historical {name}: observedPublicGap={observed:+.5f}; "
              f"z9k={family[name]['proxy_9000']['descriptive_z']:.2f} "
              f"z21k={family[name]['proxy_21000']['descriptive_z']:.2f}", flush=True)
    metrics = {name: {"pooled_ks": pool[name], "fold_ks_seed42_assignment": per_fold[name],
                      "mean_fold_ks": float(np.mean(per_fold[name])),
                      "public_9000": summary(simulations[:, 0, k]),
                      "private_21000": summary(simulations[:, 1, k])} for k, name in enumerate(names)}
    report = {"rows": len(y), "prevalence": float(np.mean(y)), "replications": repetitions,
              "rng_seed": 20261021, "test_rows": 30000, "public_rows": 9000, "private_rows": 21000,
              "rubric_public_weight": .45, "rubric_private_weight": .45,
              "numerical_checks": checks, "provenance": provenance, "metrics": metrics,
              "pairs": pair_report, "historical_family_reassessment": family,
              "partition_prevalence": {"public": summary(prevalence[:, 0]), "private": summary(prevalence[:, 1])},
              "protocol_sha256": sha256(ROOT / "docs/integration/protokol_diagnosis_public_private.md"),
              "source_sha256": sha256(Path(__file__).resolve()), "elapsed_seconds": time.time() - start,
              "limits": ["Conditional fixed-OOF simulation; these are not the friend's two recipes.",
                         "No posterior probability or causal diagnosis for unseen test labels.",
                         "No retraining/seed/context/full-train/selection variance simulated.",
                         "Mean(publicKS,privateKS) is a proxy, not pooled testKS or official rubric points.",
                         "Historical public families are correlated; descriptive z is not a causal test."]}
    stem = OUT / "167_public_private_reversal"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False))
    pl.DataFrame({f"{name}_{label}": simulations[:, partition, k]
                  for k, name in enumerate(names) for partition, label in enumerate(("public", "private"))}).write_csv(stem.with_suffix(".csv"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (candidate, baseline) in zip(axes.flat, pairs):
        delta = simulations[:, :, names.index(candidate)] - simulations[:, :, names.index(baseline)]
        ax.hist(delta[:, 0], bins=35, density=True, histtype="step", label="public9k")
        ax.hist(delta[:, 1], bins=35, density=True, histtype="step", label="private21k")
        ax.axvline(0, color="k", linewidth=.8)
        ax.axvline(-.003, color="r", linestyle=":", label="observed reversal -.003")
        ax.set(title=f"{candidate} - {baseline}; fullOOF {pool[candidate]-pool[baseline]:+.4f}", xlabel="paired deltaKS", ylabel="density")
        ax.legend(fontsize=8)
    fig.suptitle("Fixed OOF: 1000 disjoint 9k/21k splits; not a test-score forecast")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=130)
    print("pooled vs mean-fold:", {name: (round(pool[name], 6), round(metrics[name]['mean_fold_ks'], 6)) for name in names}, flush=True)
    print(f"saved {stem}.*; elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    checks = numerical_checks()
    if not a.check_only:
        main(checks)
