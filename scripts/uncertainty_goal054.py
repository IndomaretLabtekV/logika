"""Paired bootstrap uncertainty for a completed, independently audited candidate.

Resample within outer fold and outcome class. This conditions on fitted OOF
predictions: it does not include retraining variance or correct selection bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r


def ordered_ks(y, p):
    order = np.argsort(p, kind="mergesort")
    sorted_p = np.asarray(p)[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(order) - 1]
    positive = np.asarray(y)[order] == 1

    def calculate(weight):
        w = weight[order]
        a = np.cumsum(w * positive)
        b = np.cumsum(w * ~positive)
        return float(np.max(np.abs(a[ends] / a[-1] - b[ends] / b[-1])))

    return calculate


def run(candidate=None, weight=None, repetitions=300, seed=192026):
    report_bytes = (r.DEST / "independent_verification.json").read_bytes()
    report = json.loads(report_bytes)
    if candidate is None:
        row = max(
            (z for z in report["comparisons"] if z.get("candidate")),
            key=lambda z: z["ks"],
        )
    else:
        row = next(
            z
            for z in report["comparisons"]
            if z.get("candidate") == candidate and z.get("new_model_weight") == weight
        )
    assert report["completed_fold_counts"][row["candidate"]] == 5
    data, _ = r.inputs(r.specifications())
    y = data.target.to_numpy(dtype=int)
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(data, y)
    )
    base = 0.75 * r.read_oof(r.baseline_directory(5, r.FOLD_SEED), data, folds)
    base += 0.25 * r.read_oof(
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
        data,
        folds,
    )
    directory = r.DEST / "cv5_seed31027" / row["candidate"]
    new = r.read_oof(directory, data, folds)
    new = (1 - row["new_model_weight"]) * base + row["new_model_weight"] * new
    base_ks, new_ks = ordered_ks(y, base), ordered_ks(y, new)
    unit = np.ones(len(y), dtype=int)
    assert abs(base_ks(unit) - report["comparisons"][0]["ks"]) < 1e-12
    assert abs(new_ks(unit) - row["ks"]) < 1e-12
    strata = [vi[y[vi] == label] for _, vi in folds for label in [0, 1]]
    generator = np.random.default_rng(seed)
    values = []
    for iteration in range(repetitions):
        counts = np.zeros(len(y), dtype=int)
        for indices in strata:
            counts[indices] = generator.multinomial(
                len(indices), np.full(len(indices), 1 / len(indices))
            )
        b, n = base_ks(counts), new_ks(counts)
        values.append([b, n, n - b])
        if (iteration + 1) % 50 == 0:
            print("BOOTSTRAP", iteration + 1, "/", repetitions, flush=True)
    values = np.asarray(values)
    result = dict(
        candidate=row["candidate"],
        new_model_weight=row["new_model_weight"],
        audited_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        point_baseline_ks=base_ks(unit),
        point_candidate_ks=new_ks(unit),
        point_delta_ks=new_ks(unit) - base_ks(unit),
        repetitions=repetitions,
        seed=seed,
        percentile_interval_95={
            name: np.quantile(values[:, i], [0.025, 0.975]).tolist()
            for i, name in enumerate(["baseline_ks", "candidate_ks", "delta_ks"])
        },
        fraction_positive_delta=float(np.mean(values[:, 2] > 0)),
        caveat="Paired bootstrap conditional on fitted OOF predictions, stratified by fold and outcome. It omits retraining uncertainty and does not correct repeated model/feature/weight selection. The fraction is not a selection-corrected p-value.",
    )
    r.write_json(r.DEST / "paired_uncertainty.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--weight", type=float)
    parser.add_argument("--repetitions", type=int, default=300)
    args = parser.parse_args()
    assert args.repetitions >= 100
    run(args.candidate, args.weight, args.repetitions)
