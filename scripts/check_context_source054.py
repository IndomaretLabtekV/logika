"""Separate numerical context-source variation from the learned correction gain."""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r


def main():
    data, _ = r.inputs(r.specifications())
    y = data.target.astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=31027).split(data, y))
    spline = r.read_oof(
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
        data,
        folds,
    )
    old = 0.75 * r.read_oof(r.baseline_directory(5, 31027), data, folds) + 0.25 * spline
    rebuilt = (
        0.75 * r.read_oof(r.DEST / "context_holdout_base/cv5_seed31027", data, folds)
        + 0.25 * spline
    )
    corrected = r.read_oof(
        r.DEST / "cv5_seed31027/kumo_offset_l20p003_unweighted_d0p5", data, folds
    )
    result = {}
    for name, values in [
        ("original_checkpoint_recipe", old),
        ("rebuilt_without_correction", rebuilt),
        ("rebuilt_with_correction", corrected),
    ]:
        per_fold = [r.scores(y.iloc[vi], values[vi])["ks"] for _, vi in folds]
        result[name] = dict(
            **r.scores(y, values),
            fold_ks=per_fold,
            mean_fold_ks=float(np.mean(per_fold))
        )
    result["numerical_source_pooled_delta"] = (
        result["rebuilt_without_correction"]["ks"]
        - result["original_checkpoint_recipe"]["ks"]
    )
    result["correction_alone_pooled_delta"] = (
        result["rebuilt_with_correction"]["ks"]
        - result["rebuilt_without_correction"]["ks"]
    )
    result["correction_alone_fold_deltas"] = (
        np.asarray(result["rebuilt_with_correction"]["fold_ks"])
        - result["rebuilt_without_correction"]["fold_ks"]
    ).tolist()
    result["interpretation"] = (
        "Diagnostic attribution only, evaluated after all correction recipes were completed. No source draw was chosen or rejected using these metrics. The rebuilt source is numerically distinct from the public checkpoint."
    )
    r.write_json(r.DEST / "numerical_source_control.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
