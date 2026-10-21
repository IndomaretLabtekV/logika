"""Descriptive calendar residual diagnostics; never fit a correction on these labels.

This uses all training OOF outcomes for hypothesis generation, not an unbiased
test. Any subsequent calendar model must learn its parameters inside outer train.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import research_goal054 as r
from hybrid_kumo import read_data


def calendar_features(applications):
    a = pd.to_datetime(applications.application_datetime)
    d = pd.to_datetime(applications.decision_datetime)
    assert (a <= d).all()
    return pd.DataFrame(
        dict(
            application_id=applications.application_id,
            cal_year=a.dt.year,
            cal_month=a.dt.month,
            cal_day=a.dt.day,
            cal_weekday=a.dt.dayofweek,
            cal_hour=a.dt.hour,
            cal_decision_day=d.dt.day,
            cal_decision_weekday=d.dt.dayofweek,
            cal_delay_hours=(d - a).dt.total_seconds() / 3600,
        )
    )


def main():
    data, _, _, _ = read_data()
    folds = list(
        StratifiedKFold(r.SPLITS, shuffle=True, random_state=r.FOLD_SEED).split(
            data, data.target
        )
    )
    pk = r.read_oof(r.baseline_directory(r.SPLITS, r.FOLD_SEED), data, folds)
    ps = r.read_oof(
        r.ROOT / "artifacts/optimization/kumo_hybrid/cv5_seed31027/reverse_spline",
        data,
        folds,
    )
    probability = 0.75 * pk + 0.25 * ps
    apps = pd.read_csv(r.ROOT / "data/train_applications.csv")
    calendar = calendar_features(apps)
    anchor = data[["application_id", "target"]].copy()
    anchor["baseline_probability"] = probability
    diagnostic = anchor.merge(calendar, on="application_id", validate="one_to_one")
    assert diagnostic.application_id.equals(anchor.application_id)
    diagnostic["residual"] = diagnostic.target - diagnostic.baseline_probability
    diagnostic["conditional_variance"] = diagnostic.baseline_probability * (
        1 - diagnostic.baseline_probability
    )
    diagnostic["cal_delay_band"] = pd.cut(
        diagnostic.cal_delay_hours, [0, 3, 30, 200], labels=["short", "medium", "long"]
    ).astype(str)
    diagnostic["cal_year_month"] = (
        diagnostic.cal_year.astype(str)
        + "-"
        + diagnostic.cal_month.astype(str).str.zfill(2)
    )
    rows = []
    for column in [
        c for c in diagnostic if c.startswith("cal_") and c != "cal_delay_hours"
    ]:
        for value, part in diagnostic.groupby(column, observed=True, dropna=False):
            rows.append(
                dict(
                    feature=column,
                    value=str(value),
                    rows=len(part),
                    positive_rate=float(part.target.mean()),
                    mean_baseline_probability=float(part.baseline_probability.mean()),
                    mean_residual=float(part.residual.mean()),
                    descriptive_residual_z=float(
                        part.residual.sum() / np.sqrt(part.conditional_variance.sum())
                    ),
                )
            )
    table = pd.DataFrame(rows)
    destination = r.ROOT / "artifacts/tables/goal054_calendar_diagnostic.csv"
    table.to_csv(destination, index=False)
    ranges = dict(
        application_min=apps.application_datetime.min(),
        application_max=apps.application_datetime.max(),
        decision_min=apps.decision_datetime.min(),
        decision_max=apps.decision_datetime.max(),
    )
    summary = dict(
        rows=len(data),
        baseline_ks=r.scores(data.target, probability)["ks"],
        ranges=ranges,
        largest_absolute_residuals=table.assign(
            magnitude=table.descriptive_residual_z.abs()
        )
        .nlargest(20, "magnitude")
        .drop(columns="magnitude")
        .to_dict("records"),
        table=str(destination.relative_to(r.ROOT)),
        interpretation="Descriptive diagnostics reuse training labels and OOF. Z values ignore dependence and repeated comparisons, so are not formal significance tests. Do not fit a correction using these global residuals and score it on these same rows.",
    )
    r.write_json(r.DEST / "calendar_diagnostic.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
