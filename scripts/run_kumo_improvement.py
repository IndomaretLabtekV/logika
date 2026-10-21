"""Run a predeclared screen, lock one recipe, and confirm on matching folds."""

from __future__ import annotations

import argparse
import json
import time

from improve_kumo import DEST, evaluate
from summarize_kumo_improvement import export, summarize

SCREEN = "cv3_seed8247"
CONFIRM = "cv5_seed31027"


def run_candidates(names, splits=3, seed=8247, estimator_batch_size=4):
    evaluate(
        argparse.Namespace(
            candidates=names,
            splits=splits,
            fold_seed=seed,
            folds=None,
            batch_size=8192,
            estimator_batch_size=estimator_batch_size,
            test=True,
            force=False,
        )
    )


def eligible(row, wins):
    return (
        row["delta_ks"] > 0
        and row["delta_mean_fold_ks"] > 0
        and row["fold_wins"] >= wins
    )


def best_eligible(summary):
    rows = [r for r in summary["comparisons"] if eligible(r, 2)]
    return (
        max(rows, key=lambda r: (r["delta_mean_fold_ks"], r["delta_ks"]))
        if rows
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-for-screen",
        action="store_true",
        help="Wait for initial screening already running in another process.",
    )
    args = parser.parse_args()
    initial = ["core37_e8", "denoised_e8", "v5_latent100_e8", "v5_e16"]
    while args.wait_for_screen and not all(
        (DEST / SCREEN / n / f"fold{i}_metadata.json").exists()
        for n in initial
        for i in range(3)
    ):
        time.sleep(5)
    if not args.wait_for_screen:
        run_candidates(initial, estimator_batch_size=1)
    run_candidates(["tabicl_v5_e8", "tabicl_denoised_e8"])
    summary = summarize(SCREEN)
    chosen = best_eligible(summary)
    if chosen is None:
        # Conditional second stage declared before its scores are observed.
        fallback = dict(
            candidates=["v5_balanced2_e8", "v5_balanced1_e8"],
            reason="No initial candidate passed both KS criteria and 2/3 fold wins.",
            class_sampling="All outer-training positives; independent negative subsets per ensemble member.",
            probability_correction="Population odds divided by sampled-context odds.",
        )
        (DEST / "secondary_screen_plan.json").write_text(
            json.dumps(fallback, indent=2) + "\n"
        )
        run_candidates(fallback["candidates"])
        chosen = best_eligible(summarize(SCREEN))
    if chosen is None:
        chosen = dict(
            name="v5_fullcontext_e8",
            components=[dict(name="v5_fullcontext_e8", weight=1.0)],
            reason="No screen winner. Controlled context-size ablation: 50,000 versus all 56,000 training rows.",
        )
    lock = dict(
        screen_stage=SCREEN,
        confirmation_stage=CONFIRM,
        selected=chosen,
        rule="Positive pooled and mean-fold KS deltas; at least 3/5 fold wins.",
        warning="The new split reuses labels; it is a consistency check, not an untouched holdout.",
    )
    (DEST / "locked_confirmation_recipe.json").write_text(
        json.dumps(lock, indent=2) + "\n"
    )
    print("LOCKED", json.dumps(lock), flush=True)
    names = list(
        dict.fromkeys(["baseline_e8"] + [c["name"] for c in chosen["components"]])
    )
    run_candidates(names, splits=5, seed=31027)
    confirmation = summarize(CONFIRM)
    row = next(r for r in confirmation["comparisons"] if r["name"] == chosen["name"])
    verdict = dict(selected=row, passed=eligible(row, 3), criterion=lock["rule"])
    if not verdict["passed"] and chosen["name"] != "v5_fullcontext_e8":
        (DEST / "locked_confirmation_recipe_blend_failed.json").write_text(
            json.dumps(lock, indent=2) + "\n"
        )
        (DEST / "confirmation_verdict_blend_failed.json").write_text(
            json.dumps(verdict, indent=2) + "\n"
        )
        lock = dict(
            screen_stage=SCREEN,
            confirmation_stage=CONFIRM,
            selected=dict(
                name="v5_fullcontext_e8",
                components=[dict(name="v5_fullcontext_e8", weight=1.0)],
            ),
            reason="Selected 75/25 blend failed the pooled-KS criterion. Test the predefined full-context alternative: baseline cap 50,000 versus all 56,000 outer-training rows.",
            prior_failed_confirmation="confirmation_verdict_blend_failed.json",
            rule="Positive pooled and mean-fold KS deltas; at least 3/5 fold wins.",
            warning="A second candidate is assessed on the same five splits after blend failure. This is repeated local model selection, not an untouched holdout.",
        )
        (DEST / "full_context_followup_plan.json").write_text(
            json.dumps(lock, indent=2) + "\n"
        )
        (DEST / "locked_confirmation_recipe.json").write_text(
            json.dumps(lock, indent=2) + "\n"
        )
        print("LOCKED FULL CONTEXT", json.dumps(lock), flush=True)
        run_candidates(["v5_fullcontext_e8"], splits=5, seed=31027)
        row = next(
            r
            for r in summarize(CONFIRM)["comparisons"]
            if r["name"] == "v5_fullcontext_e8"
        )
        chosen = lock["selected"]
        verdict = dict(
            selected=row,
            passed=eligible(row, 3),
            criterion=lock["rule"],
            warning=lock["warning"],
            prior_failed_confirmation=lock["prior_failed_confirmation"],
        )
    if verdict["passed"]:
        verdict["submission"] = export(CONFIRM, chosen["name"])
    (DEST / "confirmation_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n"
    )
    print("VERDICT", json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
