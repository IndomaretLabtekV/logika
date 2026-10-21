"""Screen one nested fine-tuning recipe and confirm a locked blend on five folds."""

import argparse
import json
import time

from finetune_kumo import DEST, NAME, ensure_frozen, evaluate
from compare_finetune_kumo import compare, export


def arguments(splits, seed):
    return argparse.Namespace(
        splits=splits, fold_seed=seed, folds=None, test=True, force=False
    )


def eligible(row, wins):
    return (
        row["delta_ks"] > 0
        and row["delta_mean_fold_ks"] > 0
        and row["fold_wins"] >= wins
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-pilot", action="store_true")
    args = parser.parse_args()
    if args.wait_for_pilot:
        while not (DEST / "cv3_seed8247" / NAME / "fold0_metadata.json").exists():
            time.sleep(5)
    screen_args = arguments(3, 8247)
    ensure_frozen(screen_args)
    evaluate(screen_args)
    summary = compare("cv3_seed8247")
    rows = [r for r in summary["comparisons"] if eligible(r, 2)]
    if not rows:
        result = dict(
            passed=False,
            stage="screening",
            reason="No fixed fine-tuned candidate passed pooled/mean KS and two fold wins.",
            comparisons=summary["comparisons"],
        )
        (DEST / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
        print("NO SCREEN WINNER", json.dumps(result), flush=True)
        return
    chosen = max(rows, key=lambda r: (r["delta_mean_fold_ks"], r["delta_ks"]))
    lock = dict(
        screen_stage="cv3_seed8247",
        confirmation_stage="cv5_seed31027",
        selected=chosen,
        rule="Positive pooled/mean-fold KS, at least 3/5 fold wins. Inner ROC-AUC chooses checkpoints.",
        warning="Repeated local model selection on the same labels; not an untouched holdout.",
    )
    (DEST / "locked_recipe.json").write_text(json.dumps(lock, indent=2) + "\n")
    print("LOCKED", json.dumps(lock), flush=True)
    confirm_args = arguments(5, 31027)
    ensure_frozen(confirm_args)
    evaluate(confirm_args)
    row = next(
        r
        for r in compare("cv5_seed31027")["comparisons"]
        if r["name"] == chosen["name"]
    )
    result = dict(
        passed=eligible(row, 3), stage="confirmation", selected=row, rule=lock["rule"]
    )
    if result["passed"]:
        result["submission"] = export("cv5_seed31027", row["name"])
    (DEST / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print("VERDICT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
