"""Monitor one existing relational run, then audit and execute its review notebook.

This never starts a duplicate model process or changes GitHub/Kaggle state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/goal054"
NAME = "kumo_relational_v5_e8_ctx50000_h1_readout_b8192"


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def write_status(value):
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = DEST / "relational_queue_status.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def completed_folds():
    return [
        i
        for i in range(5)
        if (DEST / "cv5_seed31027" / NAME / f"fold{i}_metadata.json").exists()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--notebook-python", default=sys.executable)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    assert identity is not None, "The model process is already terminal."
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes()
    assert any(
        name in command
        for name in [
            b"scripts/research_relational054.py",
            b"scripts/run_relational054.py",
        ]
    )
    assert b"--preflight" not in command
    state = dict(
        status="training",
        model_pid=args.pid,
        process_start_ticks=identity,
        monitor_pid=os.getpid(),
        candidate=NAME,
        target_pooled_ks=0.54,
        kaggle_upload=False,
    )
    try:
        while process_identity(args.pid) == identity:
            state["completed_folds"] = completed_folds()
            write_status(state)
            time.sleep(30)
        state["completed_folds"] = completed_folds()
        assert (
            len(state["completed_folds"]) == 5
        ), "Model terminated before five folds completed; inspect relational_cv5.log."
        for label, python, script in [
            ("blending", sys.executable, "blend_relational054.py"),
            ("auditing", sys.executable, "report_goal054.py"),
            ("bootstrap", sys.executable, "uncertainty_goal054.py"),
            ("notebook", args.notebook_python, "build_goal054_notebook.py"),
        ]:
            state["status"] = label
            write_status(state)
            command = [python, "-u", str(ROOT / "scripts" / script)]
            if label == "notebook":
                command.append("--execute")
            subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads((DEST / "independent_verification.json").read_text())
        best = max(report["comparisons"], key=lambda row: row["ks"])
        state.update(
            status=(
                "finished_target_met"
                if report["target_reached"]
                else "finished_below_target"
            ),
            best_completed=best,
            target_reached=report["target_reached"],
            agent_review_required_before_goal_completion=True,
        )
        progress = json.loads((DEST / "progress.json").read_text())
        progress.update(
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            best_completed_pooled_ks=best["ks"],
            target_reached=report["target_reached"],
            completed_fold_counts=report["completed_fold_counts"],
            active_queues=[],
            notebook_status="Executed after full relational evaluation",
            status=state["status"],
        )
        (DEST / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        write_status(state)
    except Exception as error:
        state.update(
            status="failed", error=repr(error), completed_folds=completed_folds()
        )
        write_status(state)
        raise


if __name__ == "__main__":
    main()
