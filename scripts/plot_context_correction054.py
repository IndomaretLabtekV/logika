"""Plot the completed context-correction candidate against identical baseline folds."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/optimization/goal054"


def main():
    source = (DEST / "independent_verification.json").read_bytes()
    report = json.loads(source)
    base = report["comparisons"][0]
    row = next(
        r
        for r in report["comparisons"]
        if r.get("candidate") == "kumo_offset_l20p003_unweighted_d0p5"
        and r["new_model_weight"] == 1
    )
    x = np.arange(1, 6)
    delta = np.asarray(row["fold_ks"]) - base["fold_ks"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].plot(
        x, base["fold_ks"], "o-", color="#2463a5", label="Baseline Kumo–spline"
    )
    axes[0].plot(x, row["fold_ks"], "o-", color="#ce6500", label="Koreksi konteks")
    axes[0].set(xlabel="Fold", ylabel="KS fold", xticks=x)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].bar(x, delta, color="#ce6500", width=0.55)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        xlabel="Fold", ylabel="Selisih KS fold", xticks=x, ylim=(0, max(delta) * 1.3)
    )
    for fold, difference in zip(x, delta):
        axes[1].text(
            fold, difference + 0.00006, f"+{difference:.5f}", ha="center", fontsize=9
        )
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"KS pooled: {base['ks']:.6f} → {row['ks']:.6f}; target 0,54 belum tercapai",
        fontsize=12,
    )
    fig.supxlabel(
        "5 fold seed 31027. Pemilihan model memakai ulang CV; kenaikan kecil belum menjamin kenaikan public.",
        fontsize=9,
    )
    path = ROOT / "artifacts/figures/goal054_context_correction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    metadata = dict(
        candidate=row["candidate"],
        baseline_pooled_ks=base["ks"],
        candidate_pooled_ks=row["ks"],
        fold_deltas=delta.tolist(),
        source_report_sha256=hashlib.sha256(source).hexdigest(),
        figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (DEST / "context_correction_figure.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
