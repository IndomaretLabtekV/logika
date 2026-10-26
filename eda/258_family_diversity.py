"""Diversity of merge families: Spearman of per-cohort OOF ranks and of test vectors, seed-42 folds.
Usage: cd eda && ../.venv/bin/python 258_family_diversity.py"""
import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import ROOT

e250 = importlib.import_module("250_merge_blend_existing")
ids, y, tids, folds, fam_oof, fam_test, _ = e250.families(str(ROOT / "results/dsc0152/agas/v66/output/preds_v66s42.npz"))
names = [m for m in ["T", "C", "K64", "D", "R2", "KD", "KF"] if m in fam_oof]
oof = np.column_stack([fam_oof[m] for m in names])
r_oof = spearmanr(oof).statistic
r_test = spearmanr(np.column_stack([fam_test[m] for m in names if m in fam_test])).statistic
print("OOF Spearman\n", "      " + " ".join(f"{n:>6s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>6s}", " ".join(f"{x:6.3f}" for x in r_oof[i]))
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(r_oof, vmin=0.9, vmax=1)
ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{r_oof[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w")
fig.colorbar(im); ax.set_title("OOF Spearman, merge families"); fig.tight_layout()
fig.savefig(ROOT / "eda/out/258_family_diversity.png", dpi=120)
(ROOT / "eda/out/258_family_diversity.json").write_text(json.dumps({"names": names, "oof": r_oof.tolist(), "test": r_test.tolist()}, indent=1))
