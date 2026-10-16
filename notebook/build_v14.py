"""v14 notebook: v11 feature cell (eda/features.py + eda/nn_prep.py) + hybrid event-set network (v14_experiment.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v11.ipynb").read_text())
cells = nb["cells"][:13]
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
cells[0] = md("---\n\n# DSC LOGIKA UI 2026: Hybrid Event-Set Network\n\n*Hand features plus learned encoders over raw table rows.*")
cells[4] = md("## Overview and Aim\n\nTests whether a network that reads raw rows (e-commerce items, installments, account months, "
              "bureau reports, previous applications) through attention-pooled set encoders adds information beyond the "
              "hand-built features. Two variants: `hybrid` (hand features + event encoders) and `events` (event encoders only). "
              "Five outer folds (seed 42, sorted application_id), three seeds per fold, each seed early-stopped on its own 10% "
              "inner split of the training fold (AUC). Outer labels never enter fitting or stopping.")
cells[6] = md("Kaggle GPU (T4) and Internet enabled, LOGIKA competition data attached.")
cells[7] = code("%pip install -q polars==2.0.0 scikit-learn pandas pyarrow scipy")
src9 = "".join(cells[9]["source"]).replace("import tabm\n", "").replace("import rtdl_num_embeddings\n", "")
cells[9] = code(src9)
cells.append(md("---\n\n# Experiment <a name=\"4\"></a>\n\n---"))
cells.append(code((ROOT / "v14_experiment.py").read_text()))
cells.append(code("main_v14(build_all, nn_prep)"))
nb["cells"] = cells
for i, c in enumerate(nb["cells"]):
    c["id"] = f"v14-{i:03d}"
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        s = "".join(c["source"])
        if not s.startswith("%"):
            compile(s, f"v14-cell-{i}", "exec")
out = ROOT / "v14.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", len(cells), "cells ->", out)
