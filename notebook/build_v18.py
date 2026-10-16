"""v18 notebook: v10 feature cell (eda/features.py) + ranking / level-2 stacker experiment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / "v10.ipynb").read_text())
md = lambda s: dict(cell_type="markdown", metadata={}, source=s)  # noqa: E731
code = lambda s: dict(cell_type="code", metadata={}, source=s, outputs=[], execution_count=None)  # noqa: E731
cells = [md("---\n\n# DSC LOGIKA UI 2026: Inductive-Bias Variants\n\n*v18 experiment.*"),
         md("LightGBM with linear leaves, LightGBM with monotone constraints (signs from training-fold Spearman), and "
            "ExtraTrees on the hand features. Five outer folds (seed 42); early stopping on a 1/7 inner split."),
         code("%pip install -q lightgbm==4.6.0 xgboost scikit-learn polars==2.0.0 pandas pyarrow scipy"),
         code("import hashlib\nimport json\nimport os\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport polars as pl"),
         nb["cells"][12],
         code((ROOT / "v18_experiment.py").read_text()),
         code("main_v18(build_all)")]
for i, c in enumerate(cells):
    c["id"] = f"v18-{i:03d}"
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        s = "".join(c["source"])
        if not s.startswith("%"):
            compile(s, f"v18-{i}", "exec")
nb["cells"] = cells
(ROOT / "v18.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote v18.ipynb", len(cells))
